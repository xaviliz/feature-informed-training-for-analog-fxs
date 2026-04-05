import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSADGLoss(nn.Module):
    """
    Amplitude Dependent Gain (ADG) loss based on RMS envelope estimation.

    ADG(t) = env(signal, t) / (env(input, t) + ε)

    Loss = MSE(ADG_pred, ADG_target) / E[ADG_target²]
    """

    def __init__(
        self,
        sample_rate: int,
        window_ms: float = 2.0,
        log_domain: bool = False,
        floor_db: float = -90.0,
        eps: float = torch.finfo(torch.float32).eps,
        use_mae: bool = False,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.log_domain = log_domain
        self.floor_db = floor_db
        self.window_size = time_to_window(window_ms, sample_rate)
        self.sigma = self.window_size / 6
        self.use_mae = use_mae
        if self.use_mae:
            self.loss = F.l1_loss
        else:
            self.loss = NormalizedMSELoss()

    def _make_gaussian_kernel(
        self, window_size: int, sigma: float, device: str, dtype: torch.dtype
    ) -> torch.Tensor:
        half = window_size // 2
        t = torch.arange(-half, half + 1, device=device, dtype=dtype)
        g = torch.exp(-0.5 * (t / sigma) ** 2)
        g = g / g.sum()  # normalise → weighted average
        return g.reshape(1, 1, -1)

    def _envelope(self, x: torch.Tensor) -> torch.Tensor:
        x2 = x**2
        kernel = self._make_gaussian_kernel(
            self.window_size, self.sigma, x.device, x.dtype
        )

        # kernel = torch.ones(1, 1, self.window_size,
        #                    device=x.device, dtype=x.dtype) / self.window_size
        shape = x2.shape
        x2 = x2.reshape(-1, 1, shape[-2] if x.ndim == 3 else shape[-1])
        # causal padding = (window_size - 1) zeros on the LEFT, 0 on the right
        # x2_padded = F.pad(x2, (self.window_size - 1, 0))
        # out = F.conv1d(x2_padded, kernel)
        out = F.conv1d(x2, kernel, padding=self.window_size // 2)
        out = out[..., : x2.shape[-1]]
        return torch.sqrt(out + self.eps).reshape(shape)

    def _adg(self, signal: torch.Tensor, input_env: torch.Tensor) -> torch.Tensor:
        adg = self._envelope(signal) / (input_env + self.eps)
        if self.log_domain:
            floor = 10.0 ** (self.floor_db / 20.0)
            adg = 20.0 * torch.log10(adg.clamp(min=floor))
        return adg

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, input: torch.Tensor
    ) -> torch.float32:
        input_env = self._envelope(input)
        adg_target = self._adg(target, input_env)
        adg_pred = self._adg(pred, input_env)

        # Estimate errors
        if self.use_mae:
            error = self.loss(adg_pred, adg_target)
        else:
            error = self.loss.forward(adg_pred, adg_target)

        return error


def rms_envelope(
    x: torch.Tensor, window_size: int, epsilon: float = torch.finfo(torch.float32).eps
) -> torch.Tensor:
    # x: (..., T)
    x2 = x**2

    kernel = torch.ones(1, 1, window_size, device=x.device, dtype=x.dtype) / window_size

    # reshape to (N, 1, T)
    x2 = x2.reshape(-1, 1, x2.shape[-1])

    rms = F.conv1d(x2, kernel, padding=window_size // 2)

    rms = torch.sqrt(rms + epsilon)
    return rms.reshape(*x.shape)


def peak_conv_envelope(x: torch.Tensor, window_size: int) -> torch.Tensor:
    x = torch.abs(x)

    kernel = torch.ones(1, 1, window_size, device=x.device, dtype=x.dtype) / window_size

    x = x.reshape(-1, 1, x.shape[-1])

    env = F.max_pool1d(x, kernel_size=window_size, stride=1, padding=window_size // 2)

    return env.reshape(*x.shape[:-2], -1)


def min_max_normalize(x: torch.Tensor, epsilon: torch.float32) -> torch.Tensor:
    return (x - x.min()) / (x.max() - x.min() + epsilon)


def max_normalize(x: torch.Tensor) -> torch.Tensor:
    return x / torch.max(x.abs())


def hilbert_envelope(
    x: torch.Tensor,
    time_dim: int = -2,
    normalize: bool = False,
) -> torch.Tensor:
    # x: (..., T)
    X = torch.fft.fft(x, dim=time_dim)

    N = X.shape[time_dim]
    h = torch.zeros(N, device=x.device, dtype=x.dtype)

    if N % 2 == 0:
        h[0] = 1
        h[N // 2] = 1
        h[1 : N // 2] = 2
    else:
        h[0] = 1
        h[1 : (N + 1) // 2] = 2

    # reshape h for broadcasting
    shape = [1] * x.ndim
    shape[time_dim] = N
    h = h.reshape(shape)

    analytic = torch.fft.ifft(X * h, dim=time_dim)
    out = torch.abs(analytic)

    if normalize:
        out = max_normalize(out)

    return out


@torch.jit.script
def causal_envelope_triplet(
    pred: torch.Tensor,
    target: torch.Tensor,
    input: torch.Tensor,
    sample_rate: int,
    attack_time_ms: float,
    release_time_ms: float,
):  # -> tuple:
    attack = attack_time_ms / 1000.0
    release = release_time_ms / 1000.0

    ga: float = 0.0
    gr: float = 0.0

    if attack > 0.0:
        ga = math.exp(-1.0 / (sample_rate * attack))

    if release > 0.0:
        gr = math.exp(-1.0 / (sample_rate * release))

    B, T, C = pred.shape

    env_pred = torch.zeros_like(pred)
    env_target = torch.zeros_like(target)
    env_input = torch.zeros_like(input)

    state_pred = torch.zeros(B, C, device=pred.device, dtype=pred.dtype)
    state_target = torch.zeros(B, C, device=pred.device, dtype=pred.dtype)
    state_input = torch.zeros(B, C, device=pred.device, dtype=pred.dtype)

    for t in range(T):
        sample_p = pred[:, t, :].abs()
        sample_t = target[:, t, :].abs()
        sample_i = input[:, t, :].abs()

        mask_p = sample_p > state_pred
        mask_t = sample_t > state_target
        mask_i = sample_i > state_input

        state_pred = torch.where(
            mask_p,
            (1 - ga) * sample_p + ga * state_pred,
            (1 - gr) * sample_p + gr * state_pred,
        )

        state_target = torch.where(
            mask_t,
            (1 - ga) * sample_t + ga * state_target,
            (1 - gr) * sample_t + gr * state_target,
        )

        state_input = torch.where(
            mask_i,
            (1 - ga) * sample_i + ga * state_input,
            (1 - gr) * sample_i + gr * state_input,
        )

        env_pred[:, t, :] = state_pred
        env_target[:, t, :] = state_target
        env_input[:, t, :] = state_input

    return env_pred, env_target, env_input


def causal_envelope_cpu(
    x: torch.Tensor, sample_rate: int, attack_time_ms: float, release_time_ms: float
) -> torch.Tensor:

    x_cpu = x.cpu()

    attack = attack_time_ms / 1000.0
    release = release_time_ms / 1000.0

    g_a = math.exp(-1.0 / (sample_rate * attack)) if attack > 0 else 0.0
    g_r = math.exp(-1.0 / (sample_rate * release)) if release > 0 else 0.0

    B, T, C = x_cpu.shape
    env = torch.zeros_like(x_cpu)

    state = torch.zeros(B, C)

    for t in range(T):
        sample = x_cpu[:, t, :].abs()
        attack_mask = sample > state

        state = torch.where(
            attack_mask,
            (1 - g_a) * sample + g_a * state,
            (1 - g_r) * sample + g_r * state,
        )

        env[:, t, :] = state

    return env.to(x.device)


@torch.jit.script
def causal_envelope_fast(
    x: torch.Tensor, sample_rate: int, attack_time_ms: float, release_time_ms: float
) -> torch.Tensor:

    attack = attack_time_ms / 1000.0
    release = release_time_ms / 1000.0

    ga = math.exp(-1.0 / (sample_rate * attack)) if attack > 0 else 0.0
    gr = math.exp(-1.0 / (sample_rate * release)) if release > 0 else 0.0

    B, T, C = x.shape
    env = torch.zeros_like(x)
    state = torch.zeros(B, C, device=x.device, dtype=x.dtype)

    for t in range(T):
        sample = torch.abs(x[:, t, :])
        attack_mask = sample > state

        state = torch.where(
            attack_mask, (1 - ga) * sample + ga * state, (1 - gr) * sample + gr * state
        )

        env[:, t, :] = state

    return env


@torch.jit.script
def causal_envelope_jit(
    x: torch.Tensor,
    sample_rate: int = 48000,
    attack_time_ms: float = 5,
    release_time_ms: float = 30,
) -> torch.Tensor:
    """
    Fast, causal, differentiable envelope follower.

    Args:
        x: input tensor, shape (B, T, C)
        attack_time_ms: attack time in milliseconds
        release_time_ms: release time in milliseconds

    Returns:
        env: tensor of same shape as x, differentiable
    """

    attack = attack_time_ms / 1000.0
    release = release_time_ms / 1000.0

    ga = math.exp(-1.0 / (sample_rate * attack)) if attack > 0 else 0.0
    gr = math.exp(-1.0 / (sample_rate * release)) if release > 0 else 0.0

    B, T, C = x.shape

    # Permute to (B*C, T) for easier processing
    x_flat = x.permute(0, 2, 1).reshape(B * C, T)
    env = torch.zeros_like(x_flat)

    # Initial state = 0
    state = torch.zeros(B * C, device=x.device, dtype=x.dtype)

    # Loop over time (cannot be fully removed due to causality)
    for t in range(T):
        sample = x_flat[:, t].abs()  # rectification

        # attack / release update (vectorized across batch*channel)
        attack_mask = sample > state
        state = torch.where(
            attack_mask, (1 - ga) * sample + ga * state, (1 - gr) * sample + gr * state
        )

        env[:, t] = state

    # Reshape back to (B, T, C)
    env = env.reshape(B, C, T).permute(0, 2, 1)
    return env


# compile once, globally
# causal_envelope_jit = torch.compile(causal_envelope_jit)


def causal_envelope(
    x: torch.Tensor,
    sample_rate: int = 48000,
    attack_time_ms: float = 5,
    release_time_ms: float = 30,
) -> torch.Tensor:
    """
    Fast, causal, differentiable envelope follower.

    Args:
        x: input tensor, shape (B, T, C)
        attack_time_ms: attack time in milliseconds
        release_time_ms: release time in milliseconds

    Returns:
        env: tensor of same shape as x, differentiable
    """

    attack = attack_time_ms / 1000.0
    release = release_time_ms / 1000.0

    ga = math.exp(-1.0 / (sample_rate * attack)) if attack > 0 else 0.0
    gr = math.exp(-1.0 / (sample_rate * release)) if release > 0 else 0.0

    B, T, C = x.shape

    # Permute to (B*C, T) for easier processing
    x_flat = x.permute(0, 2, 1).reshape(B * C, T)
    env = torch.zeros_like(x_flat)

    # Initial state = 0
    state = torch.zeros(B * C, device=x.device, dtype=x.dtype)

    # Loop over time (cannot be fully removed due to causality)
    for t in range(T):
        sample = x_flat[:, t].abs()  # rectification

        # attack / release update (vectorized across batch*channel)
        attack_mask = sample > state
        state = torch.where(
            attack_mask, (1 - ga) * sample + ga * state, (1 - gr) * sample + gr * state
        )

        env[:, t] = state

    # Reshape back to (B, T, C)
    env = env.reshape(B, C, T).permute(0, 2, 1)
    return env


def moving_average_energy(x: torch.Tensor, window_size: int) -> torch.Tensor:

    if window_size % 2 == 0:
        raise ValueError("window_size must be odd to preserve length")

    B, T, C = x.shape

    x2 = x**2

    kernel = torch.ones(1, 1, window_size, device=x.device, dtype=x.dtype) / window_size

    # (B, T, C) → (B, C, T)
    x2 = x2.permute(0, 2, 1)

    # (B, C, T) → (B*C, 1, T)
    x2 = x2.reshape(B * C, 1, T)

    energy = F.conv1d(x2, kernel, padding=window_size // 2)

    # Back to (B, T, C)
    energy = energy.reshape(B, C, T).permute(0, 2, 1)

    return energy


def time_to_window(time_ms: float, sample_rate: int, scale: float = 1.0) -> int:
    return int(scale * time_ms * 1e-3 * sample_rate)


class EnvelopeFollower(torch.nn.Module):
    def __init__(
        self,
        sample_rate: int,
        attack_time_ms: float,
        release_time_ms: float,
        apply_rectification: bool = True,
    ) -> None:
        super().__init__()

        self.apply_rectification = apply_rectification

        attack = attack_time_ms / 1000.0
        release = release_time_ms / 1000.0

        self.ga = math.exp(-1.0 / (sample_rate * attack)) if attack > 0 else 0.0
        self.gr = math.exp(-1.0 / (sample_rate * release)) if release > 0 else 0.0

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Differentiable causal envelope.
        signal shape: (B, T, C)
        Returns: envelope of same shape
        """
        if self.apply_rectification:
            signal = signal.abs()

        B, T, C = signal.shape

        # Flatten batch × channel for vectorized time processing
        x_flat = signal.permute(0, 2, 1).reshape(B * C, T)  # (B*C, T)
        envelope_flat = torch.zeros_like(x_flat)

        # Local state tensor (fully differentiable)
        state = torch.zeros(B * C, device=signal.device, dtype=signal.dtype)

        for t in range(T):
            x_t = x_flat[:, t]

            # attack / release update
            attack_mask = x_t > state
            state = torch.where(
                attack_mask,
                (1.0 - self.ga) * x_t + self.ga * state,
                (1.0 - self.gr) * x_t + self.gr * state,
            )

            envelope_flat[:, t] = state

        # Reshape back to (B, T, C)
        envelope = envelope_flat.reshape(B, C, T).permute(0, 2, 1)
        return envelope


def to_dbfs(x, floor_db=-120.0) -> torch.Tensor:
    eps = 10.0 ** (floor_db / 20.0)
    return 20.0 * torch.log10(torch.clamp(x, min=eps))


def to_dbw(x, floor_db=-60.0) -> torch.Tensor:
    eps = 10.0 ** (floor_db / 10.0)
    return 10.0 * torch.log10(torch.clamp(x, min=eps))


class CausalADGLoss(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        dtype=torch.float32,
        attack_time_ms: float = 5,
        release_time_ms: float = 30,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.dtype = dtype
        self.epsilon = torch.finfo(self.dtype).eps
        self.attack_time_ms = attack_time_ms
        self.release_time_ms = release_time_ms
        self.normalized_mse = NormalizedMSELoss(eps=self.epsilon)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, input: torch.Tensor
    ) -> torch.float32:

        input = input.requires_grad_()

        # Downsampling envelope by factor 4
        input_ds = input[:, ::4, :]
        target_ds = target[:, ::4, :]
        pred_ds = pred[:, ::4, :]

        # Estimate input envelope
        input_envelope_ds = causal_envelope_jit(
            input_ds,
            sample_rate=self.sample_rate,
            attack_time_ms=self.attack_time_ms,
            release_time_ms=self.release_time_ms,
        )

        # Estimate target envelope
        target_envelope_ds = causal_envelope_jit(
            target_ds,
            sample_rate=self.sample_rate,
            attack_time_ms=self.attack_time_ms,
            release_time_ms=self.release_time_ms,
        )

        # Estimate predicted envelope
        pred_envelope_ds = causal_envelope_jit(
            pred_ds,
            sample_rate=self.sample_rate,
            attack_time_ms=self.attack_time_ms,
            release_time_ms=self.release_time_ms,
        )

        # Upsampling envelopes
        input_envelope = torch.repeat_interleave(input_envelope_ds, repeats=4, dim=1)
        target_envelope = torch.repeat_interleave(target_envelope_ds, repeats=4, dim=1)
        pred_envelope = torch.repeat_interleave(pred_envelope_ds, repeats=4, dim=1)

        # Estimate in-tg-adg
        in_target_adg = target_envelope / (input_envelope + self.epsilon)

        # Estimate in-pred-adg
        in_out_adg = pred_envelope / (input_envelope + self.epsilon)

        # Estimate MSE
        error = self.normalized_mse.forward(in_target_adg, in_out_adg)

        return error


class PeakADGLoss(nn.Module):
    def __init__(
        self, sample_rate: int, dtype=torch.float32, use_mae: bool = False
    ) -> None:
        super().__init__()
        self.epsilon = torch.finfo(torch.float32).eps
        self.window_size = time_to_window(5, sample_rate) | 1  # force odd
        self.use_mae = use_mae
        if use_mae:
            self.loss = F.l1_loss
            print("Using MAE loss for estimating ADG errors...")
        else:
            self.loss = NormalizedMSELoss(eps=self.epsilon)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, input: torch.Tensor
    ) -> torch.float32:

        pred = pred.requires_grad_()
        target = target.requires_grad_()
        input = input.requires_grad_()

        # Estimate input envelope
        input_envelope = peak_conv_envelope(input, self.window_size)

        # Estimate target envelope
        target_envelope = peak_conv_envelope(target, self.window_size)

        # Estimate predicted envelope
        pred_envelope = peak_conv_envelope(pred, self.window_size)

        # Estimate in-tg-adg
        in_target_adg = target_envelope / (input_envelope + self.epsilon)

        # Estimate in-pred-adg
        in_out_adg = pred_envelope / (input_envelope + self.epsilon)

        # Estimate errors
        if self.use_mae:
            error = self.loss(in_target_adg, in_out_adg)
        else:
            error = self.loss.forward(in_target_adg, in_out_adg)

        return error


class HilbertADGLoss(nn.Module):
    def __init__(self, use_mae: bool = False) -> None:
        super().__init__()
        self.epsilon = torch.finfo(torch.float32).eps
        self.use_mae = use_mae
        if use_mae:
            self.loss = F.l1_loss
            print("Using MAE loss for estimating ADG errors...")
        else:
            self.loss = NormalizedMSELoss(eps=self.epsilon)
        self.time_dim = 1

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, input: torch.Tensor
    ) -> torch.float32:

        pred = pred.requires_grad_()
        target = target.requires_grad_()
        input = input.requires_grad_()

        # Estimate input envelope
        input_envelope = hilbert_envelope(input, self.time_dim)

        # Estimate target envelope
        target_envelope = hilbert_envelope(target, self.time_dim)

        # Estimate predicted envelope
        pred_envelope = hilbert_envelope(pred, self.time_dim)

        # Estimate in-tg-adg
        in_target_adg = target_envelope / (input_envelope + self.epsilon)

        # Estimate in-pred-adg
        in_out_adg = pred_envelope / (input_envelope + self.epsilon)

        # Estimate errors
        if self.use_mae:
            error = self.loss(in_target_adg, in_out_adg)
        else:
            error = self.loss.forward(in_target_adg, in_out_adg)

        return error


class ADGLoss(nn.Module):
    def __init__(self, sample_rate: int) -> None:
        super().__init__()
        self.epsilon = torch.finfo(torch.float32).eps
        self.envelope_follower = EnvelopeFollower(sample_rate, 5, 30).cuda()
        self.normalized_mse = NormalizedMSELoss(eps=self.epsilon)
        self.window_size = time_to_window(5, sample_rate) | 1  # force odd

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, input: torch.Tensor
    ) -> torch.float32:

        pred = pred.requires_grad_()
        target = target.requires_grad_()
        input = input.requires_grad_()

        # Estimate input envelope
        # input_envelope = self.envelope_follower(input)
        # input_envelope = hilbert_envelope(input, 1)
        # input_envelope = rms_envelope(input, self.window_size)
        # input_envelope = peak_conv_envelope(input, self.window_size)
        # input_envelope = causal_envelope(input)
        input_envelope = moving_average_energy(input, self.window_size)

        # Estimate target envelope
        # target_envelope = self.envelope_follower(target)
        # target_envelope = hilbert_envelope(target, 1)
        # target_envelope = rms_envelope(target, self.window_size)
        # target_envelope = peak_conv_envelope(target, self.window_size)
        # target_envelope = causal_envelope(target)
        target_envelope = moving_average_energy(target, self.window_size)

        # Estimate predicted envelope
        # pred_envelope = self.envelope_follower(pred)
        # pred_envelope = hilbert_envelope(pred, 1)
        # pred_envelope = rms_envelope(pred, self.window_size)
        # pred_envelope = peak_conv_envelope(pred, self.window_size)
        # pred_envelope = causal_envelope(pred)
        pred_envelope = moving_average_energy(pred, self.window_size)

        # Estimate in-tg-adg
        in_target_adg = target_envelope / (input_envelope + self.epsilon)

        # Estimate in-pred-adg
        in_out_adg = pred_envelope / (input_envelope + self.epsilon)

        # Estimate MSE
        error = self.normalized_mse.forward(in_target_adg, in_out_adg)

        return error


class ESRLoss(nn.Module):
    """
    Error-to-Signal Ratio Loss
    Normalized squared error between prediction and target
    """

    def __init__(self, epsilon: float = 1e-8) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: predicted signal [batch, channels, time] or [batch, time]
            target: target signal [batch, channels, time] or [batch, time]
        Returns:
            ESR loss value
        """

        if len(pred.shape) == 3:
            pred = pred.unsqueeze(-1)
            target = target.unsqueeze(-1)

        numerator = torch.sum((target - pred) ** 2, dim=-1)
        denominator = torch.sum(target**2, dim=-1) + self.epsilon
        esr = numerator / denominator
        return esr.mean()


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-Resolution STFT Loss
    Computes STFT loss at multiple resolutions
    """

    def __init__(
        self,
        fft_sizes: list = [2048, 1024, 512],
        hop_sizes: list = [512, 256, 128],
        win_lengths: list = [2048, 1024, 512],
        window: str = "hann",
        w_sc: float = 1.0,
        w_log_mag: float = 1.0,
        w_lin_mag: float = 0.0,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__()

        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths), (
            "All parameter lists must have the same length"
        )

        self.stft_losses = nn.ModuleList()
        for fft_size, hop_size, win_length in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses.append(
                STFTLoss(
                    fft_size=fft_size,
                    hop_size=hop_size,
                    win_length=win_length,
                    window=window,
                    w_sc=w_sc,
                    w_log_mag=w_log_mag,
                    w_lin_mag=w_lin_mag,
                    epsilon=epsilon,
                )
            )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.float32:
        """
        Args:
            pred: predicted signal [batch, channels, time] or [batch, time]
            target: target signal [batch, channels, time] or [batch, time]
        """
        if len(pred.shape) == 3:
            pred = pred.squeeze(-1)
            target = target.squeeze(-1)

        total_loss = 0.0
        for stft_loss in self.stft_losses:
            total_loss += stft_loss(pred, target)

        # Average across resolutions
        return total_loss / len(self.stft_losses)


class STFTLoss(nn.Module):
    """
    Single-resolution STFT Loss with spectral convergence and log magnitude
    """

    def __init__(
        self,
        fft_size: int = 2048,
        hop_size: int = 512,
        win_length: int = 2048,
        window: str = "hann",
        w_sc: float = 1.0,
        w_log_mag: float = 1.0,
        w_lin_mag: float = 0.0,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.w_sc = w_sc
        self.w_log_mag = w_log_mag
        self.w_lin_mag = w_lin_mag
        self.epsilon = epsilon

        # Register window buffer
        self.register_buffer("window", self._get_window(window, win_length))

    def _get_window(self, window_type: str, win_length: int) -> torch.Tensor:
        if window_type == "hann":
            return torch.hann_window(win_length)
        elif window_type == "hamming":
            return torch.hamming_window(win_length)
        else:
            raise ValueError(f"Unknown window type: {window_type}")

    def _stft(self, x: torch.Tensor) -> torch.Tensor:
        """Compute STFT magnitude"""
        # x: [batch, channels, time] or [batch, time]
        if x.dim() == 3:
            batch, channels, time = x.shape
            x = x.reshape(batch * channels, time)

        stft = torch.stft(
            x,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
            center=True,
            normalized=False,
        )
        magnitude = torch.abs(stft)
        return magnitude

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.float32:
        """
        Args:
            pred: predicted signal [batch, channels, time] or [batch, time]
            target: target signal [batch, channels, time] or [batch, time]
        """

        if len(pred.shape) == 3:
            pred = pred.squeeze(-1)
            target = target.squeeze(-1)

        pred_mag = self._stft(pred)
        target_mag = self._stft(target)

        # Spectral convergence
        sc_loss = torch.norm(target_mag - pred_mag, p="fro") / (
            torch.norm(target_mag, p="fro") + self.epsilon
        )

        # Log magnitude loss
        log_mag_loss = F.l1_loss(
            torch.log(target_mag + self.epsilon), torch.log(pred_mag + self.epsilon)
        )

        # Linear magnitude loss
        lin_mag_loss = F.l1_loss(target_mag, pred_mag)

        total_loss = (
            self.w_sc * sc_loss
            + self.w_log_mag * log_mag_loss
            + self.w_lin_mag * lin_mag_loss
        )
        return total_loss


class NormalizedMSELoss(nn.Module):
    """
    Custom MSE loss normalized by the target values.

    Loss = MSE(pred, target) / |target|^2
    where MSE = mean((pred - target)^2)
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super(NormalizedMSELoss, self).__init__()
        self.eps = eps  # Small epsilon to avoid division by zero

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        if len(pred.shape) == 3:
            pred = pred.squeeze(-1)
            target = target.squeeze(-1)

        # Compute standard MSE
        mse = torch.mean((pred - target) ** 2)

        # Normalize by target magnitude squared
        target_norm_sq = torch.mean(target**2)

        # Add epsilon to avoid division by zero
        normalized_mse = mse / (target_norm_sq + self.eps)

        return normalized_mse


class SpectralFluxLoss(nn.Module):
    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: int = None,
        window: str = "hann",
    ) -> None:
        super(SpectralFluxLoss, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.register_buffer("window", torch.hann_window(self.win_length))

    def compute_spectral_flux(self, x: torch.Tensor) -> torch.Tensor:
        # Compute STFT
        stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )

        # Compute magnitude spectrogram
        mag = torch.abs(stft)  # [batch, freq_bins, time_frames]

        # Compute spectral flux (difference between consecutive frames)
        flux = torch.diff(mag, dim=-1)  # difference along time axis
        flux = torch.clamp(
            flux, min=0
        )  # only positive differences (onset detection variant)

        return flux

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        if len(pred.shape) == 3:
            pred = pred.squeeze(-1)
            target = target.squeeze(-1)

        # Compute spectral flux for both signals
        flux_pred = self.compute_spectral_flux(pred)
        flux_target = self.compute_spectral_flux(target)

        # Compute loss (L1 or L2)
        loss = F.l1_loss(flux_pred, flux_target)

        return loss
