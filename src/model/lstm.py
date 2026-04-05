import torch
import torch.nn as nn
import torch.nn.functional as F

from model.film import FiLM
from model.glu import GLU


class LSTM_film(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        conditioning_dim: int,
        batch_size: int,
        order: int,
        task_embedding_dim: int,
        device: int,
    ) -> None:
        super(LSTM_film, self).__init__()

        self.hidden_size = hidden_size
        self.task_embedding_dim = task_embedding_dim
        self.batch_size = batch_size
        self.device = device
        self.order = order
        self.model_type = "LSTM_film"

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.film = FiLM(hidden_size, conditioning_dim, use_layer_norm=1, order=order)
        self.glu = GLU(hidden_size)
        # Linear output layer
        self.linear = nn.Linear(hidden_size, output_size)
        self.reset_hidden_states()

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        c: torch.Tensor,
        optimizer: torch.optim,
        loss_id: str,
        criterion: torch.nn.functional = F.l1_loss,
        power_log: bool = False,
    ) -> float:
        """Single training step."""
        optimizer.zero_grad()

        # Forward pass
        output = self.forward(x, c, detach_states=True)

        ## Loss Preprocessing

        # Power log domain
        if power_log:
            output = 10 * torch.log10(torch.abs(output) + 1e-8)
            y = 10 * torch.log10(torch.abs(y) + 1e-8)
            # loss = torch.sqrt()

        # Compute loss
        if not "ADG" in loss_id:
            loss = criterion(output, y)
        else:
            loss = criterion(output, y, x)

        # Power log normalization
        if power_log:
            loss /= 8.32

        # Backward pass
        loss.backward()
        # Gradient clipping (important for RNNs)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        optimizer.step()

        return loss.item()

    def val_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        c: torch.Tensor,
        loss_id: str,
        criterion: torch.nn.functional = F.l1_loss,
    ) -> float:
        """Single val step."""

        output = self.forward(x, c, detach_states=True)

        # Calculate loss and update model weights
        if not "ADG" in loss_id:
            loss = criterion(output, y)
        else:
            loss = criterion(output, y, x)

        return loss.item()

    def reset_hidden_states(self) -> None:
        self.hidden_states = (
            torch.zeros(1, self.batch_size, self.hidden_size, device=self.device),
            torch.zeros(1, self.batch_size, self.hidden_size, device=self.device),
        )

    def forward(
        self, x: torch.Tensor, c: torch.Tensor, detach_states: bool = True
    ) -> torch.Tensor:

        # Detach hidden states to prevent gradients flowing through entire history
        if detach_states and self.hidden_states is not None:
            self.hidden_states = (
                self.hidden_states[0].detach(),
                self.hidden_states[1].detach(),
            )

        # x shape: (batch_size, seq_len, input_size)
        # LSTM forward pass
        lstm_out, hidden_states = self.lstm(x, self.hidden_states)
        self.hidden_states = hidden_states

        # lstm_out shape: (batch_size, seq_len, hidden_size)
        lstm_out = self.film(lstm_out, c)
        lstm_out = self.glu(lstm_out)
        output = self.linear(lstm_out)  # Shape: (batch_size, seq_len, hidden_size)

        return output
