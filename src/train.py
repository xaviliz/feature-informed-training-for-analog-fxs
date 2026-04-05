from pathlib import Path
import shutil
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Literal, Optional

from checkpoint_manager import CheckpointManager
from extract_features import EssentiaExtractor
from dataloader import DataGeneratorPickles
from losses import (
    SpectralFluxLoss,
    ESRLoss,
    MultiResolutionSTFTLoss,
    NormalizedMSELoss,
    ADGLoss,
    HilbertADGLoss,
    PeakADGLoss,
    CausalADGLoss,
    RMSADGLoss,
)
from model.lstm import LSTM_film
from utils.train import (
    save_audio_files,
    save_losses,
    plot_losses,
    save_feature_files,
    write_file,
)
from utils.dir import root_dir, generated_dir


script_path = Path(__file__).resolve()


def training(
    model: str,
    conditioning_type: str,
    dataset_name: str,
    data_dir: str,
    embedding_dim: int,
    feature: Optional[str],
    use_multiband: bool,
    extractor: str,
    predict_feature: bool,
    fs: int = 48000,
    epochs: int = 1,
    seq_len: int = 2048,
    hidden_size: int = 32,
    input_size: int = 1,
    order: int = 1,
    loss: Optional[str] = "MSE",
    lr: int = 1e-4,
    from_scratch: bool = True,
    lim_for_testing: bool = False,
    extract_in_loading: bool = False,
) -> int:

    data_dir = root_dir / data_dir

    model_name = (
        model
        + "_"
        + conditioning_type
        + "_"
        + dataset_name
        + "_E_"
        + str(embedding_dim)
        + "_F_"
        + str(feature)
        + "_PF_"
        + str(predict_feature)
        + "_LOSS_"
        + str(loss)
        + "_use_multiband_"
        + str(use_multiband)
    )
    model_path = generated_dir / script_path.stem / model_name

    if from_scratch and model_path.exists():
        shutil.rmtree(model_path)

    batch_size = 1

    print(f"cuda available: {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # Display GPU index
    if str(device) == "cuda":
        for i in range(torch.cuda.device_count()):
            print(torch.cuda.get_device_properties(i).name)

    dataset = DataGeneratorPickles(
        data_dir=data_dir,
        filename=dataset_name + "_train.pickle",
        mini_batch_size=seq_len,
        batch_size=batch_size,
        set="train",
        model=model,
        feature=feature,
        extractor=extractor,
        predict_feature=predict_feature,
        stateful=True,
        use_multiband=use_multiband,
        lim_for_testing=lim_for_testing,
        extract_in_loading=extract_in_loading,
    )

    dataset_val = DataGeneratorPickles(
        data_dir=data_dir,
        filename=dataset_name + "_test.pickle",
        mini_batch_size=seq_len,
        batch_size=batch_size,
        set="test",
        model=model,
        feature=feature,
        extractor=extractor,
        predict_feature=predict_feature,
        stateful=True,
        use_multiband=use_multiband,
        lim_for_testing=lim_for_testing,
        extract_in_loading=extract_in_loading,
    )

    if use_multiband:
        input_size = 3
        output_size = 3
    else:
        output_size = dataset.output_size

    model = LSTM_film(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        conditioning_dim=dataset.conditioning_dim,
        batch_size=batch_size,
        order=order,
        task_embedding_dim=embedding_dim,
        device=device,
    ).to(device)

    print(f"Training a model with features: {feature}\n")
    train_model(
        dataset=dataset,
        dataset_val=dataset_val,
        model=model,
        model_path=model_path,
        fs=fs,
        epochs=epochs,
        loss_id=loss,
        lr=lr,
        device=device,
        feature=feature,
        extractor=extractor,
        use_multiband=use_multiband,
        predict_feature=predict_feature,
        extract_in_loading=extract_in_loading,
    )


def train_model(
    dataset: DataGeneratorPickles,
    dataset_val: DataGeneratorPickles,
    model: str,
    model_path: Path,
    fs: int,
    epochs: int,
    loss_id: str,
    lr: float,
    device: str,
    feature: str,
    extractor: str,
    use_multiband: bool,
    predict_feature: bool,
    extract_in_loading: bool,
) -> Literal[42]:
    """Train the model on a dataset."""

    # Setup dataloader
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    val_dataloader = torch.utils.data.DataLoader(
        dataset_val, batch_size=1, shuffle=False
    )

    # Setup feature extractor
    extractor = EssentiaExtractor(
        samplerate=dataset.samplerate,
        frame_size=dataset.frame_size,
        hop_size=dataset.hop_size,
    )

    # Initialize early stopping counter
    lr_count = 0
    if loss_id == "SF":
        loss_fn = SpectralFluxLoss().to(device)
    elif loss_id == "ESR":
        loss_fn = ESRLoss().to(device)
    elif loss_id == "STFT":
        loss_fn = MultiResolutionSTFTLoss().to(device)
    elif loss_id == "NMSE":
        loss_fn = NormalizedMSELoss().to(device)
    elif loss_id == "MSE":
        loss_fn = F.mse_loss
    elif loss_id == "ADG":
        loss_fn = ADGLoss(sample_rate=fs).to(device)
    elif loss_id == "MSE_HilbertADG":
        loss_fn = HilbertADGLoss(sample_rate=fs).to(device)
    elif loss_id == "MSE_PeakADG":
        loss_fn = PeakADGLoss(sample_rate=fs).to(device)
    elif loss_id == "MSE_RMSADG":
        loss_fn = RMSADGLoss(sample_rate=fs).to(device)
    elif loss_id == "MAE_HilbertADG":
        loss_fn = HilbertADGLoss(sample_rate=fs, use_mae=True).to(device)
    elif loss_id == "MAE_PeakADG":
        loss_fn = PeakADGLoss(sample_rate=fs, use_mae=True).to(device)
    elif loss_id == "MAE_RMSADG":
        loss_fn = RMSADGLoss(sample_rate=fs, use_mae=True).to(device)
    elif loss_id == "CausalADG":
        loss_fn = CausalADGLoss(sample_rate=fs).to(device)
    else:
        loss_fn = F.l1_loss

    # Initialize checkpoint manager
    ckpt_manager = CheckpointManager(model_path / "my_checkpoints")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")
    print(f"Dataset length: {len(dataset)}")
    print("\n train batch_size", dataset.batch_size)
    print("\n val batch_size", dataset_val.batch_size)
    print("\n model_type", model.model_type)
    print("\n seq len", dataset_val.frame_size)
    print("\n conditioning_dim", dataset_val.conditioning_dim)
    print("\n embedding_dim", model.task_embedding_dim)
    print("\n output_size", dataset_val.output_size)
    print("\n order", model.order)
    print("\n lr", lr)
    print("\n epochs ", epochs)
    print("\n extractor ", extractor)
    print("\n feature ", feature)
    print("\n loss ", loss_id)
    print("\n")

    # Define the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=0,  # 1e-2
    )
    # Define the scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Load last checkpoint
    checkpoint = ckpt_manager.load_last_checkpoint(model, optimizer, device="cpu")
    if checkpoint:
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_val_loss"]
        print(f"Resuming from epoch {start_epoch}, best metric: {best_loss}")
    else:
        print("Start training from scratch...")
        best_loss = float("inf")

    train_losses, val_losses = [], []
    avg_train_loss, avg_val_loss = float("inf"), float("inf")

    # Training loop
    for epoch in range(epochs):
        train_batches = 0
        train_loss, val_loss = 0, 0
        model.train()
        for input, target, conditioning in tqdm(
            train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}", disable=False
        ):
            input = input[0].to(
                device
            )  # input shape: [1, batch_size, mini_batch_size, 1]
            target = target[0].to(
                device
            )  # target shape: [1, batch_size, mini_batch_size, 1]
            conditioning = conditioning[0].to(device)

            loss = model.train_step(
                x=input,
                y=target,
                c=conditioning,
                optimizer=optimizer,
                criterion=loss_fn,
                loss_id=loss_id,
            )
            train_loss += loss
            train_batches += 1

        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)

        # Validation phase
        if (epoch + 1) % 1 == 0:
            total_val_loss = 0
            val_batches = 0
            model.eval()
            with torch.no_grad():
                for input, target, conditioning in tqdm(
                    val_dataloader, desc=f"Validation Epoch {epoch + 1}", disable=True
                ):
                    input = input[0].to(device)
                    target = target[0].to(device)
                    conditioning = conditioning[0].to(device)

                    val_loss = model.val_step(
                        x=input,
                        y=target,
                        c=conditioning,
                        criterion=loss_fn,
                        loss_id=loss_id,
                    )

                    total_val_loss += val_loss
                    val_batches += 1

            avg_val_loss = total_val_loss / val_batches
            val_losses.append(avg_val_loss)

            # Update learning rate scheduler with validation loss
            scheduler.step()

            print(
                f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}"
            )
            print(f"Learning Rate {optimizer.param_groups[0]['lr']:.2e}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            # Save best checkpoint (assuming this is the best model so far)
            print(f"Epoch {epoch + 1}, Validation loss improved: ", best_loss)
            lr_count = 0

            # Save latest checkpoint
            state_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_loss,
                "val_loss": avg_val_loss,
                "best_val_loss": best_loss,
            }
            ckpt_manager.save_checkpoint(state_dict, is_best=True)

        else:
            lr_count += 1
            print(
                f"Epoch {epoch + 1}, Validation loss did not improved. Best val loss: ",
                best_loss,
            )
            if lr_count == 10:
                print(f"No improvements over 10 epochs -> stop")

        # Save latest checkpoint
        state_dict = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss,
            "val_loss": avg_val_loss,
            "best_val_loss": best_loss,
        }

        ckpt_manager.save_last_checkpoint(state_dict)
        dataset.on_epoch_end()

    del dataset
    filename = model_path / ("losses.json")
    save_losses(train_losses=train_losses, val_losses=val_losses, filename=filename)
    filename = model_path / ("loss_plot.png")
    plot_losses(train_losses=train_losses, val_losses=val_losses, filename=filename)

    # Load best checkpoint
    best_checkpoint = ckpt_manager.load_best_checkpoint(model, device="cpu")
    if best_checkpoint:
        print(
            f"Loaded best model with metric: {best_checkpoint.get('best_val_loss', 0)}"
        )

    model.reset_hidden_states()

    for x, z, y in zip(dataset_val.x, dataset_val.z, dataset_val.y):
        prefix = "_".join(map(str, z[0, :2].tolist()))

        if extract_in_loading == False:
            if use_multiband:
                x = dataset_val.multiband.decompose(x)[:, 0, :]
            print(x.shape)
            prediction_audio = model(
                x[None, :24000].to(device),
                z[None, :24000].to(device),
                detach_states=False,
            )
            feature_target = y[:24000, 1:]
            target_audio = y[:24000, :1]
            prediction_features = prediction_audio[0, :, 1:]

            if use_multiband:
                prediction_audio = prediction_audio[0, :, :]
            else:
                prediction_audio = prediction_audio[0, :, :1]

            if predict_feature:
                # Get the tensors
                input_tensor = z[:, dataset_val.device_param :]  # Your input
                target_tensor = feature_target.detach().cpu()
                pred_tensor = prediction_features.detach().cpu()

                # Find the minimum length (assuming they're all 2D tensors [channels, time] or similar)
                min_length = min(
                    input_tensor.shape[-1],
                    target_tensor.shape[-1],
                    pred_tensor.shape[-1],
                )

                # Crop all tensors to the minimum length
                input_cropped = input_tensor[
                    ..., :min_length
                ]  # ... preserves all previous dimensions
                target_cropped = target_tensor[..., :min_length]
                pred_cropped = pred_tensor[..., :min_length]

                save_feature_files(
                    input_cropped,
                    target_cropped,
                    pred_cropped,
                    model_path,
                    feature,
                    prefix=prefix,
                    sample_rate=fs,
                )
        else:
            prediction_audio = model(
                x[None].to(device), z[None].to(device), detach_states=False
            )

        if use_multiband:
            prediction_audio = torch.sum(prediction_audio, dim=-1, keepdim=True)
            x = torch.sum(x, dim=-1, keepdim=True)

        num_samples = int(3 * fs)
        save_audio_files(
            x[:num_samples].detach().cpu().flatten(),
            target_audio[:num_samples].detach().cpu().flatten(),
            prediction_audio[:num_samples].detach().cpu().flatten(),
            model_path,
            prefix=prefix,
            sample_rate=fs,
        )

        pred = prediction_audio.permute(1, 0).detach().cpu()
        tar = target_audio.permute(1, 0).detach().cpu()

        decimals = 6
        losses_dict = {
            "SpectralFlux": round(SpectralFluxLoss()(pred, tar).item(), decimals),
            "MultiResolutionSTFT": round(
                MultiResolutionSTFTLoss()(pred, tar).item(), decimals
            ),
            "ESR": round(ESRLoss()(pred, tar).item(), decimals),
            "NormalizedMSE": round(NormalizedMSELoss()(pred, tar).item(), decimals),
            "MSE": round(F.mse_loss(pred, tar).item(), decimals),
            "MAE": round(F.l1_loss(pred, tar).item(), decimals),
        }

        filename = model_path / "test_losses.txt"
        write_file(losses_dict, filename)

    return 42
