import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import re
from scipy.io.wavfile import write


def save_feature_files(
    input_feature: np.ndarray,
    output_feature: np.ndarray,
    prediction_feature: np.ndarray,
    model_path: Path,
    feature: str,
    prefix: str = "0",
    show_figure: bool = False,
) -> None:
    """
    Generate and save a comparison plot of input, target, and predicted features.

    This function visualizes three feature representations and saves the resulting
    figure to disk. Depending on the feature dimensionality, it either:

    - Plots time–frequency representations (e.g., STFT or spectrogram-like data)
      as magnitude spectrograms in dB, or
    - Plots 1D signals (e.g., waveform or scalar feature trajectories) as
      overlaid time-series curves.

    Parameters
    ----------
    input_feature : np.ndarray
        Input feature array. Can be a 1D signal or a 2D time–frequency
        representation.
    output_feature : np.ndarray
        Ground-truth or reference feature array with the same shape as
        `input_feature`.
    prediction_feature : np.ndarray
        Model prediction feature array with the same shape as
        `input_feature`.
    model_path : pathlib.Path
        Directory where the generated plot will be saved.
    feature : str
        Name of the feature being visualized. Used as the plot title
        for 1D features.
    prefix : str, optional
        Prefix added to the output filename. Default is "0".
    show_figure : bool, optional
        If True, displays the figure interactively after saving.
        Default is False.

    Returns
    -------
    None
        The function saves the plot to disk and does not return a value.

    Notes
    -----
    - Multi-dimensional features are converted to decibel scale using:
      `20 * log10(abs(x) + 1e-10)`.
    - Spectrogram-like features are displayed using a viridis colormap.
    - The output file is saved as "<prefix>feature_plot.png" inside
      `model_path`.

    Side Effects
    ------------
    - Writes an image file to disk.
    - Optionally displays a matplotlib figure.
    - Prints the output file path to stdout.
    """
    if (input_feature.shape[-1]) > 1:
        input_feature = (20 * np.log10(np.abs(input_feature) + 1e-10)).permute(1, 0)
        output_feature = (20 * np.log10(np.abs(output_feature) + 1e-10)).permute(1, 0)
        prediction_feature = (
            20 * np.log10(np.abs(prediction_feature) + 1e-10)
        ).permute(1, 0)

        _, axes = plt.subplots(1, 3, figsize=(15, 5))

        im1 = axes[0].imshow(
            input_feature.T, aspect="auto", origin="lower", cmap="viridis"
        )
        axes[0].set_xlabel("Time (frames)")
        axes[0].set_ylabel("Frequency (bins)")
        axes[0].set_title("input_feature")
        plt.colorbar(im1, ax=axes[0], label="Magnitude (dB)")

        # Plot second STFT
        im2 = axes[1].imshow(
            output_feature.T, aspect="auto", origin="lower", cmap="viridis"
        )
        axes[1].set_xlabel("Time (frames)")
        axes[1].set_ylabel("Frequency (bins)")
        axes[1].set_title("output_feature")
        plt.colorbar(im2, ax=axes[1], label="Magnitude (dB)")

        # Plot third STFT
        im3 = axes[2].imshow(
            prediction_feature.T, aspect="auto", origin="lower", cmap="viridis"
        )
        axes[2].set_xlabel("Time (frames)")
        axes[2].set_ylabel("Frequency (bins)")
        axes[2].set_title("prediction_feature")
        plt.colorbar(im3, ax=axes[2], label="Magnitude (dB)")

        plt.tight_layout()

    else:
        plt.figure(figsize=(12, 6))
        plt.plot(input_feature, "b-", alpha=0.9, label="input_audio")
        plt.plot(output_feature, "r-", alpha=0.7, label="output_audio")
        plt.plot(prediction_feature, "g-", alpha=0.5, label="prediction_audio")
        plt.title(feature, fontsize=16)
        plt.xlabel("Time", fontsize=14)
        plt.ylabel("Amplitude", fontsize=14)
        plt.legend(fontsize=12)

    filename = prefix + "feature_plot.png"

    # Save plot
    out_path = model_path / filename
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    if show_figure:
        plt.show()
    plt.close()
    print(f"Plot saved to {out_path}")


def save_audio_files(
    input_audio: np.ndarray,
    output_audio: np.ndarray,
    prediction_audio: np.ndarray,
    model_path: Path,
    prefix: str = "0",
    sample_rate: int = 48000,
) -> None:
    """
    Save audio files in WAV format.

    Parameters:
        input_audio (np.ndarray): Input audio data array.
        output_audio (np.ndarray): Output audio data array (processed).
        prediction_audio: Predicted labels or values (could be additional info to save).
        model_path (str): The path where to save the audio files (should exist).
    """
    # Create the model path directory if it doesn't exist
    os.makedirs(model_path, exist_ok=True)

    # Saving input audio
    input_file_path = os.path.join(model_path, prefix + "_input_audio.wav")
    input_audio = np.array(input_audio.squeeze(), dtype=np.float32)
    write(input_file_path, sample_rate, input_audio)  # Scale to int16

    # Saving output audio
    output_file_path = os.path.join(model_path, prefix + "_output_audio.wav")
    output_audio = np.array(output_audio.squeeze(), dtype=np.float32)
    write(output_file_path, sample_rate, output_audio)  # Scale to int16

    # Saving output audio
    output_file_path = os.path.join(model_path, prefix + "_prediction_audio.wav")
    prediction_audio = np.array(prediction_audio.squeeze(), dtype=np.float32)
    write(output_file_path, sample_rate, prediction_audio)  # Scale to int16

    plot(input_audio, output_audio, prediction_audio, model_path, prefix=prefix)

    print(f"Audio files saved to {model_path}")


def write_file(_dict: dict, filename: Path):
    """
    Write a dictionary to a JSON file.

    This function serializes a Python dictionary and saves it to disk
    in JSON format.

    Parameters
    ----------
    _dict : dict
        Dictionary containing the data to be written. The contents must
        be JSON-serializable.
    filename : pathlib.Path
        Path to the output JSON file.

    Returns
    -------
    None
        The function writes the file to disk and does not return a value.
    """
    with open(filename, "w") as f:
        json.dump(_dict, f)
    print(f"Losses saved to {filename}")


def plot(
    input_audio: np.ndarray,
    output_audio: np.ndarray,
    prediction_audio: np.ndarray,
    model_path: Path,
    prefix: str = "0",
    show_figure: bool = False,
) -> None:
    """
    Generate and save a comparison plot of input, target, and predicted audio signals.

    This function creates a time-domain visualization showing the input audio,
    reference output audio, and predicted audio signals overlaid for comparison.
    The resulting figure is saved to disk.

    Parameters
    ----------
    input_audio : np.ndarray
        Input audio signal array.
    output_audio : np.ndarray
        Ground-truth or reference audio signal array.
    prediction_audio : np.ndarray
        Predicted audio signal array produced by a model.
    model_path : pathlib.Path
        Directory where the generated plot will be saved.
    prefix : str, optional
        Prefix added to the output filename. Default is "0".
    show_figure : bool, optional
        If True, displays the figure interactively after saving.
        Default is False.

    Returns
    -------
    None
        The function saves the plot to disk and does not return a value.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(input_audio, "gray", label="Input Audio", alpha=0.3, linewidth=1.5)
    plt.plot(output_audio, "b-", label="Output Audio", alpha=0.5, linewidth=1.2)
    plt.plot(prediction_audio, "r--", label="Prediction Audio", alpha=0.8, linewidth=1)
    plt.title("Audio Signal Comparison", fontsize=16)
    plt.xlabel("Sample Index", fontsize=14)
    plt.ylabel("Amplitude", fontsize=14)
    plt.legend(fontsize=12, loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    filename = prefix + "_plot.png"
    plt.savefig(model_path / filename, dpi=300, bbox_inches="tight")
    if show_figure:
        plt.show()
    plt.close()
    print(f"Plot saved to {filename}")


# Function to save losses to file
def save_losses(
    train_losses: np.ndarray, val_losses: np.ndarray, filename: str = "losses.json"
) -> None:
    """
    Save training and validation loss values to a JSON file.

    This function stores arrays of training and validation losses in a
    JSON-formatted file for later analysis or visualization.

    Parameters
    ----------
    train_losses : np.ndarray
        Array containing training loss values.
    val_losses : np.ndarray
        Array containing validation loss values.
    filename : str, optional
        Name or path of the output JSON file. Default is "losses.json".

    Returns
    -------
    None
        The function writes the loss data to disk and does not return a value.
    """
    losses_dict = {"train_losses": train_losses, "val_losses": val_losses}
    with open(filename, "w") as f:
        json.dump(losses_dict, f)
    print(f"Losses saved to {filename}")


# Function to plot losses
def plot_losses(
    train_losses: np.ndarray, val_losses: np.ndarray, filename: str = "loss_plot.png"
) -> None:
    """
    Generate and save a plot of training and validation losses over epochs.

    This function creates a line plot comparing training and validation
    loss values across epochs and saves the resulting figure to disk.

    Parameters
    ----------
    train_losses : np.ndarray
        Array containing training loss values for each epoch.
    val_losses : np.ndarray
        Array containing validation loss values for each epoch.
    filename : str, optional
        Name or path of the output image file. Default is "loss_plot.png".

    Returns
    -------
    None
        The function saves the plot to disk and does not return a value.
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, "b-", label="Training Loss", linewidth=2)
    plt.plot(epochs, val_losses, "r-", label="Validation Loss", linewidth=2)
    plt.title("Training and Validation Loss Over Time", fontsize=16)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {filename}")


def natural_sort_key(s: str) -> list:
    """
    Function to use as a key for sorting strings in natural order.
    This ensures that strings with numbers are sorted in human-expected order.
    For example: ["file1", "file10", "file2"] -> ["file1", "file2", "file10"]

    Args:
        s: The string to convert to a natural sort key

    Returns:
        A list of string and integer parts that can be used for natural sorting
    """
    # Split the string into text and numeric parts
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", s)]


def compute_lcm(x: float, y: float) -> float:
    """
    Compute the least common multiple (LCM) of two numbers.

    Parameters
    ----------
    x : float
        First value.
    y : float
        Second value.

    Returns
    -------
    float
        Least common multiple of ``x`` and ``y``.
    """
    return (x * y) // math.gcd(x, y)
