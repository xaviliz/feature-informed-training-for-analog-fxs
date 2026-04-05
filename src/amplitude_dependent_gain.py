import argparse
import json
from typing import NoReturn
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from dataloader import DataGeneratorPickles
from utils.train import save_feature_files


script_path = Path(__file__).resolve()
adg_features = ("peak", "rms", "envelope")


# Estimate Amplitude Dependent Gain (ADG) function
def estimate_adg_error(
    samplerate: int,
    extractor: object,
    feature_id: str,
    audio_in: np.ndarray,
    audio_out: np.ndarray,
    target: np.ndarray,
    mode: str = "peak",
) -> np.ndarray:

    # Revise some conditions before going ahead
    conditions = [feature_id == feature for feature in adg_features]
    assert any(conditions), f"No feature_id available for {feature_id}"

    # Extract input, target and out features
    feature_in = extractor.process_audio(audio_in, samplerate, feature_id)[
        feature_id
    ].reshape(-1, 1)
    feature_out = extractor.process_audio(audio_out, samplerate, feature_id)[
        feature_id
    ].reshape(-1, 1)
    feature_tg = extractor.process_audio(target, samplerate, feature_id)[
        feature_id
    ].reshape(-1, 1)

    # Estimate ADG vectors
    in_tg_adg = feature_tg / feature_in
    in_out_adg = feature_out / feature_in

    # Estimate MSE between ADG vectors
    mse = np.mean((np.array(in_tg_adg) - np.array(in_out_adg)) ** 2)

    return in_tg_adg, in_out_adg, mse


def main(cfg: dict) -> NoReturn:

    # Make a vector with the unnormalized feature names
    samplerate = 48000
    output_path = script_path.parent / "data" / "generated" / script_path.stem
    output_path.mkdir(parents=True, exist_ok=True)

    # Initalize DataGeneratorPickles
    dataset = DataGeneratorPickles(
        data_dir=Path(cfg["data_dir"]),
        filename=cfg["dataset_name"] + "_train.pickle",
        mini_batch_size=cfg["seq_len"],
        batch_size=1,
        set="train",
        model="lstm",
        feature="peak",
        extractor=cfg["extractor"],
        predict_feature=False,
        stateful=True,
        lim_for_testing=False,  # limited for testing purpouses
        extract_in_loading=True,
    )

    # Get pickle data (x, y, z)
    print(
        f"x.shape: {dataset.x.shape}"
    )  # input (1sr dimension: signal permutations, 2nd dimension: time, 3rd dimension: number of channels)
    print(f"y.shape: {dataset.y.shape}")  # target
    print(f"z.shape: {dataset.z.shape}")  # parameter values (conditioning)

    # Prepare some values
    feature = "amplitude_dependent_gain"
    epsilon = 1e-9
    _input = dataset.x[1].numpy() + epsilon
    _target = dataset.y[1].numpy() + epsilon
    extractor = dataset.extractorObj

    print(_input.shape)
    num_samples = _input.shape[0]
    duration = num_samples / samplerate

    print(f"Number of samples: {num_samples}")
    print(f"Duration: {duration:.3f} [s]")

    # Run our ADG implementation to estimate errors
    in_tg_adg, in_out_adg, mse = estimate_adg_error(
        samplerate, dataset.extractorObj, "peak", _input, _target, _target
    )
    print(f"mse: {mse}")

    # Estimate AB signal
    ab = _input - _target
    # save_feature_files(_input, _target, ab, output_path, "ab", prefix="ab_")

    # Estimate Raw ADG signal
    _amplitude_dependent_gain = np.abs(_input) / np.abs(_target)
    # save_feature_files(
    #     _input,
    #     _target,
    #     _amplitude_dependent_gain,
    #     output_path,
    #     feature + " - rectified input / target",
    #     prefix="abs_",
    # )

    # Estimate Amplitude Dependent Gain using PEAK values
    peak_input = extractor.process_audio(_input, samplerate, "peak")["peak"].reshape(
        -1, 1
    )
    peak_target = extractor.process_audio(_target, samplerate, "peak")["peak"].reshape(
        -1, 1
    )
    print(peak_input.shape)
    print(_input.shape)

    peak_adg = peak_target / peak_input
    save_feature_files(
        peak_input,
        peak_target,
        peak_adg,
        output_path,
        feature + " - peak",
        prefix="peak_",
    )

    # Compute the Peak Amplitude Dependent Gain in dB
    db_peak_input = 20 * np.log10(peak_input)
    db_peak_target = 20 * np.log10(peak_target)
    db_peak_adg = 20 * np.log10(peak_adg)

    save_feature_files(
        db_peak_input,
        db_peak_target,
        db_peak_adg,
        output_path,
        feature + " - peak [dB]",
        prefix="peakDB_",
    )

    num_frames = db_peak_input.shape[0]
    timestamps = np.linspace(0, duration, num_samples)
    feature_timestamps = np.linspace(0, duration, num_frames)

    plot_signals(
        db_peak_input,
        db_peak_target,
        db_peak_adg,
        feature_timestamps,
        output_path,
        "Amplitude Dependent Gain",
        "peakdB",
    )
    exit()

    # Estimate Amplitude Dependent Gain using RMS values
    rms_input = extractor.process_audio(_input, samplerate, "rms")["rms"].reshape(-1, 1)
    rms_target = extractor.process_audio(_target, samplerate, "rms")["rms"].reshape(
        -1, 1
    )

    rms_adg = rms_target / rms_input
    save_feature_files(
        rms_input, rms_target, rms_adg, output_path, feature + " - rms", prefix="rms_"
    )

    # Compute the RMS Amplitude Dependent Gain in dB
    db_rms_input = 20 * np.log10(rms_input)
    db_rms_target = 20 * np.log10(rms_target)
    db_rms_adg = 20 * np.log10(rms_adg)

    save_feature_files(
        db_rms_input,
        db_rms_target,
        db_rms_adg,
        output_path,
        feature + " - rms [dB]",
        prefix="rmsDB_",
    )

    # Estimate Amplitude Dependent Gain using Envelope values
    envelope_input = extractor.process_audio(_input, samplerate, "envelope")[
        "envelope"
    ].reshape(-1, 1)
    envelope_target = extractor.process_audio(_target, samplerate, "envelope")[
        "envelope"
    ].reshape(-1, 1)

    envelope_adg = envelope_target / envelope_input
    save_feature_files(
        envelope_input,
        envelope_target,
        envelope_adg,
        output_path,
        feature + " - envelope",
        prefix="envelope_",
    )

    # Compute the Peakmeter Envelope Amplitude Dependent Gain in dB
    db_env_input = 20 * np.log10(envelope_input)
    db_env_target = 20 * np.log10(envelope_target)
    db_env_adg = 20 * np.log10(envelope_adg)

    save_feature_files(
        db_env_input,
        db_env_target,
        db_env_adg,
        output_path,
        feature + " - envelope [dB]",
        prefix="envelopeDB_",
    )


def plot_signals(
    input_feature: np.ndarray,
    output_feature: np.ndarray,
    prediction_feature: np.ndarray,
    timestamps: np.ndarray,
    model_path: Path,
    title: str,
    prefix: str = "0",
) -> None:

    plt.figure(figsize=(12, 4))
    plt.plot(timestamps, input_feature, "b-", alpha=0.9, label="input")
    plt.plot(timestamps, output_feature, "m-", alpha=0.7, label="output")
    plt.plot(
        timestamps,
        prediction_feature,
        "g-",
        alpha=0.5,
        label="Amplitude Dependent Gain",
    )
    plt.xlim([0, timestamps[-1]])
    plt.title(title, fontsize=16)
    plt.xlabel("Time [s]", fontsize=14)
    plt.ylabel("Peak Envelope [dBFS]", fontsize=14)
    plt.legend(fontsize=12)

    filename = prefix + "_adg.png"

    # Save plot
    out_path = model_path / filename
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Examples to estimate Amplitude Dependent Gain vector with different features: peakmeter envelope, peak and rms."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration file"
    )
    args = parser.parse_args()

    # Load cfg file to load config settings
    with open(args.config, "r") as f:
        cfg = json.load(f)
    main(cfg)
