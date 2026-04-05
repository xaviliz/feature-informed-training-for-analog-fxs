# We will need to define some environmental variables for Essentia
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# We will need some warnings
import warnings

# We will need some Essentia algorithms
import essentia


from essentia import Pool


from essentia.standard import (
    PitchYinFFT,
    RMS,
    Crest,
    Flatness,
    Entropy,
    ZeroCrossingRate,
    Spectrum,
    Windowing,
    Envelope,
    FlatnessDB,
    Flux,
    SpectralCentroidTime,
    Centroid,
    ConstantQ,
    MonoLoader,
)


# Deactivate essentia warnings
essentia.log.warningActive = False

# We will need to handle with arrays
import numpy as np

# We will need to define some paths
from pathlib import Path

# We will need to get some timings
import time

from utils.dir import generated_dir
from utils.extractor import Features, write_json


script_dir = Path(__file__).resolve().parent
generated_dir = generated_dir / script_dir.stem

generated_dir.mkdir(parents=True, exist_ok=True)


class EssentiaExtractor:
    def __init__(
        self, samplerate: int, frame_size: int, hop_size: int, normalized: bool = True
    ) -> None:
        self.samplerate = samplerate
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.normalized = normalized
        self.nyquist = int(self.samplerate * 0.5)
        self.spectrum_size = int(self.frame_size * 0.5)

        # Initialize Essentia algorithms
        self.loader = MonoLoader()
        self.es_windowing = Windowing(
            type="hann", size=self.frame_size, normalized=False, zeroPhase=False
        )
        self.es_rms = RMS()
        # Peak envelope based on a Peak metering
        attack = 5  # ms
        release = 30  # ms
        self.es_envelope = Envelope(
            attackTime=attack, releaseTime=release, sampleRate=self.samplerate
        )
        self.es_crest = Crest()
        self.es_zero_crossing = ZeroCrossingRate()
        self.es_flatness = Flatness()
        self.es_entropy = Entropy()
        self.es_spectrum = Spectrum(size=self.frame_size)
        self.es_pitch = PitchYinFFT()
        self.es_spectral_flux = Flux()
        self.es_spectral_flatness = FlatnessDB()
        self.es_centroid = Centroid(range=self.samplerate // 2)
        self.es_spectral_centroid_time = SpectralCentroidTime()
        self.es_cqt = ConstantQ()
        self.pool = Pool()

    def __call__(self, frame: np.ndarray, feature: str = None) -> Pool:

        # Estimate features
        if feature is not None:
            feature = Features(feature)

        if feature is None or feature == Features.RMS:
            self.pool.add("rms", self.es_rms(frame))

        if feature is None or feature == Features.PEAK:
            self.pool.add("peak", np.max(np.abs(frame)))

        if feature is None or feature == Features.ENVELOPE:
            self.pool.add("envelope", self.es_envelope(frame))  # .astype(float))

        if feature is None or feature == Features.ZERO_CROSSING_RATE:
            self.pool.add("zero_crossing_rate", self.es_zero_crossing(frame))

        rectified_frame = np.abs(frame)

        if feature is None or feature == Features.FLATNESS:
            self.pool.add(
                "flatness", self.es_flatness(rectified_frame)
            )  # fails with negative values

        if feature is None or feature == Features.CREST:
            self.pool.add("crest", self.es_crest(rectified_frame))

        if feature is None or feature == Features.ENTROPY:
            self.pool.add("entropy", self.es_entropy(rectified_frame))

        spectrum = self.es_spectrum(self.es_windowing(frame))
        if feature is None or feature == Features.STFT:
            self.pool.add("stft", spectrum)

        if feature is None or feature == Features.PITCH:
            pitch, _ = self.es_pitch(spectrum)
            if self.normalized:
                pitch = pitch / self.nyquist
            self.pool.add("pitch", pitch)

        if feature is None or feature == Features.SPECTRAL_FLUX:
            spectral_flux = self.es_spectral_flux(spectrum)
            if self.normalized:
                spectral_flux = spectral_flux / self.spectrum_size
            self.pool.add("spectral_flux", spectral_flux)  # halfRectify = False

        if feature is None or feature == Features.SPECTRAL_FLATNESS:
            self.pool.add("spectral_flatness", self.es_spectral_flatness(spectrum))

        if feature is None or feature == Features.SPECTRAL_CENTROID:
            out = self.es_centroid(spectrum)
            if self.normalized:
                out = out / self.nyquist
            self.pool.add("spectral_centroid", out)

        return self.pool

    def process_audio(
        self,
        audio: np.ndarray,
        samplerate: int,
        feature: str = None,
        out_path: Path = None,
    ) -> dict:

        # Configure algos that depends in samplerate
        self.pool.clear()

        if samplerate != self.samplerate:
            warnings.warn(
                f"Samplerate has been changed from {self.samplerate}Hz to {samplerate}Hz"
            )
            self.es_cqt.configure(sampleRate=samplerate)
            self.es_envelope.configure(sampleRate=samplerate)
            self.samplerate = samplerate

        # Single-channel processing
        if audio.ndim > 1:
            num_samples, num_channels = audio.shape
            audio = audio.mean(axis=1, dtype=np.float32)
        else:
            num_samples = audio.shape[0]
            audio = np.float32(audio)

        duration = num_samples / samplerate

        # Prepare some variables
        pind, n = (0, 0)
        frame_duration = self.frame_size / samplerate
        start_time = time.time()

        # Slice signal in frames and process them
        while pind <= (len(audio) - self.frame_size):
            # print(f"Processed {(frame_duration * n):.2f}/{duration:.2f}s", end="\r")
            frame = audio[pind : pind + self.frame_size]
            self(frame, feature)
            n += 1
            pind += self.frame_size

        end_time = time.time() - start_time

        # print(f"end_time: {end_time:.2f}s")
        # print(f"rtf: {(end_time/duration):.2f}")

        # Collect features from the pool in a dict
        out = pool_to_dict(self.pool)

        # Attach cfg parameters to dict
        cfg = list()
        cfg.append(["sample_rate", "frame_size", "hop_size"])
        cfg.append([samplerate, self.frame_size, self.hop_size])

        # Store features cfg in a json/pickle file
        if out_path is not None:
            print(f"Storing features and config in {out_path}")
            write_json(out, out_path.with_suffix(".jsonl"))  # write features

            # write configure in a tsv
            with open(out_path.with_suffix(".tsv"), "w") as f:
                for row in cfg:
                    f.write("\t".join(map(str, list(row))) + "\n")

        return out


def pool_to_dict(pool) -> dict:
    # Workaround to convert Pool to dict
    descs = pool.descriptorNames()
    result = {}

    for d in descs:
        # print(d)
        value = pool[d]
        result[d] = value
    return result


if __name__ == "__main__":
    # Prepare some values and timestamps
    samplerate = 48000
    duration = 5
    t = np.arange(0, duration, 1 / samplerate)

    # Generate sine wave
    audio = np.sin(2 * np.pi * 1000 * t)

    frame_size = 1024
    hop_size = 1024

    # process this audio
    start_time = time.time()
    extractor = EssentiaExtractor(
        samplerate=samplerate, frame_size=frame_size, hop_size=hop_size
    )
    out = extractor.process_audio(audio, samplerate)

    end_time = time.time()
    rtf = (end_time - start_time) / duration

    print(out)
    print(f"rtf: {rtf}")
