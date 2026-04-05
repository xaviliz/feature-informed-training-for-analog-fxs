import json
from pathlib import Path
import numpy as np
from enum import Enum


# Normalize an array between a given min and max
def normalize(arr: np.ndarray, old_min: float, old_max: float) -> float | np.ndarray:
    if old_max == old_min:
        return 0.0

    arr = np.asarray(arr, dtype=np.float32)
    return (arr - old_min) / (old_max - old_min)


# JSON functionalities
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj) -> json.JSONEncoder:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return obj.astype(float)
        return json.JSONEncoder.default(self, obj)


def write_json(data: dict, out_path: Path, jsonl: bool = True) -> None:
    with open(str(out_path), "w", encoding="utf-8") as outputFile:
        if not jsonl:
            json.dump(data, outputFile, cls=NumpyEncoder, indent=4)
        else:
            for item in data:
                outputFile.write(json.dumps(item, cls=NumpyEncoder) + "\n")


def read_json(json_path: str, jsonl: bool = True) -> dict:
    with open(json_path, encoding="utf-8") as f:
        if jsonl:
            data = []
            for line in f:
                data.append(json.loads(line))
        else:
            data = json.load(f)
    return data


class ExtendedEnum(Enum):
    @classmethod
    def list(cls) -> list[str]:
        return list(map(str, cls))

    def __str__(self) -> str:
        return str(self.name.lower())

    @classmethod
    def names(cls):
        return list(cls)

    @classmethod
    def get_id_from_name(cls, name: str) -> str:
        out = None
        for option in cls:
            if str(option) == name:
                out = option
        if out is None:
            raise ValueError(f"The {name} option doesn't exist. Choices: {cls.list()}")
        return out


class Features(ExtendedEnum):
    # Feature sizes
    ENVELOPE = "envelope"  # [n, frame_size] | normalized
    STFT = "stft"  # [n, frame_size // 2 -1]
    PITCH = "pitch"  # [n, 1] | normalized
    RMS = "rms"  # [n, 1] | normalized
    PEAK = "peak"  # [n, 1] | normalized
    FLATNESS = "flatness"  # [n, 1] | normalized
    CREST = "crest"  # [n, 1] | normalized
    ENTROPY = "entropy"  # [n, 1] | normalized
    ZERO_CROSSING_RATE = "zero_crossing_rate"  # [n, 1] | normalized
    SPECTRAL_FLUX = "spectral_flux"  # [n, 1] | normalized
    SPECTRAL_FLATNESS = "spectral_flatness"  # [n, 1] | normalized
    SPECTRAL_CENTROID = "spectral_centroid"  # [n, 1] | normalized
