from pathlib import Path


script_path = Path(__file__).resolve()
utils_dir = script_path.parent
src_dir = utils_dir.parent
root_dir = src_dir.parent
data_dir = root_dir / "data"

dataset_dir = data_dir / "dataset"
drc_ds_dir = dataset_dir / "optical-dynamic-range-compressors-la-2a-cl-1b"

generated_dir = data_dir / "generated"
