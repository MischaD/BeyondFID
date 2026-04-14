"""
beyondfid-check — quick sanity check for data paths before running the full pipeline.

Usage:
    beyondfid-check path/to/data
    beyondfid-check path/to/splits.csv
    beyondfid-check path/to/all_images.h5
"""

import os
import argparse
from beyondfid.log import logger


def check_path(path, h5_dataset_key="images", filename_key="FileName"):
    """
    Load metadata and the first image from a data path and print a summary.

    Supports folders, CSV files, and HDF5 files. For CSV files, per-split
    counts are shown. Returns True if loading succeeded, False otherwise.
    """
    print(f"\nChecking: {path}")
    print("-" * 60)

    if not os.path.exists(path):
        print(f"ERROR: path does not exist: {path}")
        return False

    try:
        if path.endswith(".h5"):
            return _check_h5(path, h5_dataset_key)
        elif path.endswith(".csv"):
            return _check_csv(path, filename_key)
        elif os.path.isdir(path):
            return _check_dir(path)
        else:
            print(f"ERROR: unsupported format '{os.path.splitext(path)[1]}'. "
                  f"Expected a directory, .csv, or .h5 file.")
            return False
    except Exception as e:
        print(f"ERROR while loading: {e}")
        return False


def _load_first(dataset):
    """Load the first item from a dataset and return (image_tensor, path_or_index)."""
    item, idx, item_path = dataset[0]
    return item, item_path


def _print_image_stats(img, label=""):
    prefix = f"  {label}: " if label else "  "
    print(f"{prefix}shape={tuple(img.shape)}, dtype={img.dtype}, "
          f"min={img.min():.3f}, max={img.max():.3f}")


def _check_h5(path, dataset_key):
    from beyondfid.data.datasets import H5Dataset

    print(f"Type:  HDF5")
    dataset = H5Dataset(path, dataset_key=dataset_key)
    n = len(dataset)
    print(f"Key:   '{dataset.dataset_key}'")
    print(f"Count: {n} images")

    if n == 0:
        print("WARNING: file contains 0 images")
        return False

    img, item_path = _load_first(dataset)
    print(f"First entry (index {item_path}):")
    _print_image_stats(img)
    return True


def _check_csv(path, filename_key):
    import pandas as pd
    from beyondfid.data.datasets import GenericDataset

    print(f"Type:  CSV")
    df = pd.read_csv(path)

    if filename_key not in df.columns:
        available = list(df.columns)
        print(f"ERROR: column '{filename_key}' not found. Available columns: {available}")
        print(f"Hint:  use --filename_key to specify the correct column name.")
        return False

    print(f"Key:   '{filename_key}'")
    print(f"Count: {len(df)} rows total")

    if "Split" in df.columns:
        for split, grp in df.groupby("Split"):
            print(f"  {split}: {len(grp)} rows")

    # Try loading the first row as an image
    basedir = os.path.dirname(os.path.abspath(path))
    first_file = str(df[filename_key].iloc[0])
    print(f"First entry: {first_file}")

    # Check if it looks like a file path (not an integer index)
    if not first_file.lstrip("-").isdigit():
        file_list = list(df[filename_key].astype(str))
        dataset = GenericDataset(file_list, basedir)
        img, item_path = _load_first(dataset)
        _print_image_stats(img)
    else:
        print("  (integer indices — pair with an H5 file for image loading)")

    return True


def _check_dir(path):
    from beyondfid.data import ALLOWED_EXTENSIONS, get_data_from_folder
    from beyondfid.data.datasets import GenericDataset

    print(f"Type:  directory")
    file_list, _ = get_data_from_folder(path, fe_name="check")
    n = len(file_list)
    print(f"Count: {n} files")

    if n == 0:
        exts = ", ".join(ALLOWED_EXTENSIONS)
        print(f"WARNING: no files found. Allowed extensions: {exts}")
        return False

    print(f"First entry: {file_list[0]}")
    dataset = GenericDataset(file_list, basedir=path)
    img, _ = _load_first(dataset)
    _print_image_stats(img)
    return True


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Quick sanity check for train/test/synth paths before a full beyondfid run.\n\n"
            "Reports file count and loads the first image from each path to verify "
            "dimensions and normalisation. Mirrors the beyondfid argument interface so "
            "you can copy-paste the same paths."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  beyondfid-check data/train/ data/test/ data/synth/\n"
            "  beyondfid-check data/splits.csv data/splits.csv data/synth.csv\n"
            "  beyondfid-check data/splits.csv data/splits.csv data/synth.csv --filename_key path\n"
            "  beyondfid-check data/train.h5 data/test.h5 data/synth.h5\n"
            "  beyondfid-check data/train.h5 data/test.h5 data/synth.h5 --h5_dataset_key data\n"
        ),
    )
    parser.add_argument("pathtrain", help="Train data: folder, .csv, or .h5.")
    parser.add_argument("pathtest",  help="Test data: folder, .csv, or .h5.")
    parser.add_argument("pathsynth", help="Synthetic data: folder, .csv, or .h5.")
    parser.add_argument("--h5_dataset_key", default="images",
                        help="HDF5 dataset key for image data. Default: %(default)s")
    parser.add_argument("--filename_key", default="FileName",
                        help="CSV column name for file paths or indices. Default: %(default)s")
    args = parser.parse_args()

    all_ok = True
    for label, path in [("train", args.pathtrain), ("test", args.pathtest), ("synth", args.pathsynth)]:
        print(f"\n[{label}]")
        ok = check_path(path, h5_dataset_key=args.h5_dataset_key, filename_key=args.filename_key)
        all_ok = all_ok and ok

    print()
    if all_ok:
        print("All paths OK.")
    else:
        print("One or more paths failed — see errors above.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
