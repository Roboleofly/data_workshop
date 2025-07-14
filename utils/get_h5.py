import argparse
import h5py
from pathlib import Path
import sys

def list_items(h5obj, indent=""):
    for key, item in h5obj.items():
        if isinstance(item, h5py.Group):
            print(f"{indent}📂 {key}/")
            list_items(item, indent + "    ")
        elif isinstance(item, h5py.Dataset):
            print(f"{indent}📄 {key}  shape={item.shape}  dtype={item.dtype}")
        else:
            print(f"{indent}❓ {key}  (type={type(item)})")

def main():
    parser = argparse.ArgumentParser(description="List all items in an HDF5 file.")
    parser.add_argument(
        "--path", "-p",
        default="/media/jushen/leofly-liao/datasets/h5/agilex/algo_compare/agilex_cobotmagic2_dualArm-gripper-3cameras_5_collect button/success_episodes/0707_135407/data/trajectory.hdf5",              # 默认值
        help="Path to the .h5 or .hdf5 file (default: ./data.h5)"
    )
    args = parser.parse_args()

    h5_path = Path(args.path).expanduser()
    if not h5_path.is_file():
        sys.exit(f"Error: {h5_path} not found or is not a file.")

    with h5py.File(h5_path, "r") as f:
        print(f"\n📘 Listing items in: {h5_path}\n")
        list_items(f)

if __name__ == "__main__":
    main()
