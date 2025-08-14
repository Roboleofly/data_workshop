import argparse
import h5py
from pathlib import Path
import sys
import numpy as np 
import matplotlib.pyplot as plt 

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
        default="/media/jushen/leofly-liao/datasets/h5/agilex/agilex_cobotmagic3_dualArm-gripper-3cameras_2_find_out_packaging_tape_into_the_other_basket_20250703/success_episodes/0703_114638/data/trajectory.hdf5",              # 默认值
        help="Path to the .h5 or .hdf5 file (default: ./data.h5)"
    )
    args = parser.parse_args()

    h5_path = Path(args.path).expanduser()
    if not h5_path.is_file():
        sys.exit(f"Error: {h5_path} not found or is not a file.")

    # draw flag 
    is_draw = False 
    save_path = "chassis_twist_plot.png"

    with h5py.File(h5_path, "r") as root:
        print(f"\n📘 Listing items in: {h5_path}\n")
        list_items(root)

        print(root["language_instruction"][()].decode('utf-8'))

        if is_draw:
            chassis_states = root['puppet/chassis_state_coor']
            chassis_twists = root['puppet/chassis_state_twist']
            n = max(500, len(chassis_states))

            all_crrt_twists = []

            for index in range(n):
                chassis_state = chassis_states[index]
                chassis_twist = chassis_twists[index]

                print(chassis_twist)

                all_crrt_twists.append(chassis_twist)

            all_crrt_twists = np.array(all_crrt_twists)  # Shape: [N, 6] or more depending on chassis_state dim

            # 绘图
            plt.figure(figsize=(12, 8))
            for i in range(all_crrt_twists.shape[1]):
                plt.plot(all_crrt_twists[:, i], label=f'twist[{i}]')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.title('Chassis Twist Curve')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            # 保存图像
            plt.savefig(save_path)
            plt.close()
            print(f"Twist plot saved to {save_path}")


if __name__ == "__main__":
    main()
