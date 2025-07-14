# -*- coding: utf-8 -*-
"""Parallel version of h52lerobot.py

This script processes episodes in parallel in order to speed up the
conversion from hdf5 files to the `LeRobotDataset` format. Episode data
is parsed in worker processes and the resulting frames are added to the
final dataset sequentially in the main process to avoid file system
conflicts.
"""

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import shutil
import json
import ast
import h5py
import numpy as np
import cv2
from tqdm import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def load_config(config_path: str) -> dict:
    """Load and process configuration file."""
    with open(config_path, "r") as f:
        features = json.load(f)

    shape_fields = ["action", "observation.state"]
    for field in shape_fields:
        if field in features:
            features[field]["shape"] = ast.literal_eval(features[field]["shape"])

    logging.info("Loaded features config: %s", features)
    return features


def initialize_dataset(repo_id: str, tgt_path: str, fps: int,
                        robot_type: str, features: dict) -> LeRobotDataset:
    """Create a dataset directory, removing the old one if present."""
    dataset_path = Path(tgt_path) / repo_id
    if dataset_path.exists():
        shutil.rmtree(dataset_path)
        logging.warning("Removed existing dataset: %s", dataset_path)

    logging.info("Creating new dataset: %s", dataset_path)
    return LeRobotDataset.create(
        repo_id=repo_id,
        root=str(dataset_path),
        fps=fps,
        robot_type=robot_type,
        features=features,
    )


def _process_episode_worker(args):
    """Worker function to load data from a single episode."""
    episode_path, task_name, image_size = args
    try:
        with h5py.File(episode_path, "r") as file:
            puppet_state = np.array(file["puppet/arm_joint_position"])
            chassis_twist = np.array(file["puppet/chassis_state_twist"])
            puppet_state = np.concatenate([puppet_state, chassis_twist], axis=1)

            camera_top_rgb_images = [
                cv2.resize(
                    cv2.imdecode(img_compressed, cv2.IMREAD_COLOR),
                    image_size,
                )
                for img_compressed in file["observations/rgb_images/camera_front"]
            ]

            camera_wrist_left_rgb_images = [
                cv2.resize(
                    cv2.imdecode(img_compressed, cv2.IMREAD_COLOR),
                    image_size,
                )
                for img_compressed in file["observations/rgb_images/camera_left"]
            ]

            camera_wrist_right_rgb_images = [
                cv2.resize(
                    cv2.imdecode(img_compressed, cv2.IMREAD_COLOR),
                    image_size,
                )
                for img_compressed in file["observations/rgb_images/camera_right"]
            ]

            rgb_camera_top = np.stack(camera_top_rgb_images)
            rgb_camera_wrist_left = np.stack(camera_wrist_left_rgb_images)
            rgb_camera_wrist_right = np.stack(camera_wrist_right_rgb_images)

    except (FileNotFoundError, OSError, KeyError) as e:
        logging.error("Skipped %s: %s", episode_path, str(e))
        return None

    frame_list = []
    for i in range(len(puppet_state)):
        frame_data = {
            "task": task_name,
            "action": puppet_state[i],
            "observation.state": puppet_state[i],
            "observation.images.camera_top": rgb_camera_top[i],
            "observation.images.camera_wrist_left": rgb_camera_wrist_left[i],
            "observation.images.camera_wrist_right": rgb_camera_wrist_right[i],
        }
        frame_list.append(frame_data)
    return frame_list


def main():
    parser = argparse.ArgumentParser(description="Parallel Dataset Conversion")
    parser.add_argument(
        "--config",
        type=str,
        default="/media/jushen/leofly-liao/workspace/data_workshop/converter/feature_config.json",
        help="Path to config JSON file",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="agilex_3_collect_button",
        help="Dataset repository ID",
    )
    parser.add_argument(
        "--src_root",
        type=str,
        default="/media/jushen/leofly-liao/datasets/h5/agilex/algo_compare/agilex_cobotmagic2_dualArm-gripper-3cameras_5_collect button/success_episodes",
        help="Source data directory",
    )
    parser.add_argument(
        "--tgt_path",
        type=str,
        default="/media/jushen/leofly-liao/datasets/lerobot/agilex/algo_compare",
        help="Target output directory",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="turn_on_light_switch",
        help="Task name identifier",
    )
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--robot_type", type=str, default="agilex_3", help="Robot type identifier")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel worker processes",
    )
    args = parser.parse_args()

    features = load_config(args.config)
    dataset = initialize_dataset(
        repo_id=args.repo_id,
        tgt_path=args.tgt_path,
        fps=args.fps,
        robot_type=args.robot_type,
        features=features,
    )

    src_root = Path(args.src_root)
    episodes = [ep for ep in src_root.iterdir() if ep.is_dir()]
    tasks = [
        (ep / "data" / "trajectory.hdf5", args.task_name, (640, 480))
        for ep in episodes
    ]

    logging.info("Start processing %d episodes with %d workers", len(tasks), args.workers)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(tqdm(executor.map(_process_episode_worker, tasks), total=len(tasks), desc="Episodes"))

    for frames in results:
        if not frames:
            continue
        for frame_data in frames:
            dataset.add_frame(frame_data)
        dataset.save_episode()

    logging.info("Dataset conversion completed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
