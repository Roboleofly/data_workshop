      
# -*- coding: utf-8 -*-
import argparse
import logging
import os
import shutil
import h5py
import numpy as np
import cv2
import json
import ast
from tqdm import tqdm
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def load_config(config_path: str) -> dict:
    """Load and process configuration file"""
    with open(config_path, 'r') as f:
        features = json.load(f)
    
    # Convert shape fields from string to tuple
    shape_fields = ["action", "observation.state"]
    for field in shape_fields:
        if field in features:
            features[field]["shape"] = ast.literal_eval(features[field]["shape"])
    
    logging.info(f"Loaded features config: {features}")
    return features

def initialize_dataset(repo_id: str, 
                      tgt_path: str, 
                      fps: int,
                      robot_type: str,
                      features: dict) -> LeRobotDataset:
    """Initialize dataset instance"""
    dataset_path = Path(tgt_path) / repo_id
    
    # Clean up existing dataset if it exists
    if dataset_path.exists():
        shutil.rmtree(dataset_path)
        logging.warning(f"Removed existing dataset: {dataset_path}")
    
    # Create new dataset
    logging.info(f"Creating new dataset: {dataset_path}")
    return LeRobotDataset.create(
        repo_id=repo_id,
        root=str(dataset_path),
        fps=fps,
        robot_type=robot_type,
        features=features,
    )

def process_episode(episode_path: Path, 
                   dataset: LeRobotDataset,
                   task_name: str,
                   image_size: tuple = (640, 480)) -> bool:
    """Process single episode data"""
    try:
        with h5py.File(episode_path, 'r') as file:
            print(file['puppet'].keys())
            # Read robotic arm joint data
            puppet_state = np.array(file["puppet/arm_joint_position"])
            chassis_twist = np.array(file["puppet/chassis_state_twist"])
            puppet_state = np.concatenate([puppet_state, chassis_twist], axis=1)
            # Process RGB images
            camera_top_rgb_images = [
                cv2.resize(
                    cv2.imdecode(img_compressed, cv2.IMREAD_COLOR),
                    image_size
                ) for img_compressed in file["observations/rgb_images/camera_front"]
            ]
            
            camera_wrist_left_rgb_images = [
                cv2.resize(
                    cv2.imdecode(img_compressed, cv2.IMREAD_COLOR),
                    image_size
                ) for img_compressed in file["observations/rgb_images/camera_left"]
            ]

            camera_wrist_right_rgb_images = [
                cv2.resize(
                    cv2.imdecode(img_compressed, cv2.IMREAD_COLOR),
                    image_size
                ) for img_compressed in file["observations/rgb_images/camera_right"]
            ]

            rgb_camera_top = np.stack(camera_top_rgb_images)
            rgb_camera_wrist_left = np.stack(camera_wrist_left_rgb_images)
            rgb_camear_wrist_right = np.stack(camera_wrist_right_rgb_images)
        
    except (FileNotFoundError, OSError, KeyError) as e:
        logging.error(f"Skipped {episode_path}: {str(e)}")
        return False
    
    # Add frame data to dataset
    num_frames = len(puppet_state)
    for i in tqdm(range(num_frames), desc=f"Processing {episode_path.name}", position=1, leave=False):
        frame_data = {
            'task': task_name,
            "action": puppet_state[i],
            "observation.state": puppet_state[i],
            "observation.images.camera_top": rgb_camera_top[i],
            "observation.images.camera_wrist_left": rgb_camera_wrist_left[i],
            "observation.images.camera_wrist_right": rgb_camear_wrist_right[i],
        }
        dataset.add_frame(frame_data)
    
    return True

def main():
    """Main execution function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Dataset Conversion Tool")
    parser.add_argument("--config", type=str, default="/media/jushen/leofly-liao/workspace/data_workshop/converter/feature_config.json", help="Path to config JSON file")
    parser.add_argument("--repo_id", type=str, default="agilex_3_pick_tape", help="Dataset repository ID")
    parser.add_argument("--src_root", type=str, default="/media/jushen/leofly-liao/datasets/h5/agilex/agilex_cobotmagic3_dualArm-gripper-3cameras_2_find_out_packaging_tape_into_the_other_basket_20250703/success_episodes/", help="Source data directory")
    parser.add_argument("--tgt_path", type=str, default="/media/jushen/leofly-liao/datasets/lerobot/agilex", help="Target output directory")
    parser.add_argument("--task_name", type=str, default="pick up the tape", help="Task name identifier")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--robot_type", type=str, default="agilex_3", help="Robot type identifier")
    args = parser.parse_args()

    features = load_config(args.config)

    # Initialize dataset
    dataset = initialize_dataset(
        repo_id=args.repo_id,
        tgt_path=args.tgt_path,
        fps=args.fps,
        robot_type=args.robot_type,
        features=features
    )

    # Process all episodes
    src_root = Path(args.src_root)
    episodes = [ep for ep in src_root.iterdir() if ep.is_dir()]
    
    logging.info(f"Start processing {len(episodes)} episodes...")

    for ep_dir in tqdm(episodes, desc="All episodes", position=0):
        ep_path = ep_dir / "data" / "trajectory.hdf5"

        # ▶ 处理单集时的帧级进度条，放到下一行显示
        if process_episode(ep_path,
                       dataset,
                       args.task_name,
                       ):                     # process_episode 里保持 position=1
            dataset.save_episode()
            logging.info(f"Saved episode: {ep_dir.name}")

    # for ep_dir in episodes:
    #     ep_path = ep_dir / "data" / "trajectory.hdf5"
    #     if process_episode(ep_path, dataset, args.task_name):
    #         dataset.save_episode()
    #         logging.info(f"Saved episode: {ep_dir.name}")

    logging.info("Dataset conversion completed!")

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    main()

    