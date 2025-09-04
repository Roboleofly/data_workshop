import h5py
import numpy as np
import cv2
import traceback
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm
import concurrent.futures
from pathlib import Path
from threading import Lock
import json
import ast
import logging
import os


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        features = json.load(f)

    shape_fields = ["action", "observation.state"]
    for field in shape_fields:
        if field in features:
            features[field]["shape"] = ast.literal_eval(features[field]["shape"])

    print(f"Loaded features config: {features}")
    return features


class LerobotH5Episodes:
    def __init__(self, dataset=None):
        self.data = None
        self.dataset = dataset

        # TODO  how to get task name from the h5 file 

    def load_episode(self, episode_path: Path):
        image_size = (640, 480)
        try:
            with h5py.File(episode_path, 'r') as file:
                self.language_instruction = file["language_instruction"][()].decode('utf-8')
                self.puppet_state = np.array(file["puppet/arm_joint_position"])
                chassis_twist = np.array(file["puppet/chassis_state_twist"])
                self.puppet_state = np.concatenate([self.puppet_state, chassis_twist], axis=1)

                def decode_images(hdf5_dataset):
                    return [
                        cv2.resize(
                            cv2.imdecode(np.frombuffer(img_compressed, np.uint8), cv2.IMREAD_COLOR),
                            image_size
                        ) for img_compressed in hdf5_dataset
                    ]

                self.rgb_camera_top = np.stack(decode_images(file["observations/rgb_images/camera_front"]))
                self.rgb_camera_wrist_left = np.stack(decode_images(file["observations/rgb_images/camera_left"]))
                self.rgb_camera_wrist_right = np.stack(decode_images(file["observations/rgb_images/camera_right"]))

        except Exception as e:
            print(f'Data extract error {episode_path}: {e}')
            return

    def _save_lerobot_episode(self):
        try:
            num_frames = len(self.rgb_camera_top)
            for i in range(num_frames):
                frame_data = {
                    'task': self.language_instruction,
                    "action": self.puppet_state[i],
                    "observation.state": self.puppet_state[i],
                    "observation.images.camera_top": self.rgb_camera_top[i],
                    "observation.images.camera_wrist_left": self.rgb_camera_wrist_left[i],
                    "observation.images.camera_wrist_right": self.rgb_camera_wrist_right[i],
                }
                self.dataset.add_frame(frame_data)

            self.dataset.save_episode()

        except Exception as e:
            print(f"Error saving episode: {e} {traceback.format_exc()}")


def create_lerobot_dataset(repo_id, robot_name, feature_config_pth, output_dir):
    features = load_config(feature_config_pth)
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_dir,
        fps=30,
        robot_type=robot_name,
        features=features,
    )
    dataset.start_image_writer(num_processes=2, num_threads=10) # num_processes=2, num_threads = 10
    dataset.lock = Lock()
    return dataset


class Task:
    def __init__(self, name, src_root):
        self.name = name
        self.src_root = src_root
        self.episodes = []

    def get_episodes(self):
        src_root = Path(self.src_root)
        episodes_dirs = [ep for ep in src_root.iterdir() if ep.is_dir()]
        for ep_dir in episodes_dirs:
            episode_path = ep_dir / "data" / "trajectory.hdf5"
            if episode_path.exists():
                self.episodes.append(episode_path)
        return self.episodes


class Converter:
    def __init__(self, robot_name, workers, feature_config_pth):
        self.robot_name = robot_name
        self.workers = workers
        self.feature_config_pth = feature_config_pth
        self.dataset = None

    def convert(self, task, repo_id, output_dir):
        output_dir = Path(output_dir) / task.name
        episodes = task.get_episodes()

        if output_dir.exists():
            print(f"Warning: {output_dir} already exists, skipping.")
            return

        self.dataset = create_lerobot_dataset(repo_id, self.robot_name, self.feature_config_pth, output_dir)

        def _process_episode(ep_path: Path):
            converter = self._get_converter()
            converter.load_episode(ep_path)
            with self.dataset.lock:
                converter._save_lerobot_episode()

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = [executor.submit(_process_episode, ep) for ep in episodes]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(episodes), desc=task.name):
                try:
                    future.result()
                except Exception as e:
                    print(f"Task: {task.name}, Error: {e}")

    def _get_converter(self):
        return LerobotH5Episodes(self.dataset)


if __name__ == '__main__':

    # # 所有 repo_id 都是当前目录下的文件夹名
    # src_root_base = "/media/jushen/leofly-liao/datasets/h5/agilex/mobile"
    # output_dir = "/media/jushen/leofly-liao/datasets/lerobot/agilex/mobile"

    # # repo_ids = sorted(os.listdir(src_root_base))  # 获取所有文件夹名

    # repo_ids = ["agilex_cobotmagic2_dualArm-gripper-3cameras_5_arrange_dishes_pick_lettuce_and_serve_dishes","agilex_cobotmagic2_dualArm-gripper-3cameras_5_arrange_dishes_pick_lettuce_and_serve_dishes_20250710","agilex_cobotmagic2_dualArm-gripper-3cameras_5_arrange_dishes_pick_lettuce_and_serve_dishes_20250711"]

    # converter = Converter(
    #     robot_name="agilex_3",
    #     workers=2,
    #     feature_config_pth="/media/jushen/leofly-liao/workspace/data_workshop/converter/feature_config.json"
    # )

    # for repo_id in repo_ids:
    #     src_root = os.path.join(src_root_base, repo_id, "success_episodes")
    #     if not os.path.isdir(src_root):
    #         continue  # 跳过非文件夹

    #     task = Task(
    #         name=repo_id,
    #         src_root=src_root
    #     )

    #     print(f"Converting {repo_id} ...")
    #     converter.convert(task, repo_id=repo_id, output_dir=output_dir)

    # repo_id = "agilex_cobotmagic2_dualArm-gripper-3cameras_5_arrange_dishes"

    # converter = Converter(
    #     robot_name='agilex_3',
    #     workers= 1, # 4
    #     feature_config_pth='/media/jushen/leofly-liao/workspace/data_workshop/converter/feature_config.json'
    # )

    # task = Task(
    #     name=repo_id,
    #     src_root='/media/jushen/leofly-liao/datasets/h5/agilex/mobile/agilex_cobotmagic2_dualArm-gripper-3cameras_5_arrange_dishes/success_episodes/train'
    # )

    # converter.convert(task, repo_id=repo_id, output_dir='/media/jushen/leofly-liao/datasets/lerobot/agilex/mobile')


    repo_ids = [
        # "agilex_cobotmagic2_dualArm-gripper-3cameras_5_arrange_dishes_20250709",
        "agilex_cobotmagic2_dualArm-gripper-3cameras_5_arrange_dishes_pick_lettuce_and_serve_dishes_20250710",
        "agilex_cobotmagic2_dualArm-gripper-3cameras_5_back_shelf",
        "agilex_cobotmagic2_dualArm-gripper-3cameras_5_back_shelf_20250729",
        "agilex_cobotmagic2_dualArm-gripper-3cameras_5_back_shelf_20250730",
        "agilex_cobotmagic2_dualArm-gripper-3cameras_5_bowl_faucet",
        "agilex_cobotmagic2_dualArm-gripper-3cameras_5_bowl_faucet_20250721",
        "agilex_cobotmagic2_dualArm-gripper-3cameras_5_brake_pads",
        "agilex_cobotmagic2_dualArm-gripper-3cameras_5_brake_pads_250804",
        "agilex_cobotmagic2_dualArm-gripper-3cameras_5_brake_pads_250805",
        "agilex_cobotmagic2_dualArm-gripper-3cameras_5_move_apple",
        "agilex_cobotmagic2_dualArm-gripper-3cameras_5_move_apple_20250716",
        "agilex_cobotmagic2_dualArm-gripper-3cameras_5_move_corn",
        "agilex_cobotmagic2_dualArm-gripper-3cameras_5_move_corn_20250714",
        "agilex_cobotmagic2_dualArm-gripper-3cameras_5_move_forward",
        "agilex_cobotmagic2_dualArm-gripper-3cameras_5_organize_bottom",
        "agilex_cobotmagic2_dualArm-gripper-3cameras_5_place_coca",
        "agilex_cobotmagic2_dualArm-gripper-3cameras_5_place_coca_20250722",
        "agilex_cobotmagic2_dualArm-gripper-3cameras_5_supermarket_pick_apple",
        "agilex_cobotmagic2_dualArm-gripper-3cameras_5_supermarket_pick_apple_20250723",
    ]

    converter = Converter(
        robot_name="agilex_3",
        workers=1,
        feature_config_pth="/media/jushen/leofly-liao/workspace/data_workshop/converter/feature_config.json",
    )

    output_dir = "/media/jushen/leofly-liao/datasets/lerobot/agilex/mobile"
    base_src_root = "/media/jushen/leofly-liao/datasets/h5/agilex/mobile"

    for repo_id in repo_ids:
        task = Task(
            name=repo_id,
            src_root=f"{base_src_root}/{repo_id}/success_episodes",
        )
        print(f"Converting {repo_id} ...")
        converter.convert(task, repo_id=repo_id, output_dir=output_dir)