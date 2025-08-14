import h5py
import numpy as np
import cv2
import traceback
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tqdm
import concurrent
import Path
import Lock
import json 
import ast 
import logging


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

class LerobotH5Episodes:
    """
    gello 数采方式，单臂，episode 数据
    """

    def __init__(self, task_name='', dataset=None):
        self.data = None
        self.action = {}
        self.observation = {}
        self.video_data = {
            'rgb_images': {},
        }
        self.observation_depth_images = {}
        self.task_name = task_name
        self.dataset = dataset

    def load_episode(self, episode):
        """
        Load data from the specified path.
        """
        data_path = episode.path
        image_size = (640, 480)
        
        try:
            self.data = list(data_path.glob('*.hdf5'))[0]

            with h5py.File(self.data, 'r') as file:
                # Read robotic arm joint data
                self.puppet_state = np.array(file["puppet/arm_joint_position"])
                self.chassis_twist = np.array(file["puppet/chassis_state_twist"])
                self.puppet_state = np.concatenate([self.puppet_state, self.chassis_twist], axis=1)
                # Process RGB images
                self.camera_top_rgb_images = [
                    cv2.resize(
                        cv2.imdecode(img_compressed, cv2.IMREAD_COLOR),
                        image_size
                    ) for img_compressed in file["observations/rgb_images/camera_front"]
                ]
                
                self.camera_wrist_left_rgb_images = [
                    cv2.resize(
                        cv2.imdecode(img_compressed, cv2.IMREAD_COLOR),
                        image_size
                    ) for img_compressed in file["observations/rgb_images/camera_left"]
                ]

                self.camera_wrist_right_rgb_images = [
                    cv2.resize(
                        cv2.imdecode(img_compressed, cv2.IMREAD_COLOR),
                        image_size
                    ) for img_compressed in file["observations/rgb_images/camera_right"]
                ]

                self.rgb_camera_top = np.stack(self.camera_top_rgb_images)
                self.rgb_camera_wrist_left = np.stack(self.camera_wrist_left_rgb_images)
                self.rgb_camear_wrist_right = np.stack(self.camera_wrist_right_rgb_images)

        except Exception as e:
            print(f'Data extract error {data_path}: {e}')
            return
            
    def _save_lerobot_episode(self):
        """
        Save the processed data to the specified output path.
        """
            
        try:
            num_frames = len(self.video_data['rgb_images'][list(self.video_data['rgb_images'].keys())[0]])
            for i in range(num_frames):
                frame_data = {
                    'task': self.task_name,
                    "action": self.puppet_state[i],
                    "observation.state": self.puppet_state[i],
                    "observation.images.camera_top": self.rgb_camera_top[i],
                    "observation.images.camera_wrist_left": self.rgb_camera_wrist_left[i],
                    "observation.images.camera_wrist_right": self.rgb_camear_wrist_right[i],
                }

                self.dataset.add_frame(frame_data)
            self.dataset.save_episode()
        except Exception as e:
            # 打印堆栈
            print(f"Error saving episode: {e} {traceback.format_exc()}")
          

def create_lerobot_dataset( repo_id, robot_name, feature_config_pth, output_dir):
    print(f"create_lerobot_dataset {repo_id},  robot_anme: {robot_name}, feature_config: {feature_config_pth}, output_dir: {output_dir}")

    # select the lerobot dataset feature config 
    features = load_config(feature_config_pth)

    # features = new_0518_feature_map[robot_name]
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_dir,
        fps=30,
        robot_type=robot_name,
        features=features,
    )
    dataset.start_image_writer(num_processes=2, num_threads=10)
    dataset.lock = Lock()
    return dataset

class Task:
    def __init__(self, name, src_root):
        self.name = name
        self.src_root = src_root
        self.episodes = []

    def get_episodes(self):
        # 假设每个任务下有多个 episode 文件
        src_root = Path(self.src_root)
        episodes_dirs = [ep for ep in src_root.iterdir() if ep.is_dir()]
        for ep_dir in episodes_dirs:
            episode_path = ep_dir / "data" / "trajectory.hdf5"
            if episode_path.exists():
                self.episodes.append(episode_path)

class Converter:
    def __init__(self, robot_name, workers, language_instruction):
        self.robot_name = robot_name
        self.workers = workers
        self.language_instruction = language_instruction
        self.dataset = None
    
    # Define the output directory and create the dataset 
    def convert(self, task, repo_id,  output_dir):

        if not output_dir:
            return
        output_dir = Path(output_dir) / task.name

        #TODO episodes 
        episodes = task.get_episodes()
        print("======= episodes =======")
        print(episodes)
 

        if output_dir.exists():
            print(f"Warning: {output_dir} already exists, it will be skipped")
            return
        
        # create_lerobot_dataset( repo_id, robot_name, feature_config_pth, output_dir)
        self.dataset = create_lerobot_dataset(repo_id, self.robot_name, output_dir)

        def _process_episode(episode, output_dir ):
            output_path = Path(output_dir) / episode.state
    
            converter = self._get_converter(self.robot_name, output_dir, task.name)
            converter.load_episode(episode)
            # 如果是 lerobot 类型，此处需要加锁
            if self.dataset is not None:
                with self.dataset.lock:
                    converter._save_lerobot_episode()
            else:
                converter._save_lerobot_episode()


        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = [
                executor.submit(
                    _process_episode,
                    episode,
                    output_dir,
                )
                for episode in episodes
            ]
            with tqdm(concurrent.futures.as_completed(futures), total=len(episodes), desc=task.name) as pbar:
                for future in pbar:
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Task: {task.name}, Error processing episode: {e}")
        return
        
    
    def _get_converter(self, task_name):
        # return trh lerobot h5 episodes
        return LerobotH5Episodes(task_name, self.dataset)


# converter + lerobot episode 

if __name__ == '__main__':
    converter = Converter(robot_name='agilex_3', workers=4, language_instruction='pick up the tape')

    task = Task(name='pick up the tape', src_root='/media/jushen/leofly-liao/datasets/h5/agilex/agilex_cobotmagic3_dualArm-gripper-3cameras_2_find_out_packaging_tape_into_the_other_basket_20250703')

    converter.convert(task, output_dir='/media/jushen/leofly-liao/datasets/lerobot/agilex')

