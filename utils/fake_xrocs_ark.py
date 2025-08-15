import cv2 
import torch 
import numpy as np 
from h5_loader import ReadH5Files

import torch.nn.functional as F
import os 
import json 
import matplotlib.pyplot as plt 
import h5py


ark_info_dict = {  'camera_names': ['camera_head', 'camera_left', 'camera_right'],
                   'camera_sensors': ['rgb_images', 'depth_images'],
                   'arms': ['puppet'],
                   'controls': ['arm_joint_position', 'chassis_state_coor', 'chassis_state_twist'] }

class Fake_xros():
    def __init__(self, robo_info, h5_path):
        self.robot_info = robo_info
        self.h5_path = h5_path
        self.read_h5files = ReadH5Files(self.robot_info)

        with h5py.File(self.h5_path, 'r') as file:
            self.language_instruction = file["language_instruction"][()].decode('utf-8')
            print(f"task name: {self.language_instruction}")
            # print(file['observations']['rgb_images'].keys())

        _, control_dict, _, _, _ = self.read_h5files.execute(self.h5_path, camera_frame=0, use_depth_image=False)


        # without hand joint position 
        # == ark 
        self.episode_qpos_arm = control_dict['puppet']['arm_joint_position']
        self.episode_action_arm = control_dict['puppet']['arm_joint_position']
        self.episode_action_chassis_state = control_dict['puppet']['chassis_state_coor']
        self.episode_action_chassis_twist = control_dict['puppet']['chassis_state_twist']
        
        # to bulid the chassis action format  <-- ground truth 
        tmp_action_chassis = self.episode_action_chassis_state
        tmp_action_chassis[:,:3] = self.episode_action_chassis_twist

        self.episode_action = np.concatenate([self.episode_action_arm, tmp_action_chassis],axis=-1)
        self.episode_len = len(self.episode_qpos_arm)

        # self.episode_qpos_arm = control_dict['puppet']['arm_joint_position']
        # self.episode_qpos_hand = control_dict['puppet']['hand_joint_position']
        # self.episode_action_arm = control_dict['puppet']['arm_joint_position']
        # self.episode_action_hand = control_dict['puppet']['hand_joint_position']
        # self.episode_action = np.concatenate([self.episode_action_arm, self.episode_action_hand],axis=-1)
        # self.episode_len = len(self.episode_qpos_arm)
    
    def get_len(self):
        return self.episode_len
    
    def get_prompt(self):
        return self.language_instruction
        
    def get_obs(self, index):
        image_dict, _, _, _, _ = self.read_h5files.execute(self.h5_path, camera_frame=index, use_depth_image=False)
        
        # bulid the image obs 

        # == ark
        _, fake_head_image = cv2.imencode('.jpg', image_dict[self.robot_info['camera_sensors'][0]]['camera_head'])
        _, fake_left_image = cv2.imencode('.jpg', image_dict[self.robot_info['camera_sensors'][0]]['camera_left'])
        _, fake_right_image = cv2.imencode('.jpg', image_dict[self.robot_info['camera_sensors'][0]]['camera_right'])

        # _, fake_front_image = cv2.imencode('.jpg', image_dict[self.robot_info['camera_sensors'][0]]['camera_front'])
        # _, fake_left_image = cv2.imencode('.jpg', image_dict[self.robot_info['camera_sensors'][0]]['camera_left'])
        # _, fake_right_image = cv2.imencode('.jpg', image_dict[self.robot_info['camera_sensors'][0]]['camera_right'])
        # _, fake_top_image = cv2.imencode('.jpg', image_dict[self.robot_info['camera_sensors'][0]]['camera_top'])
        # _, fake_wrist_left_image = cv2.imencode('.jpg', image_dict[self.robot_info['camera_sensors'][0]]['camera_wrist_left'])
        # _, fake_wrist_right_image = cv2.imencode('.jpg', image_dict[self.robot_info['camera_sensors'][0]]['camera_wrist_right'])

        # bulid the fake obs 

        # == ark
        fake_obs = {
            'images': {
                'left': fake_left_image,
                'right': fake_right_image,
                'head': fake_head_image
            },
            'arm_joints': {
                'left': self.episode_qpos_arm[index][:6],
                'right': self.episode_qpos_arm[index][6:]
            },
            'chassis_state': {
                'chassis_coor': self.episode_action_chassis_state[index],
                'chassis_vel': self.episode_action_chassis_twist[index]
            }
        }


        # fake_obs = {
        #     'images': {
        #         'left': fake_left_image,
        #         'right': fake_right_image,
        #         'top': fake_top_image,
        #         'front': fake_front_image,
        #         'wrist_left': fake_wrist_left_image,
        #         'wrist_right': fake_wrist_right_image
        #     },
        #     'arm_joints': {
        #         'left': self.episode_qpos_arm[index][:6],
        #         'right': self.episode_qpos_arm[index][6:]
        #     },
        #     'hand_joints': {
        #         'left': np.array([self.episode_qpos_hand[index][0]]),
        #         'right': np.array([self.episode_qpos_hand[index][1]])
        #     }
        # }

        return fake_obs
    

if __name__ == "__main__":

    h5_path = "/media/jushen/leofly-liao/datasets/h5/ark/ark_lift_dualArm-gripper-1cameras_3_move_yellow_beetle_car_and_pick_and_place_it_into_the_blue_storage_box_20250718/success_episodes/train/0718_175127/data/trajectory.hdf5"
    
    # camera_names = ark_info_dict['camera_names']
    # print(camera_names)

    fake_xrocs = Fake_xros(ark_info_dict, h5_path) 

    for i in range(fake_xrocs.get_len()):
        obs = fake_xrocs.get_obs(i)
        print(f"{i}: {obs}")