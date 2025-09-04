import numpy as np
import matplotlib.pyplot as plt

class JointTrajectoryPlotter:
    def __init__(self, save_path='joint_vs_loss.png'):
        self.save_path = save_path

    def plot(self, pred_actions, ground_actions, l2_losses):
        pred_actions = np.array(pred_actions)
        ground_actions = np.array(ground_actions)
        l2_losses = np.array(l2_losses)

        num_joints = pred_actions.shape[1]
        half = num_joints // 2   # 前一半 & 后一半

        colors = plt.cm.tab20(np.linspace(0, 1, num_joints))

        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

        # ====== 子图1：前一半关节 ======
        ax_front = axes[0]
        for j in range(0, half):
            ax_front.plot(pred_actions[:, j], color=colors[j], linestyle='-', label=f'Pred J{j}')
            ax_front.plot(ground_actions[:, j], color=colors[j], linestyle='--', label=f'GT J{j}')
        ax_front.set_ylabel('Joint Value')
        ax_front.set_title(f'First Half Joints (0–{half-1})')
        ax_front.legend(loc='upper right', ncol=4)

        # ====== 子图2：后一半关节 ======
        ax_back = axes[1]
        for j in range(half, num_joints):
            ax_back.plot(pred_actions[:, j], color=colors[j], linestyle='-', label=f'Pred J{j}')
            ax_back.plot(ground_actions[:, j], color=colors[j], linestyle='--', label=f'GT J{j}')
        ax_back.set_ylabel('Joint Value')
        ax_back.set_title(f'Second Half Joints ({half}–{num_joints-1})')
        ax_back.legend(loc='upper right', ncol=4)

        # ====== 子图3：L2 Loss ======
        ax_loss = axes[2]
        ax_loss.plot(l2_losses, color='black', linestyle='-', linewidth=1.5, label='L2 Loss')
        ax_loss.set_xlabel('Step')
        ax_loss.set_ylabel('L2 Loss')
        ax_loss.set_title('L2 Loss Curve')
        ax_loss.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(self.save_path)
        plt.close()
