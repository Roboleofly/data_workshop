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
        colors = plt.cm.tab10(np.linspace(0, 1, num_joints))  # 每个关节一个颜色

        fig, ax1 = plt.subplots(figsize=(12, 6))

        # 绘制关节预测和GT
        for j in range(num_joints):
            ax1.plot(pred_actions[:, j], color=colors[j], linestyle='-', label=f'Pred Joint {j}')
            ax1.plot(ground_actions[:, j], color=colors[j], linestyle='--', label=f'GT Joint {j}')

        ax1.set_xlabel('Step')
        ax1.set_ylabel('Joint Value')
        ax1.set_title('Predicted vs Ground Truth Joint Trajectories & L2 Loss')
        
        # 第二个y轴绘制L2 Loss
        ax2 = ax1.twinx()
        ax2.plot(l2_losses, color='black', linestyle='-', linewidth=1.5, label='L2 Loss')
        ax2.set_ylabel('L2 Loss')

        # 合并图例
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

        plt.tight_layout()
        plt.savefig(self.save_path)
        plt.close()
