import torch
import numpy as np

class ActionChunkProcessor:
    def __init__(self, 
                 total_steps: int,
                 action_dim: int,
                 query_frequency: int = 10,
                 num_queries: int = 50,
                 temporal_agg: bool = False,
                 agg_mode: str = "exp",   # "exp", "mean", "last"
                 device: str = "cuda"):
        """
        :param total_steps: 部署的总时间步数
        :param action_dim: 动作维度
        :param query_frequency: 推理的时间间隔
        :param num_queries: 每次预测的 chunk 长度
        :param temporal_agg: 是否启用时间聚合
        :param agg_mode: 聚合模式 ("exp", "mean", "last")
        :param device: 设备
        """
        self.T = total_steps
        self.action_dim = action_dim
        self.query_frequency = query_frequency
        self.num_queries = num_queries
        self.temporal_agg = temporal_agg
        self.agg_mode = agg_mode
        self.device = device

        self.reset()

    def reset(self):
        """复位：清空缓存"""
        self.all_time_actions = torch.zeros((self.T, self.T, self.action_dim), device=self.device)
        self.current_chunk = None
        self.last_query_t = None
        print("[ActionChunkProcessor] Reset done.")

    def update_chunk(self, action_pred: torch.Tensor, t: int):
        """
        更新缓存的动作预测
        :param action_pred: [num_queries, action_dim]
        :param t: 当前时间步
        """
        self.current_chunk = action_pred
        self.last_query_t = t

        end_idx = min(t + self.num_queries, self.T)
        self.all_time_actions[t, t:end_idx] = action_pred[:end_idx - t]

    def get_action(self, t: int):
        """
        根据 t 获取对应的动作
        :param t: 当前时间步
        :return: [1, action_dim]
        """
        if self.temporal_agg:
            actions_for_curr_step = self.all_time_actions[:, t]  # [T, action_dim]
            actions_populated = torch.any(actions_for_curr_step != 0, dim=1)
            actions_for_curr_step = actions_for_curr_step[actions_populated]

            if actions_for_curr_step.shape[0] > 0:
                if self.agg_mode == "exp":
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights /= exp_weights.sum()
                    exp_weights = torch.tensor(exp_weights, device=self.device).unsqueeze(1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)

                elif self.agg_mode == "mean":
                    raw_action = actions_for_curr_step.mean(dim=0, keepdim=True)

                elif self.agg_mode == "last":
                    raw_action = actions_for_curr_step[-1].unsqueeze(0)

                else:
                    raise ValueError(f"Unknown agg_mode: {self.agg_mode}")
            else:
                raw_action = torch.zeros((1, self.action_dim), device=self.device)

        else:
            if self.current_chunk is None:
                raise RuntimeError("No action chunk has been set. Call update_chunk() first.")

            idx = (t - self.last_query_t) % self.current_chunk.shape[0]
            raw_action = self.current_chunk[idx].unsqueeze(0)

        return raw_action


# ================================ Example usage ======================================

# processor = ActionChunkProcessor(
#     total_steps=3000,
#     action_dim=14,
#     query_frequency=16,
#     num_queries=50,
#     temporal_agg=True,
#     agg_mode="exp"
# )

# for t in range(processor.T):
#     # 每 query_frequency 步更新一次 chunk
#     if t % processor.query_frequency == 0:
#         action_pred = self.infer_model.Inference_UR_Dual_Arm(obs, self.current_task)
#         processor.update_chunk(action_pred, t)

#     # 读取动作
#     raw_action = processor.get_action(t)

