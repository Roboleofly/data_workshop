import threading
import time
import torch

class AsyncActionInferencer:
    def __init__(self, infer_func, processor, query_frequency=16, sleep_time=0.01):
        """
        :param infer_func: 推理函数, 输入obs,task -> action_chunk [num_queries, action_dim]
        :param processor: ActionChunkProcessor 实例
        :param query_frequency: 推理频率 (多少步触发一次推理)
        :param sleep_time: 后台线程休眠间隔
        """
        self.infer_func = infer_func
        self.processor = processor
        self.query_frequency = query_frequency
        self.sleep_time = sleep_time

        self.obs = None
        self.task = None
        self.t = 0

        self._thread = None
        self._running = False
        self._lock = threading.Lock()

    def start(self):
        """启动后台推理线程"""
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self):
        """停止后台推理线程"""
        self._running = False
        if self._thread is not None:
            self._thread.join()

    def reset(self):
        """复位"""
        self.processor.reset()
        self.t = 0

    def set_obs_task(self, obs, task):
        """更新观测和任务"""
        with self._lock:
            self.obs = obs
            self.task = task

    def step(self, t):
        """
        主循环调用，返回动作
        """
        self.t = t
        return self.processor.get_action(t)

    def _worker(self):
        """后台线程：异步推理并更新 chunk"""
        while self._running:
            with self._lock:
                obs = self.obs
                task = self.task
                t = self.t

            if obs is not None and task is not None and (t % self.query_frequency == 0):
                try:
                    # VLA 推理
                    action_pred = self.infer_func(obs, task)  # [num_queries, action_dim]
                    self.processor.update_chunk(action_pred, t)
                except Exception as e:
                    print(f"[AsyncActionInferencer] 推理异常: {e}")

            time.sleep(self.sleep_time)


# ================================ Example usage ======================================

# 初始化 processor 和 inferencer
# processor = ActionChunkProcessor(total_steps=3000, action_dim=14, query_frequency=16, num_queries=50)
# inferencer = AsyncActionInferencer(self.infer_model.Inference_UR_Dual_Arm, processor, query_frequency=16)

# # 启动后台线程
# inferencer.start()

# for t in range(processor.T):
#     obs = self.robot_station.get_obs()
#     inferencer.set_obs_task(obs, self.current_task)

#     # 获取动作 (不会卡顿)
#     raw_action = inferencer.step(t)

