import numpy as np

class BaselineCalculator2:
    def __init__(self,
                 window_size=15,
                 weight_decay=0.85,      # 历史轨迹权重指数衰减
                 base_beta=0.25,         # 跨 epoch EMA 权重
                 min_beta=0.05,
                 local_smooth_alpha=0.25 # 时间轴平滑
                 ):
        self.window_size = window_size
        self.weight_decay = weight_decay
        self.base_beta = base_beta
        self.min_beta = min_beta
        self.local_smooth_alpha = local_smooth_alpha

        self.history = []              # [(returns, times)]
        self.global_baseline = None    # (times, values)

    # 收集历史轨迹
    def update2(self, returns, times):
        self.history.append((np.array(returns), np.array(times)))
        if len(self.history) > self.window_size:
            self.history.pop(0)

    # 计算 baseline
    def get_baseline2(self, ep, cur_returns, cur_times):
        cur_returns = np.array(cur_returns)
        cur_times = np.array(cur_times)
        T = len(cur_returns)

        # 没有历史时 baseline=当前奖励
        if len(self.history) == 0 or ep <= 2:
            baseline = cur_returns.copy()
            self.global_baseline = (cur_times, baseline)
            return baseline

        num_hist = len(self.history)

        # -------- 1) 历史轨迹权重：指数衰减（最近权重最高） --------
        weights = np.array([self.weight_decay ** (num_hist - 1 - i) for i in range(num_hist)])
        weights = weights / (weights.sum() + 1e-12)

        # -------- 2) 每条历史轨迹插值对齐到当前时间轴 --------
        aligned = []
        for (hist_ret, hist_time), w in zip(self.history, weights):
            interp = np.interp(cur_times, hist_time, hist_ret,
                               left=hist_ret[0], right=hist_ret[-1])
            aligned.append(interp * w)

        # 合并成历史基线
        hist_baseline = np.sum(aligned, axis=0)

        # -------- 3) 时间轴局部平滑（去噪） --------
        alpha = self.local_smooth_alpha
        smoothed = np.empty_like(hist_baseline)
        smoothed[0] = hist_baseline[0]
        for i in range(1, T):
            smoothed[i] = alpha * hist_baseline[i] + (1 - alpha) * smoothed[i - 1]

        # -------- 4) 动态 beta（避免 std 大时过度依赖历史） --------
        reward_std = float(np.std(cur_returns))
        beta = self.base_beta / (1.0 + 0.5 * reward_std)
        beta = float(np.clip(beta, self.min_beta, self.base_beta))

        # -------- 5) 单层跨 epoch EMA --------
        if self.global_baseline is None:
            final_baseline = smoothed
        else:
            prev_times, prev_vals = self.global_baseline
            prev_interp = np.interp(cur_times, prev_times, prev_vals,
                                    left=prev_vals[0], right=prev_vals[-1])
            final_baseline = beta * smoothed + (1 - beta) * prev_interp

        # 更新
        self.global_baseline = (cur_times, final_baseline.copy())
        return final_baseline.copy()
