"""
SamplerRT: 基于 sampler.py 的窗口采样器（保持 create_indices 语义不变），
在不改变输出窗口长度（sequence_length = horizon）的前提下：
 - 对 obs 键仅按需加载前 n_obs_steps 帧（key_first_k 技巧，显著减小 I/O/RAM）
 - 对 action 键始终加载完整 horizon 帧（不做截断）

用法建议：
 - 构造时传入 obs_keys（需要节省内存的观测键列表）、n_obs_steps 与 action_key（默认 'action'）。

注意：
 - 本采样器只控制“从 ReplayBuffer 读取多少帧放入临时 sample”；
   输出 data 的时间维始终是 sequence_length（通过端点填充补齐），以对齐训练期窗口。
 - 对启用了 key_first_k 的 obs 键，sample 中超过 k_data 的部分不会从存储加载，
   但最终 data 会被正确用端点值填充，不影响后续 compute_loss。
"""
from typing import Optional, Dict, List
import numpy as np
from diffusion_policy.common.replay_buffer import ReplayBuffer
from .sampler import create_indices  # 复用基础版的索引构建


class SequenceSamplerRT:
    def __init__(self, 
        replay_buffer: ReplayBuffer, 
        sequence_length: int,
        pad_before: int = 0,
        pad_after: int = 0,
        keys: Optional[List[str]] = None,
        episode_mask: Optional[np.ndarray] = None,
        # RT 定制参数（简化：不再暴露 key_first_k，由内部根据 obs_keys + n_obs_steps 自动处理）
        obs_keys: Optional[List[str]] = None,
        n_obs_steps: Optional[int] = None,
        action_key: str = 'action',
        # 是否将每个 episode 的窗口数统一到同一值（不足的复制最后一个窗口）
        equalize_windows_per_episode: bool = False,
    ):
        """
        参数：
          - replay_buffer: 扁平化的回放缓冲区（支持按键索引与切片）
          - sequence_length: 窗口统一长度（通常等于 horizon）
          - pad_before/pad_after: 允许窗口跨 episode 边界的补边幅度
          - keys: 需要返回的键集；默认使用 replay_buffer.keys()
          - episode_mask: 指定哪些 episode 可被采样（如训练/验证划分）
          - obs_keys: 观测键列表（将为这些键应用 n_obs_steps 的按需加载）
          - n_obs_steps: 观测时间步数（仅对 obs_keys 生效）
          - action_key: 动作键名（该键始终加载完整 horizon）
        """
        super().__init__()
        assert sequence_length >= 1

        if keys is None:
            keys = list(replay_buffer.keys())

        episode_ends = replay_buffer.episode_ends[:]
        if episode_mask is None:
            episode_mask = np.ones(episode_ends.shape, dtype=bool)

        if np.any(episode_mask):
            indices = create_indices(
                episode_ends,
                sequence_length=sequence_length,
                pad_before=pad_before,
                pad_after=pad_after,
                episode_mask=episode_mask,
            )
        else:
            indices = np.zeros((0, 4), dtype=np.int64)

        # 组装内部按需加载策略：
        # - 对 obs_keys 应用 n_obs_steps（若给出）
        # - 动作键不做裁剪，始终完整加载
        _key_first_k: Dict[str, int] = {}
        if (obs_keys is not None) and (n_obs_steps is not None):
            for k in obs_keys:
                if k != action_key:
                    _key_first_k[k] = int(n_obs_steps)

        # 保存状态
        self.indices = indices
        self.keys = list(keys)  # 防止 OmegaConf list 性能问题
        self.sequence_length = sequence_length
        self.replay_buffer = replay_buffer
        self.key_first_k = _key_first_k
        self.action_key = action_key
        self.pad_before = int(pad_before)
        self.pad_after = int(pad_after)
        self.episode_mask = episode_mask
        self.episode_ends = episode_ends
        # 计算 episode 起点
        self.episode_starts = np.zeros_like(self.episode_ends)
        if len(self.episode_ends) > 0:
            self.episode_starts[1:] = self.episode_ends[:-1]

        # 预计算：每个 episode 能产生多少个窗口，以及其在全局 indices 中的偏移
        self._build_episode_window_stats()
        # 建立与当前 self.indices 相匹配的拼接映射（未等长对齐）
        self._build_concat_mapping(equalized=False)
        # 可选：将每个 episode 的窗口数对齐（复制最后一个窗口行）
        if equalize_windows_per_episode:
            self._equalize_windows_per_episode()

    def __len__(self):
        return len(self.indices)

    def sample_sequence(self, idx: int):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]
        result: Dict[str, np.ndarray] = dict()

        for key in self.keys:
            input_arr = self.replay_buffer[key]

            # 动作键：始终完整加载窗口片段
            if key == self.action_key:
                sample = input_arr[buffer_start_idx:buffer_end_idx]
            # 观测键（或其它键）且启用 key_first_k：只加载前 K 帧
            elif key in self.key_first_k:
                n_data = buffer_end_idx - buffer_start_idx
                k_data = min(self.key_first_k[key], n_data)
                # 预分配并仅写入前 k 帧（剩余部分稍后通过端点填充到 data）
                sample = np.full((n_data,) + input_arr.shape[1:], fill_value=np.nan, dtype=input_arr.dtype)
                if k_data > 0:
                    sample[:k_data] = input_arr[buffer_start_idx: buffer_start_idx + k_data]
            else:
                # 默认：整段加载
                sample = input_arr[buffer_start_idx:buffer_end_idx]

            # 统一的端点填充到固定窗口长度（sequence_length）
            data = sample
            if (sample_start_idx > 0) or (sample_end_idx < self.sequence_length):
                data = np.zeros(
                    shape=(self.sequence_length,) + input_arr.shape[1:],
                    dtype=input_arr.dtype,
                )
                # 头部填充：用 sample[0]
                if sample_start_idx > 0:
                    # 防御：若 sample 全为 NaN（极端情况 n_data=0），跳过赋值
                    if (len(sample) > 0):
                        data[:sample_start_idx] = sample[0]
                # 尾部填充：用 sample[-1]
                if sample_end_idx < self.sequence_length:
                    if (len(sample) > 0):
                        data[sample_end_idx:] = sample[-1]
                # 中间有效段拷贝
                if (len(sample) > 0):
                    data[sample_start_idx:sample_end_idx] = sample
            result[key] = data

        # ---- 附加窗口元信息（可被上游像 action 一样读取）----
        # 计算 (ep_id, 本地窗口 id) 以及各类统计
        # 使用当前拼接映射（支持对齐与非对齐两种模式）
        sel_ids = self._concat_sel_ids
        cum = self._concat_cum_counts
        # 在 cum 中二分查找所属的 episode 段
        # cum 形如 [0, c0, c0+c1, ...]
        k = int(np.searchsorted(cum, idx, side='right') - 1)
        k = max(0, min(k, len(sel_ids) - 1))
        ep_id = int(sel_ids[k])
        local_wid = int(idx - cum[k])

        ep_len = int(self.episode_length(ep_id))
        orig_win_count = int(self._ep_window_counts[ep_id])
        # 当前 indices 所代表的每集窗口数（对齐后为 max_count，否则为原始）
        post_win_count = int(self._concat_counts_by_sel[k]) if k < len(self._concat_counts_by_sel) else orig_win_count

        # 构建 [H, 5] 的整型张量（每个时间步相同），便于与 action 同形时间维拼接与读取
        info_row = np.array([
            ep_id,               # 1) episode id
            ep_len,              # 2) episode 原始长度
            orig_win_count,      # 3) 切分的原始窗口数（对齐前）
            post_win_count,      # 4) 填充/对齐后的窗口数（当前 indices 的每集窗口数）
            local_wid            # 5) 当前窗口在该 episode 中的本地 id
        ], dtype=np.int64)
        window_info = np.repeat(info_row.reshape(1, 5), repeats=self.sequence_length, axis=0)
        result['window_info'] = window_info

        return result

    # -------- 辅助：基于 pad_before/pad_after 与 sequence_length 的逐 episode 窗口统计 --------
    def _build_episode_window_stats(self):
        episode_ends = self.episode_ends
        mask = self.episode_mask
        seq_len = self.sequence_length
        pad_b, pad_a = self.pad_before, self.pad_after
        n_eps = len(episode_ends)
        starts = np.zeros(n_eps, dtype=np.int64)
        starts[1:] = episode_ends[:-1]
        counts = np.zeros(n_eps, dtype=np.int64)
        for i in range(n_eps):
            if (mask is not None) and (not mask[i]):
                counts[i] = 0
                continue
            ep_len = int(episode_ends[i] - starts[i])
            min_start = -min(max(pad_b, 0), seq_len-1)
            max_start = ep_len - seq_len + min(max(pad_a, 0), seq_len-1)
            counts[i] = max(0, (max_start - min_start + 1))
        self._ep_window_counts = counts
        self._ep_window_offsets = np.zeros_like(counts)
        if len(counts) > 0:
            self._ep_window_offsets[1:] = np.cumsum(counts)[:-1]

    def episode_window_count(self, ep_id: int) -> int:
        """返回第 ep_id 条 episode 的窗口数量（已考虑 mask 与 pad）。"""
        return int(self._ep_window_counts[ep_id])

    def episode_window_global_offset(self, ep_id: int) -> int:
        """返回第 ep_id 条 episode 的窗口在全局 indices 中的起始偏移。"""
        return int(self._ep_window_offsets[ep_id])

    def map_episode_window_to_global(self, ep_id: int, w_local: int) -> int:
        """把 (ep_id, 本地窗口序号) 映射到全局窗口 idx。"""
        return self.episode_window_global_offset(ep_id) + int(w_local)

    # -------- 统计/等长窗对齐辅助 --------
    def selected_episode_ids(self) -> np.ndarray:
        """返回被采样的 episode 下标（考虑 episode_mask）。"""
        if self.episode_mask is None:
            return np.arange(len(self.episode_ends), dtype=np.int64)
        return np.nonzero(self.episode_mask)[0].astype(np.int64)

    def episode_length(self, ep_id: int) -> int:
        """返回第 ep_id 条 episode 的长度（帧数）。"""
        start = 0 if ep_id == 0 else int(self.episode_ends[ep_id-1])
        end = int(self.episode_ends[ep_id])
        return end - start

    def _equalize_windows_per_episode(self):
        """将每个 episode 的窗口数对齐到同一值：
        - 取 max_count = 所有被采样 episode 的最大窗口数
        - 对每个 episode，若窗口数不足，则复制其“最后一个窗口”直至补齐
        - 更新 self.indices 为对齐后的新索引数组
        """
        counts = self._ep_window_counts
        offsets = self._ep_window_offsets
        sel_ids = self.selected_episode_ids()
        if len(sel_ids) == 0:
            return
        max_count = int(np.max(counts[sel_ids]))
        if np.all(counts[sel_ids] == max_count):
            return  # 已等长，无需处理

        new_rows = []
        raw = self.indices
        # 记录等长对齐后的每个被选 episode 的窗口数（均为 max_count）
        counts_by_sel = []
        for ep_id in sel_ids:
            c = int(counts[ep_id])
            off = int(offsets[ep_id])
            if c <= 0:
                raise RuntimeError(f"Episode {ep_id} 无可用窗口，无法对齐窗口数！请检查 pad/horizon 设置或丢弃该 episode。")
            rows = raw[off:off+c]
            if c < max_count:
                pad_rows = np.repeat(rows[-1:], max_count - c, axis=0)
                rows = np.concatenate([rows, pad_rows], axis=0)
            new_rows.append(rows)
            counts_by_sel.append(max_count)
        self.indices = np.concatenate(new_rows, axis=0)
        # 构建当前拼接映射（等长对齐模式）
        self._build_concat_mapping(equalized=True, counts_by_sel=np.asarray(counts_by_sel, dtype=np.int64), sel_ids=sel_ids)

    # --------- 当前 indices 的拼接映射（把全局窗口 idx -> (ep_id, 本地窗口 id)） ---------
    def _build_concat_mapping(self, equalized: bool, counts_by_sel: Optional[np.ndarray] = None, sel_ids: Optional[np.ndarray] = None):
        # 选择被启用的 episode 列表及其窗口数（原始或对齐后的）
        if sel_ids is None:
            sel_ids = self.selected_episode_ids()
        else:
            sel_ids = np.asarray(sel_ids, dtype=np.int64)

        if counts_by_sel is None:
            # 未对齐：使用原始 counts
            counts_by_sel = self._ep_window_counts[sel_ids]
        else:
            counts_by_sel = np.asarray(counts_by_sel, dtype=np.int64)

        # 累计和用于二分
        cum = np.zeros(len(sel_ids) + 1, dtype=np.int64)
        if len(sel_ids) > 0:
            cum[1:] = np.cumsum(counts_by_sel)

        self._concat_sel_ids = sel_ids
        self._concat_counts_by_sel = counts_by_sel
        self._concat_cum_counts = cum
        self._eq_mode = bool(equalized)
