from typing import Dict, List
import torch
import numpy as np
import h5py
from tqdm import tqdm
import zarr
import os
import shutil
import copy
import json
import hashlib
from filelock import FileLock
from threadpoolctl import threadpool_limits
import concurrent.futures
import multiprocessing
from omegaconf import OmegaConf
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset, LinearNormalizer
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import get_val_mask
from diffusion_policy.common.normalize_util import (
    robomimic_abs_action_only_normalizer_from_stat,
    robomimic_abs_action_only_dual_arm_normalizer_from_stat,
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats
)
register_codecs()


class RobomimicReplayImageDatasetRTV7(BaseImageDataset):
    """
     核心思想与整体流程（仅说明，不影响运行）：
     1) 物理化预填充（per-episode pre-fill）：
         - 对每个 episode 的 obs 和 action 进行“首部复制 + 尾部复制”的物理化填充，生成一段新的连续缓存；
         - 本实现采用：首部复制 horizon 帧、尾部复制 horizon-1 帧（注意这是一个有意选择，见下方说明）；
            因而每个 episode 的“填充后长度”L_pad = real_len + horizon + (horizon-1) = real_len + 2*horizon - 1；
         - 与很多文档里“首尾各复制 horizon-1 帧”的方案相比，本实现首部多复制了 1 帧，使得 warmup 标记长度为 h（见下文）。

     2) 每集窗口映射 W_e（不单独存表，运行时隐式计算）：
         - 对于某一集，窗口起点 m 给出：
            W_e(m) = (obs_start=m, obs_end=m+n_obs_steps-1, act_start=m, act_end=m+horizon-1)；
         - 每集可取窗口总数 window_num = real_len + horizon（与当前首部复制 h 的实现对应）。

     3) 每个 epoch 的“全局二维映射总表”：
         - 形状为 [batch_size, max_cols]，每个单元存一个二元组 (episode_id, m)；
         - 构造策略为“贪心填充到最短的行 + 随机重复补齐到等长列数”，保证所有行列对齐；
         - 训练时，沿“列”为一个 batch，保证 batch 内样本来自不同 episode，且相邻列大概率延续同一行对应的 episode，提升时序/IO 局部性。

     4) __getitem__ 的映射规则：
         - 给定全局样本下标 idx：行 p = idx % batch_size，列 q = idx // batch_size；
         - 读取 (e, m) = global_pairs[p, q]，按 W_e(m) 切片物理化缓存，直接返回 obs / action；
         - 同时返回 window_info（形状 [horizon, 5]），每一帧包含五个字段：[is_warmup, episode_index, episode_new_pos_id, episode_id, buffer_pos_id]，供上游 workspace 做预热阶段的损失屏蔽等逻辑。

     关于“horizon 与 horizon-1”的对齐说明：
     - 文档里常见的另一种记法是：首尾均复制 horizon-1 帧，此时 L_pad = real_len + 2*(horizon-1)；window_num = real_len + (horizon-1)。
     - 本实现采用“首部复制 horizon、尾部复制 horizon-1”，得到 L_pad = real_len + 2*horizon - 1；window_num = real_len + horizon。
     - 两者在窗口切片的覆盖范围与 warmup 段长度上存在 1 帧的“端点包含/排除”的不同选择；本实现以便于实现“前 h 帧 warmup 标记”的方式组织，功能等价，训练语义一致。

     其他：
     - pad_before/pad_after 不再作为入参，内部由 horizon 决定（首 h，尾 h-1）；
     - window_obs 支持 n_obs_steps（默认为 1，取单帧观测）。
    """

    def __init__(self,
            shape_meta: dict,
            dataset_path: str,
            horizon=1,
            n_obs_steps=None,
            abs_action=False,
            rotation_rep='rotation_6d',  # ignored when abs_action=False
            use_legacy_normalizer=False,
            use_cache=False,
            seed=42,
            val_ratio=0.0,
            n_demo=100,
            batch_size: int = 1,
        ):
        self.n_demo = n_demo
        rotation_transformer = RotationTransformer(
            from_rep='axis_angle', to_rep=rotation_rep)

        # ===== 构建基础 ReplayBuffer（与原始实现一致，可用缓存） =====
        # 说明：
        # - 将 robomimic 的 HDF5 数据集转换为一个统一的 ReplayBuffer（zarr 形式），
        # - 支持可选的“压缩缓存到磁盘并复用”，避免多次重复解析/转码图片带来的开销。
        replay_buffer = None
        if use_cache:
            cache_zarr_path = dataset_path + f'.{n_demo}.' + '.zarr.zip'
            cache_lock_path = cache_zarr_path + '.lock'
            print('Acquiring lock on cache.')
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    try:
                        print('Cache does not exist. Creating!')
                        replay_buffer = _convert_robomimic_to_replay(
                            store=zarr.MemoryStore(), 
                            shape_meta=shape_meta, 
                            dataset_path=dataset_path, 
                            abs_action=abs_action, 
                            rotation_transformer=rotation_transformer,
                            n_demo=n_demo)
                        print('Saving cache to disk.')
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(
                                store=zip_store
                            )
                    except Exception as e:
                        shutil.rmtree(cache_zarr_path)
                        raise e
                else:
                    print('Loading cached ReplayBuffer from Disk.')
                    with zarr.ZipStore(cache_zarr_path, mode='r') as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore())
                    print('Loaded!')
        else:
            replay_buffer = _convert_robomimic_to_replay(
                store=zarr.MemoryStore(), 
                shape_meta=shape_meta, 
                dataset_path=dataset_path, 
                abs_action=abs_action, 
                rotation_transformer=rotation_transformer,
                n_demo=n_demo)

        # ===== 解析 obs 键 =====
        # 说明：将 obs 中的 rgb/low_dim 两类键拆分记录，便于后续分别处理（图像转置/归一化 vs 低维直接拼接）。
        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)

        # ===== 训练/验证划分 =====
        # 说明：依照 seed 与 val_ratio 进行按-episode 的划分，此数据集在构建“物理化缓存 + 全局映射表”时只使用训练 episodes。
        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask

        # ===== 保存基础属性 =====
        self.replay_buffer = replay_buffer
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.abs_action = abs_action
        self.n_obs_steps = int(n_obs_steps) if n_obs_steps is not None else 1
        self.train_mask = train_mask
        self.horizon = int(horizon)
        self.use_legacy_normalizer = use_legacy_normalizer
        self.batch_size = int(batch_size) if batch_size is not None else 1
        if self.batch_size <= 0:
            self.batch_size = 1

        # ===== 基础 episode 边界 =====
        # 说明：由 ReplayBuffer 的 episode_ends 还原出每集的 [start, end)，用于后续按集构建缓存。
        self.episode_ends: np.ndarray = np.asarray(self.replay_buffer.episode_ends[:], dtype=np.int64)
        self.episode_starts: np.ndarray = np.zeros_like(self.episode_ends)
        if len(self.episode_ends) > 0:
            self.episode_starts[1:] = self.episode_ends[:-1]
        self.episode_starts[0] = 0

        # 仅使用训练 episodes 构建缓存与映射
        self.train_episode_ids: np.ndarray = np.nonzero(self.train_mask)[0].astype(np.int64)
        if self.train_episode_ids.size == 0:
            raise RuntimeError('No training episodes available!')

    # ===== 物理化预填充缓存（内存）+ 每集 W_e / episode_pos =====
    # 说明：
    # - 对每个训练集 episode，生成带“首 h、尾 h-1”复制的物理化缓存：obs 与 action 都以相同方式处理；
    # - 对应的每集“可取窗口数” W_num = real_len + h；
    # - episode_pos 的 5 个字段：
    #   [0] is_warmup           ：预热标记（本实现将前 h 帧标为 1，其余为 0），供上游决定是否屏蔽损失；
    #   [1] episode_index       ：在原始 episode 中的帧序号（前部复制段为 0，中段为 0..real_len-1，尾段为 real_len-1）；
    #   [2] episode_new_pos_id  ：在“填充后 episode”中的位置（0..L_pad-1）；
    #   [3] episode_id          ：该样本来自的原始 episode 编号；
    #   [4] buffer_pos_id       ：该样本在“所有训练 episode 的拼接缓存”中的绝对位置（用于全局定位与可视化）。
        h = self.horizon
        S = self.n_obs_steps
        self._ep_cache: Dict[int, Dict[str, np.ndarray]] = {}
        self._ep_episode_pos: Dict[int, np.ndarray] = {}
        self._ep_window_num: Dict[int, int] = {}
        self._ep_lpad: Dict[int, int] = {}
        self._ep_real_len: Dict[int, int] = {}

        # 计算 buffer_pos_id 的全局基址
        # 说明：所有训练 episode 的“填充后缓存”按顺序首尾相接，acc 为累积长度，供第 e 集的样本计算绝对位置。
        base_offsets: Dict[int, int] = {}
        acc = 0
        for e in self.train_episode_ids:
            start = int(self.episode_starts[e])
            end = int(self.episode_ends[e])
            real_len = end - start
            if real_len <= 0:
                raise RuntimeError(f"Episode {e} has non-positive length: {real_len}")
            L_pad = real_len + 2*h - 1
            W_num = real_len + h
            self._ep_real_len[e] = real_len
            self._ep_lpad[e] = L_pad
            self._ep_window_num[e] = W_num
            base_offsets[e] = acc
            acc += L_pad
        self._total_padded_len = acc

        # 实际构建每集缓存
        # 说明：
        # - action 与 obs 都按“首 h / 尾 h-1”的策略进行 np.repeat + np.concatenate；
        # - rgb 仍保持 NHWC 存储（后续 __getitem__ 再转为 NCHW / 归一化到 [0,1]）；
        # - lowdim/action 统一存为 float32；
        # - episode_pos 的五元信息在此一次性构建完毕。
        for e in tqdm(self.train_episode_ids, desc='Building padded episode cache (train)'):
            start = int(self.episode_starts[e])
            end = int(self.episode_ends[e])
            real_len = self._ep_real_len[e]
            L_pad = self._ep_lpad[e]
            # per-episode cache
            ep_cache = {}
            # 处理动作
            act = self.replay_buffer['action'][start:end]
            pre = np.repeat(act[0:1], h, axis=0)
            post = np.repeat(act[-1:], h-1, axis=0)
            act_pad = np.concatenate([pre, act, post], axis=0)
            assert act_pad.shape[0] == L_pad
            ep_cache['action'] = act_pad.astype(np.float32)
            # 处理 obs（rgb/lowdim）
            for key in self.rgb_keys:
                arr = self.replay_buffer[key][start:end]
                pre = np.repeat(arr[0:1], h, axis=0)
                post = np.repeat(arr[-1:], h-1, axis=0)
                pad = np.concatenate([pre, arr, post], axis=0)
                assert pad.shape[0] == L_pad
                ep_cache[key] = pad  # uint8, NHWC
            for key in self.lowdim_keys:
                arr = self.replay_buffer[key][start:end]
                pre = np.repeat(arr[0:1], h, axis=0)
                post = np.repeat(arr[-1:], h-1, axis=0)
                pad = np.concatenate([pre, arr, post], axis=0)
                assert pad.shape[0] == L_pad
                ep_cache[key] = pad.astype(np.float32)
            self._ep_cache[e] = ep_cache

            # episode_pos: [L_pad,5]
            base = base_offsets[e]
            ep_pos = np.zeros((L_pad, 5), dtype=np.int64)
            # new_episode_pos_id
            ep_pos[:, 2] = np.arange(L_pad, dtype=np.int64)
            # episode_id
            ep_pos[:, 3] = e
            # warm_up_flag：本实现选择“前 h 帧”为 warmup
            ep_pos[:h, 0] = 1
            # episode_pos_id（映射到原始 episode 的 index）：
            # 前段（复制）= 0；中段 = 0..real_len-1；尾段（复制）= real_len-1。
            if real_len > 0:
                mid_start = h
                mid_end = h + real_len  # exclusive
                ep_pos[mid_start:mid_end, 1] = np.arange(real_len, dtype=np.int64)
                if mid_end < L_pad:
                    ep_pos[mid_end:, 1] = real_len - 1
            # buffer_pos_id
            ep_pos[:, 4] = base + ep_pos[:, 2]
            self._ep_episode_pos[e] = ep_pos

        # ===== 全局映射表（单表）=====
        # 说明：
        # - 以 batch_size 为行数，逐集“向最短行追加”的方式均衡填充各行；
        # - 记录 (episode_id, m) 对；列数按本 epoch 的“最长行长度”对齐；
        # - 对未满的行进行“随机重复”补齐，保证所有行长度一致；
        # - 最终 __len__ = batch_size * max_cols；__getitem__ 按列组织 batch。
        self._rng = np.random.RandomState(seed)
        self._build_global_table()

    def get_validation_dataset(self):
        # 拷贝并用验证集 episodes 重建缓存与单表
        # 说明：验证集构造流程与训练集一致（但随机种子可不同），便于评估时沿同样的索引/批处理规则读取数据。
        val_set = copy.copy(self)
        val_set.train_mask = ~self.train_mask
        val_set.train_episode_ids = np.nonzero(val_set.train_mask)[0].astype(np.int64)
        if val_set.train_episode_ids.size == 0:
            return val_set

        h = val_set.horizon
        S = val_set.n_obs_steps
        val_set._ep_cache = {}
        val_set._ep_episode_pos = {}
        val_set._ep_window_num = {}
        val_set._ep_lpad = {}
        val_set._ep_real_len = {}
        # base offsets
        base_offsets: Dict[int, int] = {}
        acc = 0
        for e in val_set.train_episode_ids:
            start = int(val_set.episode_starts[e])
            end = int(val_set.episode_ends[e])
            real_len = end - start
            L_pad = real_len + 2*h - 1
            W_num = real_len + h
            val_set._ep_real_len[e] = real_len
            val_set._ep_lpad[e] = L_pad
            val_set._ep_window_num[e] = W_num
            base_offsets[e] = acc
            acc += L_pad
        val_set._total_padded_len = acc

        for e in tqdm(val_set.train_episode_ids, desc='Building padded episode cache (val)'):
            start = int(val_set.episode_starts[e])
            end = int(val_set.episode_ends[e])
            real_len = val_set._ep_real_len[e]
            L_pad = val_set._ep_lpad[e]
            ep_cache = {}
            act = val_set.replay_buffer['action'][start:end]
            pre = np.repeat(act[0:1], h, axis=0)
            post = np.repeat(act[-1:], h-1, axis=0)
            act_pad = np.concatenate([pre, act, post], axis=0)
            ep_cache['action'] = act_pad.astype(np.float32)
            for key in val_set.rgb_keys:
                arr = val_set.replay_buffer[key][start:end]
                pre = np.repeat(arr[0:1], h, axis=0)
                post = np.repeat(arr[-1:], h-1, axis=0)
                pad = np.concatenate([pre, arr, post], axis=0)
                ep_cache[key] = pad
            for key in val_set.lowdim_keys:
                arr = val_set.replay_buffer[key][start:end]
                pre = np.repeat(arr[0:1], h, axis=0)
                post = np.repeat(arr[-1:], h-1, axis=0)
                pad = np.concatenate([pre, arr, post], axis=0)
                ep_cache[key] = pad.astype(np.float32)
            val_set._ep_cache[e] = ep_cache

            base = base_offsets[e]
            ep_pos = np.zeros((L_pad, 5), dtype=np.int64)
            ep_pos[:, 2] = np.arange(L_pad, dtype=np.int64)
            ep_pos[:, 3] = e
            ep_pos[:h, 0] = 1
            if real_len > 0:
                mid_start = h
                mid_end = h + real_len
                ep_pos[mid_start:mid_end, 1] = np.arange(real_len, dtype=np.int64)
                if mid_end < L_pad:
                    ep_pos[mid_end:, 1] = real_len - 1
            ep_pos[:, 4] = base + ep_pos[:, 2]
            val_set._ep_episode_pos[e] = ep_pos

        val_set._rng = np.random.RandomState(12345)
        val_set._build_global_table()
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        # 说明：
        # - 复用项目默认的归一化策略：动作按配置选择绝对/相对与旋转表征变换；
        # - 低维观测根据键名（pos/quat/qpos）选择范围或恒等归一化；
        # - 图像使用 [0,1] 范围归一化（具体在 __getitem__ 时完成通道重排与缩放）。
        normalizer = LinearNormalizer()

        # action
        stat = array_to_stats(self.replay_buffer['action'])
        if self.abs_action:
            if stat['mean'].shape[-1] > 10:
                this_normalizer = robomimic_abs_action_only_dual_arm_normalizer_from_stat(stat)
            else:
                this_normalizer = robomimic_abs_action_only_normalizer_from_stat(stat)
            if self.use_legacy_normalizer:
                this_normalizer = normalizer_from_stat(stat)
        else:
            this_normalizer = get_identity_normalizer_from_stat(stat)
        normalizer['action'] = this_normalizer

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])
            if key.endswith('pos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('quat'):
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('qpos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                raise RuntimeError('unsupported')
            normalizer[key] = this_normalizer

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self):
        return int(self.batch_size * self._max_cols)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 读取规则（与“全局二维映射总表”对齐）：
        # 1) 将全局样本下标 idx 映射为 (p, q) = (idx % B, idx // B)；
        # 2) 取 (e, m) = global_pairs[p, q]；
        # 3) 以 W_e(m) = (m, m+S-1, m, m+h-1) 切片该 episode 的物理化缓存，得到 obs 与 action；
        # 4) 同步返回 window_info（[h, 5]），供上游 workspace 做 warmup 屏蔽（基于字段 is_warmup=window_info[:,0]）。
        threadpool_limits(1)
        B = self.batch_size
        p = idx % B
        q = idx // B
        if q >= self._max_cols:
            q = self._max_cols - 1
        e = int(self._global_pairs[p, q, 0])
        m = int(self._global_pairs[p, q, 1])

        h = self.horizon
        S = self.n_obs_steps
        # 每集窗口映射（物理缓存坐标）
        # 说明：此处不单独存储 W_e，直接由 m 与 (S, h) 即时计算：
        # obs[m : m+S]，action[m : m+h]（右端为闭区间，因此末端索引需 -1）。
        obs_s = m
        obs_e = m + S - 1
        act_s = m
        act_e = m + h - 1
        L_pad = self._ep_lpad[e]
        # 边界裁剪（理论上应在范围内，兜底，防越界）
        obs_s = max(0, min(obs_s, L_pad - 1))
        obs_e = max(0, min(obs_e, L_pad - 1))
        act_s = max(0, min(act_s, L_pad - 1))
        act_e = max(0, min(act_e, L_pad - 1))

        ep_cache = self._ep_cache[e]
        # 取 obs
        # - rgb：NHWC -> NCHW，并缩放到 [0,1]
        # - lowdim：保持 float32
        obs_dict = {}
        for key in self.rgb_keys:
            seq = ep_cache[key][obs_s: obs_e + 1]  # (S,H,W,C)
            seq = np.moveaxis(seq, -1, 1).astype(np.float32) / 255.0  # (S,C,H,W)
            obs_dict[key] = seq
        for key in self.lowdim_keys:
            seq = ep_cache[key][obs_s: obs_e + 1].astype(np.float32)
            obs_dict[key] = seq

        actions = ep_cache['action'][act_s: act_e + 1].astype(np.float32)
        window_info = self._ep_episode_pos[e][act_s: act_e + 1]

        out = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(actions),
            'window_info': torch.from_numpy(window_info.astype(np.int64))
        }
        return out

    # ========= 内部：构建全局映射表 =========
    def _build_global_table(self):
        # 目标：构建形如 [B, max_cols, 2] 的表，第二维存 (episode_id, m)。
        # 步骤：
        # 1) 将训练 episodes 打乱，依次把每个 episode 的所有 m ∈ [0, window_num) 追加到“当前最短的行”；
        # 2) 统计各行长度，取最大者为 max_cols；
        # 3) 对每一行，用“随机重复本行已有单元”的方式补到 max_cols；
        # 4) __len__ 返回 B*max_cols；第 q 列形成一个 batch（按 __getitem__ 中的 p=idx%B, q=idx//B）。
        B = self.batch_size
        ep_ids = list(map(int, list(self.train_episode_ids)))
        self._rng.shuffle(ep_ids)
        rows = [[] for _ in range(B)]
        row_lens = np.zeros(B, dtype=np.int64)

        for e in ep_ids:
            wnum = int(self._ep_window_num[e])
            r = int(np.argmin(row_lens))
            rows[r].extend([(e, m) for m in range(wnum)])
            row_lens[r] += wnum

        max_cols = int(row_lens.max()) if B > 0 else 0
        for r in range(B):
            deficit = max_cols - len(rows[r])
            if deficit > 0:
                if len(rows[r]) == 0:
                    fallback_e = ep_ids[0]
                    rows[r].extend([(fallback_e, 0)] * deficit)
                else:
                    idxs = self._rng.randint(0, len(rows[r]), size=deficit)
                    rows[r].extend([rows[r][i] for i in idxs])

        pairs = np.zeros((B, max_cols, 2), dtype=np.int32)
        for r in range(B):
            for c, (e, m) in enumerate(rows[r]):
                pairs[r, c, 0] = e
                pairs[r, c, 1] = m

        self._global_pairs = pairs
        self._max_cols = max_cols


def _convert_actions(raw_actions, abs_action, rotation_transformer):
    # 说明：若配置为绝对动作 + 指定旋转表征，则将 robomimic 的原始动作（含 axis-angle）转换到目标空间。
    #       支持单臂/双臂的拼接与展平，返回 float32。
    actions = raw_actions
    if abs_action:
        is_dual_arm = False
        if raw_actions.shape[-1] == 14:
            raw_actions = raw_actions.reshape(-1,2,7)
            is_dual_arm = True

        pos = raw_actions[...,:3]
        rot = raw_actions[...,3:6]
        gripper = raw_actions[...,6:]
        rot = rotation_transformer.forward(rot)
        raw_actions = np.concatenate([
            pos, rot, gripper
        ], axis=-1).astype(np.float32)
    
        if is_dual_arm:
            raw_actions = raw_actions.reshape(-1,20)
        actions = raw_actions
    return actions


def _convert_robomimic_to_replay(store, shape_meta, dataset_path, abs_action, rotation_transformer, 
        n_workers=None, max_inflight_tasks=None, n_demo=100):
    # 说明：robomimic(HDF5) -> ReplayBuffer(zarr) 的一次性转换：
    # - 低维/动作：按 episode 串接为单一数组；动作可选绝对坐标与旋转变换；
    # - 图像：逐帧 JPEG2000 编码存入 zarr（chunks=(1,H,W,C)），多线程流水写入；
    # - meta：记录 episode_ends 以恢复分集边界。
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    rgb_keys = list()
    lowdim_keys = list()
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        shape = attr['shape']
        type = attr.get('type', 'low_dim')
        if type == 'rgb':
            rgb_keys.append(key)
        elif type == 'low_dim':
            lowdim_keys.append(key)
    
    root = zarr.group(store)
    data_group = root.require_group('data', overwrite=True)
    meta_group = root.require_group('meta', overwrite=True)

    with h5py.File(dataset_path) as file:
        demos = file['data']
        episode_ends = list()
        prev_end = 0
        for i in range(n_demo):
            demo = demos[f'demo_{i}']
            episode_length = demo['actions'].shape[0]
            episode_end = prev_end + episode_length
            prev_end = episode_end
            episode_ends.append(episode_end)
        n_steps = episode_ends[-1]
        episode_starts = [0] + episode_ends[:-1]
        _ = meta_group.array('episode_ends', episode_ends, 
            dtype=np.int64, compressor=None, overwrite=True)

        # 低维与动作
        for key in tqdm(lowdim_keys + ['action'], desc="Loading lowdim data"):
            data_key = 'obs/' + key
            if key == 'action':
                data_key = 'actions'
            this_data = list()
            for i in range(n_demo):
                demo = demos[f'demo_{i}']
                this_data.append(demo[data_key][:].astype(np.float32))
            this_data = np.concatenate(this_data, axis=0)
            if key == 'action':
                this_data = _convert_actions(
                    raw_actions=this_data,
                    abs_action=abs_action,
                    rotation_transformer=rotation_transformer
                )
                assert this_data.shape == (n_steps,) + tuple(shape_meta['action']['shape'])
            else:
                assert this_data.shape == (n_steps,) + tuple(shape_meta['obs'][key]['shape'])
            _ = data_group.array(
                name=key,
                data=this_data,
                shape=this_data.shape,
                chunks=this_data.shape,
                compressor=None,
                dtype=this_data.dtype
            )
        
        def img_copy(zarr_arr, zarr_idx, hdf5_arr, hdf5_idx):
            try:
                zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
                _ = zarr_arr[zarr_idx]
                return True
            except Exception:
                return False
        
        with tqdm(total=n_steps*len(rgb_keys), desc="Loading image data", mininterval=1.0) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = set()
                for key in rgb_keys:
                    data_key = 'obs/' + key
                    shape = tuple(shape_meta['obs'][key]['shape'])
                    c,h,w = shape
                    this_compressor = Jpeg2k(level=50)
                    img_arr = data_group.require_dataset(
                        name=key,
                        shape=(n_steps,h,w,c),
                        chunks=(1,h,w,c),
                        compressor=this_compressor,
                        dtype=np.uint8
                    )
                    for episode_idx in range(n_demo):
                        demo = demos[f'demo_{episode_idx}']
                        hdf5_arr = demo['obs'][key]
                        for hdf5_idx in range(hdf5_arr.shape[0]):
                            if len(futures) >= max_inflight_tasks:
                                completed, futures = concurrent.futures.wait(futures, 
                                    return_when=concurrent.futures.FIRST_COMPLETED)
                                for f in completed:
                                    if not f.result():
                                        raise RuntimeError('Failed to encode image!')
                                pbar.update(len(completed))

                            zarr_idx = episode_starts[episode_idx] + hdf5_idx
                            futures.add(
                                executor.submit(img_copy, 
                                    img_arr, zarr_idx, hdf5_arr, hdf5_idx))
                completed, futures = concurrent.futures.wait(futures)
                for f in completed:
                    if not f.result():
                        raise RuntimeError('Failed to encode image!')
                pbar.update(len(completed))

    replay_buffer = ReplayBuffer(root)
    return replay_buffer


def normalizer_from_stat(stat):
    # 说明：legacy 线性归一化（可选），取全量最大绝对值做对称缩放。
    max_abs = np.maximum(stat['max'].max(), np.abs(stat['min']).max())
    scale = np.full_like(stat['max'], fill_value=1/max_abs)
    offset = np.zeros_like(stat['max'])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )
