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


class RobomimicReplayImageDatasetRT(BaseImageDataset):
    """
    与 diffusion_policy.dataset.robomimic.replay_image_dataset.RobomimicReplayImageDataset
    保持接口基本一致，但改用 SequenceSamplerRT：
    - 在采样阶段只加载 obs 的前 n_obs_steps；action 加载完整 horizon；
    - 额外返回 window_info: torch.int64 张量（来自 sampler 的 [H,5] 元信息）。
    不引入窗口等长 equalize 或交错顺序，保持与原版训练流程的长度和访问模式尽量一致。
    """

    def __init__(self,
            shape_meta: dict,
            dataset_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            abs_action=False,
            rotation_rep='rotation_6d',  # ignored when abs_action=False
            use_legacy_normalizer=False,
            use_cache=False,
            seed=42,
            val_ratio=0.0,
        n_demo=100,
        # 新增：为“单表版本”提供行数（需与 DataLoader.batch_size 保持一致，且 shuffle=False）
        batch_size: int = 1,
        ):
        self.n_demo = n_demo
        rotation_transformer = RotationTransformer(
            from_rep='axis_angle', to_rep=rotation_rep)

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

        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask

        self.replay_buffer = replay_buffer
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.abs_action = abs_action
        self.n_obs_steps = int(n_obs_steps) if n_obs_steps is not None else 1
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.use_legacy_normalizer = use_legacy_normalizer

        # ==== 基础 episode 视图 ====
        self.episode_ends: np.ndarray = np.asarray(self.replay_buffer.episode_ends[:], dtype=np.int64)
        self.episode_starts: np.ndarray = np.zeros_like(self.episode_ends)
        if len(self.episode_ends) > 0:
            self.episode_starts[1:] = self.episode_ends[:-1]
        self.episode_starts[0] = 0

        # 仅使用训练集 episodes 构建索引与映射
        self.train_episode_ids: np.ndarray = np.nonzero(self.train_mask)[0].astype(np.int64)
        if self.train_episode_ids.size == 0:
            raise RuntimeError('No training episodes available!')

        # 记录每集真实长度与窗口数
        self._ep_real_len = {}
        self._ep_window_num = {}
        self._ep_lpad = {}
        for e in self.train_episode_ids:
            start = int(self.episode_starts[e])
            end = int(self.episode_ends[e])
            real_len = end - start
            if real_len <= 0:
                raise RuntimeError(f"Episode {e} has non-positive length: {real_len}")
            window_num = real_len + self.horizon
            lpad = real_len + 2 * self.horizon - 1
            self._ep_real_len[e] = real_len
            self._ep_window_num[e] = window_num
            self._ep_lpad[e] = lpad

        # 构建 buffer_pos_id 的全局基址（按 episode 升序，对训练集累计）
        self._ep_base = {}
        acc = 0
        for e in self.train_episode_ids:
            self._ep_base[e] = acc
            acc += self._ep_lpad[e]
        self._total_padded_len = acc

        # 预构 episode_pos 元信息（[L_pad,5]）
        # 列为: [warm_up_flag, episode_pos_id, new_episode_pos_id, episode_id, buffer_pos_id]
        self._ep_episode_pos = {}
        for e in self.train_episode_ids:
            real_len = self._ep_real_len[e]
            lpad = self._ep_lpad[e]
            base = self._ep_base[e]
            arr = np.zeros((lpad, 5), dtype=np.int64)
            # new_episode_pos_id
            arr[:, 2] = np.arange(lpad, dtype=np.int64)
            # episode_id
            arr[:, 3] = e
            # warm_up_flag: new_pos < horizon
            arr[: self.horizon, 0] = 1
            # episode_pos_id: 映射到原始位置（warmup=0；中段=pos-h；尾段=real_len-1）
            # 前段 [0..h-1] → 0
            # 中段 [h..h+real_len-1] → 0..real_len-1
            # 尾段 [h+real_len .. lpad-1] → real_len-1
            if real_len > 0:
                # 中段
                mid_start = self.horizon
                mid_end = self.horizon + real_len  # exclusive
                arr[mid_start:mid_end, 1] = np.arange(real_len, dtype=np.int64)
                # 尾段
                if mid_end < lpad:
                    arr[mid_end:, 1] = real_len - 1
            # buffer_pos_id = base + new_episode_pos_id
            arr[:, 4] = base + arr[:, 2]
            self._ep_episode_pos[e] = arr

        # ==== 构建单张全局映射表（行= batch_size；列数对齐到相同长度）====
        self.batch_size = int(batch_size) if batch_size is not None else 1
        if self.batch_size <= 0:
            self.batch_size = 1
        if self.batch_size > len(self.train_episode_ids):
            # 仍允许，但某些行会被大量重复填充
            print(f"[v4] Warning: batch_size({self.batch_size}) > #train_episodes({len(self.train_episode_ids)}). Rows will be padded by repeats.")

        self._rng = np.random.RandomState(seed)
        self._build_global_table()

    def get_validation_dataset(self):
        # 复制并重建映射（使用验证集 episodes）。注意：val 的 batch_size 默认沿用训练集的 batch_size，
        # 请确保 DataLoader 使用相同的 batch_size 且 shuffle=False。
        val_set = copy.copy(self)
        val_set.train_mask = ~self.train_mask
        # 重新计算 episode 视图
        val_set.train_episode_ids = np.nonzero(val_set.train_mask)[0].astype(np.int64)
        val_set._ep_real_len = {}
        val_set._ep_window_num = {}
        val_set._ep_lpad = {}
        for e in val_set.train_episode_ids:
            start = int(val_set.episode_starts[e])
            end = int(val_set.episode_ends[e])
            real_len = end - start
            window_num = real_len + val_set.horizon
            lpad = real_len + 2 * val_set.horizon - 1
            val_set._ep_real_len[e] = real_len
            val_set._ep_window_num[e] = window_num
            val_set._ep_lpad[e] = lpad
        # base 与 episode_pos
        val_set._ep_base = {}
        acc = 0
        for e in val_set.train_episode_ids:
            val_set._ep_base[e] = acc
            acc += val_set._ep_lpad[e]
        val_set._total_padded_len = acc
        val_set._ep_episode_pos = {}
        for e in val_set.train_episode_ids:
            real_len = val_set._ep_real_len[e]
            lpad = val_set._ep_lpad[e]
            base = val_set._ep_base[e]
            arr = np.zeros((lpad, 5), dtype=np.int64)
            arr[:, 2] = np.arange(lpad, dtype=np.int64)
            arr[:, 3] = e
            arr[: val_set.horizon, 0] = 1
            if real_len > 0:
                mid_start = val_set.horizon
                mid_end = val_set.horizon + real_len
                arr[mid_start:mid_end, 1] = np.arange(real_len, dtype=np.int64)
                if mid_end < lpad:
                    arr[mid_end:, 1] = real_len - 1
            arr[:, 4] = base + arr[:, 2]
            val_set._ep_episode_pos[e] = arr
        # 重建单表
        val_set._rng = np.random.RandomState(12345)
        val_set._build_global_table()
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action
        stat = array_to_stats(self.replay_buffer['action'])
        if self.abs_action:
            if stat['mean'].shape[-1] > 10:
                # dual arm
                this_normalizer = robomimic_abs_action_only_dual_arm_normalizer_from_stat(stat)
            else:
                this_normalizer = robomimic_abs_action_only_normalizer_from_stat(stat)
            
            if self.use_legacy_normalizer:
                this_normalizer = normalizer_from_stat(stat)
        else:
            # already normalized
            this_normalizer = get_identity_normalizer_from_stat(stat)
        normalizer['action'] = this_normalizer

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])

            if key.endswith('pos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('quat'):
                # quaternion is in [-1,1] already
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
        # 全局映射表大小（行=batch_size，列=max_cols）
        return int(self.batch_size * self._max_cols)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 限制解码库并行，降低 OpenMP 线程争用
        threadpool_limits(1)

        # 行列映射：id -> (p, q)；p 为行索引（取余），q 为列索引（取整）
        B = self.batch_size
        p = idx % B
        q = idx // B
        if q >= self._max_cols:
            # 理论不会发生（len 已按 B*_max_cols），兜底
            q = self._max_cols - 1
        e = int(self._global_pairs[p, q, 0])
        m = int(self._global_pairs[p, q, 1])

        # 计算窗口索引（相对→绝对），统一使用基值 j = m - (h-1)
        h = self.horizon
        S = self.n_obs_steps
        j_base = m - (h - 1)

        ep_start = int(self.episode_starts[e])
        real_len = int(self._ep_real_len[e])

        # ---- 使用切片 + 边界复制，避免 zarr 高级索引 ----
        def fetch_seq(key: str, length: int) -> np.ndarray:
            # 计算相对区间与边界填充
            start_rel = j_base
            end_rel = j_base + length - 1
            if start_rel >= real_len:
                # 完全越过右端：全用最后一帧
                anchor = self.replay_buffer[key][ep_start + real_len - 1]
                return np.repeat(anchor[None], length, axis=0)
            if end_rel < 0:
                # 完全越过左端：全用第 0 帧
                anchor = self.replay_buffer[key][ep_start + 0]
                return np.repeat(anchor[None], length, axis=0)

            pre_pad = max(0, -start_rel)
            post_pad = max(0, end_rel - (real_len - 1))
            # 中央可用切片范围（相对）
            s_rel = max(0, start_rel)
            e_rel = min(real_len - 1, end_rel)
            if e_rel >= s_rel:
                s_abs = ep_start + s_rel
                e_abs = ep_start + e_rel
                mid = self.replay_buffer[key][s_abs: e_abs + 1]
            else:
                # 理论少见，但兜底：取第 0 帧作为 mid
                mid = self.replay_buffer[key][ep_start + 0: ep_start + 1]

            # 快路径：无需填充，直接返回切片
            if pre_pad == 0 and post_pad == 0:
                return mid

            # 需要填充，构造三段拼接
            parts = []
            if pre_pad > 0:
                parts.append(np.repeat(mid[0:1], pre_pad, axis=0))
            parts.append(mid)
            if post_pad > 0:
                last = parts[-1][-1:]
                parts.append(np.repeat(last, post_pad, axis=0))
            return np.concatenate(parts, axis=0)

        # 取数据
        obs_dict = {}
        for key in self.rgb_keys:
            seq = fetch_seq(key, S)  # (S,H,W,C)
            seq = np.moveaxis(seq, -1, 1).astype(np.float32) / 255.0  # (S,C,H,W)
            obs_dict[key] = seq
        for key in self.lowdim_keys:
            seq = fetch_seq(key, S).astype(np.float32)  # (S,D)
            obs_dict[key] = seq

        actions = fetch_seq('action', h).astype(np.float32)  # (h,D)

        # window_info: 来自每集 episode_pos，在 new_episode_pos 维度上取 [m .. m+h-1]
        ep_pos = self._ep_episode_pos[e]
        window_info = ep_pos[m : m + h]

        out = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(actions),
            'window_info': torch.from_numpy(window_info.astype(np.int64))
        }
        return out

    # ========= 内部构建函数 =========
    def _build_global_table(self):
        """
        构建单张全局映射表，形状 [B, max_cols, 2]，每个元素为 (episode_id, window_m)。
        填充策略：将乱序的 episodes 依次填入当前最短行；最后对齐到 max_cols，
        通过从各行已有元素中随机重复来补齐。
        """
        B = self.batch_size
        ep_ids = list(map(int, list(self.train_episode_ids)))
        self._rng.shuffle(ep_ids)
        # 每行序列
        rows = [[] for _ in range(B)]
        row_lens = np.zeros(B, dtype=np.int64)

        for e in ep_ids:
            wnum = int(self._ep_window_num[e])
            # 找到当前最短的行（若有多条，取索引最小）
            r = int(np.argmin(row_lens))
            # 追加该 episode 的所有窗口 (e, 0..wnum-1)
            rows[r].extend([(e, m) for m in range(wnum)])
            row_lens[r] += wnum

        max_cols = int(row_lens.max()) if B > 0 else 0
        # 补齐每行到相同列数
        for r in range(B):
            deficit = max_cols - len(rows[r])
            if deficit > 0:
                if len(rows[r]) == 0:
                    # 若该行为空，随便挑 episode 的窗口补齐（取第一个 episode 的窗口 0 重复）
                    # 更合理方式是轮询全局 episodes，这里简单实现
                    fallback_e = ep_ids[0]
                    rows[r].extend([(fallback_e, 0)] * deficit)
                else:
                    idxs = self._rng.randint(0, len(rows[r]), size=deficit)
                    rows[r].extend([rows[r][i] for i in idxs])

        # 转为 ndarray [B, max_cols, 2]
        pairs = np.zeros((B, max_cols, 2), dtype=np.int32)
        for r in range(B):
            for c, (e, m) in enumerate(rows[r]):
                pairs[r, c, 0] = e
                pairs[r, c, 1] = m

        self._global_pairs = pairs
        self._max_cols = max_cols


def _convert_actions(raw_actions, abs_action, rotation_transformer):
    actions = raw_actions
    if abs_action:
        is_dual_arm = False
        if raw_actions.shape[-1] == 14:
            # dual arm
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
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    # parse shape_meta
    rgb_keys = list()
    lowdim_keys = list()
    # construct compressors and chunks
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
        # count total steps
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

        # save lowdim data
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
                # make sure we can successfully decode
                _ = zarr_arr[zarr_idx]
                return True
            except Exception as e:
                return False
        
        with tqdm(total=n_steps*len(rgb_keys), desc="Loading image data", mininterval=1.0) as pbar:
            # one chunk per thread, therefore no synchronization needed
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
                                # limit number of inflight tasks
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
    max_abs = np.maximum(stat['max'].max(), np.abs(stat['min']).max())
    scale = np.full_like(stat['max'], fill_value=1/max_abs)
    offset = np.zeros_like(stat['max'])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )
