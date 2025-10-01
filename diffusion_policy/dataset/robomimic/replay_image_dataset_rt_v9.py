from typing import Dict, List, Tuple, Optional
import math
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
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
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
    def __init__(self,
            shape_meta: dict,
            dataset_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            abs_action=False,
            rotation_rep='rotation_6d', # ignored when abs_action=False
            use_legacy_normalizer=False,
            use_cache=False,
            seed=42,
            val_ratio=0.0,
            n_demo=100,
            # 新增参数：控制返回整条episode时的episode数量
            episodes_per_item=1,
        ):
        """
        初始化Robomimic数据集加载器
        
        参数:
            shape_meta: 数据形状元数据
            dataset_path: 数据集路径
            horizon: 预测步长
            pad_before: 序列前填充
            pad_after: 序列后填充
            n_obs_steps: 观测步数
            abs_action: 是否使用绝对动作
            rotation_rep: 旋转表示方法
            use_legacy_normalizer: 是否使用旧版归一化器
            use_cache: 是否使用缓存
            seed: 随机种子
            val_ratio: 验证集比例
            n_demo: 使用的演示数量
            episodes_per_item: 每次返回的episode数量
        """
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
                    # 缓存不存在，创建新缓存
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
                            replay_buffer.save_to_store(store=zip_store)
                    except Exception as e:
                        # 清理无效缓存
                        if os.path.exists(cache_zarr_path):
                            shutil.rmtree(cache_zarr_path)
                        else: print('No cache to clean up.')
                        raise e
                else:
                    # 加载已存在的缓存
                    print('Loading cached ReplayBuffer from Disk.')
                    with zarr.ZipStore(cache_zarr_path, mode='r') as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore())
                    print('Loaded!')
        else:
            # 不使用缓存，直接加载数据
            replay_buffer = _convert_robomimic_to_replay(
                store=zarr.MemoryStore(), 
                shape_meta=shape_meta, 
                dataset_path=dataset_path, 
                abs_action=abs_action, 
                rotation_transformer=rotation_transformer,
                n_demo=n_demo)

        # 提取RGB和低维键
        rgb_keys = []
        lowdim_keys = []
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            data_type = attr.get('type', 'low_dim')
            if data_type == 'rgb':
                rgb_keys.append(key)
            elif data_type == 'low_dim':
                lowdim_keys.append(key)
        
        # 创建训练/验证掩码
        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        
        # 存储关键属性
        self.replay_buffer = replay_buffer
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.abs_action = abs_action
        self.n_obs_steps = n_obs_steps
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.use_legacy_normalizer = use_legacy_normalizer
        self.episodes_per_item = int(episodes_per_item)# 每个 dataset 索引返回的 episode 数量
        
        # 计算episode范围
        episode_ends = replay_buffer.episode_ends[:]
        episode_starts = [0] + episode_ends[:-1].tolist()
        
        # 仅存储训练集的episode范围
        self.episode_ranges = []
        for i in range(len(episode_starts)):
            if train_mask[i]:
                self.episode_ranges.append((episode_starts[i], episode_ends[i])) # 存储episode的起止索引

        # 设置当前使用的episode范围
        self.episode_ranges_current = self.episode_ranges
        
        # 计算最大episode长度用于填充
        if self.episode_ranges_current:
            self.max_episode_len = max(e - s for s, e in self.episode_ranges_current)
        else:
            self.max_episode_len = None

        # 初始化阶段一次性做等长一致性检查（基于元数据的起止索引，不触发数据读取）
        if self.max_episode_len is not None:
            for (s, e) in self.episode_ranges_current:
                if (e - s) != self.max_episode_len:
                    raise RuntimeError(
                        f"Episode range length mismatch in metadata: got {e - s}, "
                        f"expected {self.max_episode_len}. Please rebuild cache to ensure equal-length episodes.")

    # 重构：验证集处理逻辑
    def get_validation_dataset(self):
        """创建验证集副本"""
        val_set = copy.copy(self)
        
        # 计算验证集的episode范围
        episode_ends = self.replay_buffer.episode_ends[:]
        episode_starts = [0] + episode_ends[:-1].tolist()
        
        val_set.episode_ranges = []
        for i in range(len(episode_starts)):
            if not self.train_mask[i]:
                val_set.episode_ranges.append((episode_starts[i], episode_ends[i]))
        
        val_set.episode_ranges_current = val_set.episode_ranges
        
        # 计算验证集的最大长度，用于填充
        if val_set.episode_ranges:
            val_set.max_episode_len = max(e - s for s, e in val_set.episode_ranges)
        else:
            val_set.max_episode_len = None
        
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        """创建数据归一化器（严格排除填充帧）。

        基于 original_episode_lengths 仅使用每条 episode 的有效段做统计，
        避免末帧重复填充带来的统计偏差。适用于 train/val 两个 split，
        因为会遍历 self.episode_ranges_current。
        """
        normalizer = LinearNormalizer()

        # 若没有元数据（例如旧缓存），退回到旧行为
        try:
            orig_lengths = self.replay_buffer.meta['original_episode_lengths'][:]
            max_len = int(self.replay_buffer.meta['max_episode_len'][0])
        except Exception:
            orig_lengths = None
            max_len = None

        def gather_valid_numpy(key: str) -> np.ndarray:
            """拼接当前 split 下的有效（未填充）片段，返回 [N_valid, ...] np.ndarray"""
            if orig_lengths is None or max_len is None:
                # 旧缓存兜底：使用整个数组
                return np.array(self.replay_buffer[key])

            parts = []
            for start, end in self.episode_ranges_current:
                # 统一长度缓存下，start/end 对应等长切片
                ep_idx = start // max_len
                valid_len = int(orig_lengths[ep_idx])
                if valid_len <= 0:
                    continue
                # 从起点取有效段
                arr = self.replay_buffer[key][start:start + valid_len]
                parts.append(np.asarray(arr))
            if len(parts) == 0:
                # 空数据兜底：返回一个形状合理但空的数组以避免崩溃
                ref = self.replay_buffer[key]
                shape_tail = tuple(ref.shape[1:])
                return np.zeros((0,) + shape_tail, dtype=ref.dtype)
            return np.concatenate(parts, axis=0)

        # 动作归一化（基于未填充的有效段）
        action_np = gather_valid_numpy('action')
        stat = array_to_stats(action_np)
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

        # 观测值归一化（同样仅基于有效段）
        for key in self.lowdim_keys:
            obs_np = gather_valid_numpy(key)
            stat = array_to_stats(obs_np)

            if key.endswith('pos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('quat'):
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('qpos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                raise RuntimeError(f'不支持的观测类型: {key}')
            normalizer[key] = this_normalizer

        # 图像归一化：固定范围，不需要统计
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()

        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        print("被调用：get_all_actions")
        return torch.from_numpy(self.replay_buffer['action'])
    
    def __len__(self):
        """返回数据集长度"""
        # 支持返回多个episode
        if self.episodes_per_item > 1:
            print(f"episodes_per_item: {self.episodes_per_item}")
            return math.ceil(len(self.episode_ranges_current) / self.episodes_per_item)
        return len(self.episode_ranges_current)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取数据集项"""
        threadpool_limits(1)
        if self.episodes_per_item <= 1:
            print("单条导入中……")
            return self._get_single_episode_item(idx)
        else:
            print("多条导入中……")
            return self._get_multi_episode_item(idx)
    
    def _get_single_episode_item(self, idx: int) -> Dict[str, torch.Tensor]:
        """处理单条episode的数据项"""
        start_idx, end_idx = self.episode_ranges_current[idx]
        # 加载单个episode的数据"""
        data = {}
        for key in self.replay_buffer.keys():
            data[key] = self.replay_buffer[key][start_idx:end_idx]
        # 处理单个episode数据
        # 处理观测数据
        obs_dict = {}
        for key in self.rgb_keys:
            # 移动通道维度并归一化
            obs_dict[key] = np.moveaxis(data[key], -1, 1).astype(np.float32) / 255.
        for key in self.lowdim_keys:
            obs_dict[key] = data[key].astype(np.float32)
        
        # 处理动作数据
        action_np = data['action'].astype(np.float32)
        
        # 运行期不再做逐键时长检查，等长校验已在 __init__ 基于元数据完成
        
        # 返回该 episode 的有效长度（不返回逐步掩码）
        try:
            max_len = int(self.replay_buffer.meta['max_episode_len'][0])
            ep_idx = start_idx // max_len
            valid_len = int(self.replay_buffer.meta['original_episode_lengths'][ep_idx])
        except Exception:
            # 旧缓存兜底：使用当前序列长度
            valid_len = int(action_np.shape[0])

        return {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(action_np),
            'valid_lengths': torch.tensor([valid_len], dtype=torch.long)
        }

    def _get_multi_episode_item(self, idx: int) -> Dict[str, torch.Tensor]:
        """处理多条episode的数据项"""
        start_group = idx * self.episodes_per_item
        ranges = self.episode_ranges_current[start_group:start_group + self.episodes_per_item]
        
        samples = []
        for start_idx, end_idx in ranges:
            # 加载单个episode的数据
            data = {}
            for key in self.replay_buffer.keys():
                data[key] = self.replay_buffer[key][start_idx:end_idx]
            samples.append(data)

        # 处理多个episode数据
        processed_samples = []
        for data in samples:
            # 处理每个episode的观测数据
            obs_dict = {}
            for key in self.rgb_keys:
                obs_dict[key] = np.moveaxis(data[key], -1, 1).astype(np.float32) / 255.
            for key in self.lowdim_keys:
                obs_dict[key] = data[key].astype(np.float32)
            
            # 处理动作数据
            action = data['action'].astype(np.float32)
            processed_samples.append({'obs': obs_dict, 'action': action})
        
        # 运行期不再做逐键时长检查，等长校验已在 __init__ 完成
        
        # 转换为torch张量
        obs_torch = {}
        for key in self.rgb_keys + self.lowdim_keys:
            obs_torch[key] = torch.stack([
                torch.from_numpy(sample['obs'][key]) for sample in processed_samples
            ], dim=0)
        
        action_torch = torch.stack([
            torch.from_numpy(sample['action']) for sample in processed_samples
        ], dim=0)
        
        # 非填充路径下：返回每个 episode 的有效长度向量
        vals = []
        try:
            max_len = int(self.replay_buffer.meta['max_episode_len'][0])
            orig_lengths = self.replay_buffer.meta['original_episode_lengths'][:]
        except Exception:
            max_len = None
            orig_lengths = None
        for (start_idx, end_idx) in ranges:
            if orig_lengths is not None and max_len is not None:
                ep_idx = start_idx // max_len
                valid_len = int(orig_lengths[ep_idx])
            else:
                valid_len = int(end_idx - start_idx)
            vals.append(valid_len)
        valid_lengths = torch.tensor(vals, dtype=torch.long)
        return {
            'obs': obs_torch,
            'action': action_torch,
            'valid_lengths': valid_lengths
        }
    
    


# 其余辅助函数保持不变
def _convert_actions(raw_actions, abs_action, rotation_transformer):
    """将 robomimic 的动作转换为目标表示。

    支持输入形状：
    - [T, D]
    - [B, T, D] 或更多前导维度，最后一维为动作维度。
    返回与输入相同的前导维度，最后一维为转换后的动作维度（10 或 20）。
    """
    actions = raw_actions
    if not abs_action:
        return actions

    # 保留前导维度，展平到二维进行处理，再还原
    orig_shape = raw_actions.shape
    last_dim = orig_shape[-1]
    flat = raw_actions.reshape(-1, last_dim)

    is_dual_arm = False
    if last_dim == 14:
        # 双臂 [N, 14] -> [N, 2, 7]
        flat = flat.reshape(-1, 2, 7)
        is_dual_arm = True

    pos = flat[..., :3]
    rot = flat[..., 3:6]
    gripper = flat[..., 6:]
    # rotation_transformer.forward 支持广播，最后一维为 3
    rot = rotation_transformer.forward(rot)
    flat = np.concatenate([pos, rot, gripper], axis=-1).astype(np.float32)

    if is_dual_arm:
        # [N, 2, 10] -> [N, 20]
        flat = flat.reshape(-1, 20)
        out_last = 20
    else:
        # [N, 10]
        out_last = 10

    actions = flat.reshape(*orig_shape[:-1], out_last)
    return actions

def _convert_robomimic_to_replay(store, shape_meta, dataset_path, abs_action, rotation_transformer, 
        n_workers=None, max_inflight_tasks=None, n_demo=100):
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    # 解析形状元数据
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
        episode_lengths = []
        for i in range(n_demo):
            demo = demos[f'demo_{i}']
            episode_lengths.append(demo['actions'].shape[0])
        
        # 添加数据集长度统计输出
        min_length = min(episode_lengths)
        max_length = max(episode_lengths)
        avg_length = sum(episode_lengths) / len(episode_lengths)
        print(f"[数据集统计] 总演示数: {n_demo}")
        print(f"[数据集统计] 最小长度: {min_length} 步")
        print(f"[数据集统计] 最大长度: {max_length} 步")
        print(f"[数据集统计] 平均长度: {avg_length:.2f} 步")
        print(f"[数据集统计] 长度范围: {min_length} - {max_length} 步")

        max_episode_len = max(episode_lengths)
        print(f"[缓存预处理] 最大episode长度: {max_episode_len}")

        # 记录原始长度信息，便于后续统计或可选使用
        _ = meta_group.array('original_episode_lengths', np.asarray(episode_lengths, dtype=np.int64),
                             dtype=np.int64, compressor=None, overwrite=True)

        # 统一后的 episode_ends 变为 [max, 2*max, ...]
        episode_ends = [(i + 1) * max_episode_len for i in range(n_demo)]
        _ = meta_group.array('episode_ends', episode_ends, dtype=np.int64, compressor=None, overwrite=True)
        _ = meta_group.array('max_episode_len', np.asarray([max_episode_len], dtype=np.int64),
                             dtype=np.int64, compressor=None, overwrite=True)

        # 低维数据和动作数据填充
        for key in tqdm(lowdim_keys + ['action'], desc="Loading lowdim data (padded)"):
            data_key = 'obs/' + key
            if key == 'action':
                data_key = 'actions'
            shape = shape_meta['action']['shape'] if key == 'action' else shape_meta['obs'][key]['shape']
            arr = np.zeros((n_demo, max_episode_len) + tuple(shape), dtype=np.float32)
            for i in range(n_demo):
                demo = demos[f'demo_{i}']
                d = demo[data_key][:].astype(np.float32)
                T = d.shape[0]
                if key == 'action':
                    # 先对每个 episode 做动作转换，再填充
                    d = _convert_actions(d, abs_action=abs_action, rotation_transformer=rotation_transformer)
                arr[i, :T] = d
                if T < max_episode_len:
                    arr[i, T:] = d[-1]  # 用最后一帧填充

            if key == 'action':
                # 动作在每条序列内已转换，整体形状应与 shape_meta['action'] 对齐
                assert arr.shape == (n_demo, max_episode_len) + tuple(shape_meta['action']['shape'])
            else:
                assert arr.shape == (n_demo, max_episode_len) + tuple(shape_meta['obs'][key]['shape'])

            # 存储为 [n_demo*max_episode_len, ...]，便于后续按帧连续访问
            arr_flat = arr.reshape(n_demo * max_episode_len, *arr.shape[2:])
            _ = data_group.array(
                name=key,
                data=arr_flat,
                shape=arr_flat.shape,
                chunks=arr_flat.shape,  # 连续布局，一次性 chunk；低维数据通常较小
                compressor=None,
                dtype=arr_flat.dtype
            )

        # 图像数据填充
        def img_copy(img_arr, zarr_idx, img):
            try:
                img_arr[zarr_idx] = img
                _ = img_arr[zarr_idx]
                return True
            except Exception:
                return False

        for key in rgb_keys:
            shape = tuple(shape_meta['obs'][key]['shape'])
            c, h, w = shape
            this_compressor = Jpeg2k(level=50)
            arr = np.zeros((n_demo, max_episode_len, h, w, c), dtype=np.uint8)
            for i in range(n_demo):
                demo = demos[f'demo_{i}']
                d = demo['obs'][key][:]  # [T, h, w, c]
                T = d.shape[0]
                arr[i, :T] = d
                if T < max_episode_len:
                    arr[i, T:] = d[-1]  # 用最后一帧填充
            arr_flat = arr.reshape(n_demo * max_episode_len, h, w, c)
            img_arr = data_group.require_dataset(
                name=key,
                shape=arr_flat.shape,
                chunks=(1, h, w, c),  # 与 v8 保持逐帧块，避免超大块
                compressor=this_compressor,
                dtype=np.uint8
            )
            # 并行写入（限制 inflight 任务数量，控制内存）
            with tqdm(total=arr_flat.shape[0], desc=f"Writing image {key}", mininterval=1.0) as pbar:
                with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                    futures = set()
                    for zarr_idx in range(arr_flat.shape[0]):
                        img = arr_flat[zarr_idx]
                        futures.add(executor.submit(img_copy, img_arr, zarr_idx, img))
                        if len(futures) >= max_inflight_tasks:
                            completed, futures = concurrent.futures.wait(
                                futures, return_when=concurrent.futures.FIRST_COMPLETED
                            )
                            for f in completed:
                                if not f.result():
                                    raise RuntimeError('Failed to encode image!')
                            pbar.update(len(completed))
                    completed, futures = concurrent.futures.wait(futures)
                    for f in completed:
                        if not f.result():
                            raise RuntimeError('Failed to encode image!')
                    pbar.update(len(completed))

    # 验证所有数据 shape
    for key in lowdim_keys + ['action']:
        arr = data_group[key][:]
        assert arr.shape[0] == n_demo * max_episode_len, f"{key} shape mismatch: {arr.shape}"
    for key in rgb_keys:
        arr = data_group[key][:]
        assert arr.shape[0] == n_demo * max_episode_len, f"{key} shape mismatch: {arr.shape}"

    # 元数据一致性校验：episode_ends 必须等于累计的等长片段
    ep_ends = meta_group['episode_ends'][:].tolist()
    assert ep_ends == [(i + 1) * max_episode_len for i in range(n_demo)], "episode_ends metadata mismatch"

    replay_buffer = ReplayBuffer(root)
    return replay_buffer

def normalizer_from_stat(stat):
    """从统计信息创建归一化器"""
    max_abs = np.maximum(stat['max'].max(), np.abs(stat['min']).max())
    scale = np.full_like(stat['max'], fill_value=1/max_abs)
    offset = np.zeros_like(stat['max'])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )
