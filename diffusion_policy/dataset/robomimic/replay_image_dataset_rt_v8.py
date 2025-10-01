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
        """创建数据归一化器"""
        normalizer = LinearNormalizer()

        # 动作归一化
        stat = array_to_stats(self.replay_buffer['action'])
        if self.abs_action:
            if stat['mean'].shape[-1] > 10:
                # 双臂情况
                this_normalizer = robomimic_abs_action_only_dual_arm_normalizer_from_stat(stat)
            else:
                this_normalizer = robomimic_abs_action_only_normalizer_from_stat(stat)
            
            if self.use_legacy_normalizer:
                this_normalizer = normalizer_from_stat(stat)
        else:
            # 已归一化
            this_normalizer = get_identity_normalizer_from_stat(stat)
        normalizer['action'] = this_normalizer

        # 观测值归一化
        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])

            if key.endswith('pos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('quat'):
                # 四元数已在[-1,1]范围内
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('qpos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                raise RuntimeError(f'不支持的观测类型: {key}')
            normalizer[key] = this_normalizer

        # 图像归一化
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
            return self._get_single_episode_item(idx)
        else:
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
        
        # 如果需要填充
        if self.max_episode_len is not None:
            obs_dict, action_np = self._pad_data(obs_dict, action_np)
        
        return {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(action_np)
        }

    def _pad_data(self, obs_dict: Dict[str, np.ndarray], action_np: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """填充数据到最大长度"""
        # 填充观测数据
        for key in obs_dict:
            v = obs_dict[key]
            T = v.shape[0]
            if T < self.max_episode_len:
                last = v[-1:]
                pad_cnt = self.max_episode_len - T
                pad = np.repeat(last, pad_cnt, axis=0)
                obs_dict[key] = np.concatenate([v, pad], axis=0)
            elif T > self.max_episode_len:
                obs_dict[key] = v[:self.max_episode_len]
        
        # 填充动作数据
        Ta = action_np.shape[0]
        if Ta < self.max_episode_len:
            last = action_np[-1:]
            pad_cnt = self.max_episode_len - Ta
            pad = np.repeat(last, pad_cnt, axis=0)
            action_np = np.concatenate([action_np, pad], axis=0)
        elif Ta > self.max_episode_len:
            action_np = action_np[:self.max_episode_len]
        
        return obs_dict, action_np



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
        
        # 如果需要填充
        if self.max_episode_len is not None:
            return self._pad_multi_episodes(processed_samples)
        
        # 转换为torch张量
        obs_torch = {}
        for key in self.rgb_keys + self.lowdim_keys:
            obs_torch[key] = torch.stack([
                torch.from_numpy(sample['obs'][key]) for sample in processed_samples
            ], dim=0)
        
        action_torch = torch.stack([
            torch.from_numpy(sample['action']) for sample in processed_samples
        ], dim=0)
        
        return {
            'obs': obs_torch,
            'action': action_torch
        }
    
    def _pad_multi_episodes(self, samples: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
        """填充多个episode数据到相同长度"""
        # 填充观测数据
        obs_batch = {k: [] for k in self.rgb_keys + self.lowdim_keys}
        for sample in samples:
            for k in obs_batch:
                v = sample['obs'][k]
                T = v.shape[0]
                if T >= self.max_episode_len:
                    obs_batch[k].append(v[:self.max_episode_len])
                else:
                    last = v[-1:]
                    pad_cnt = self.max_episode_len - T
                    pad = np.repeat(last, pad_cnt, axis=0)
                    obs_batch[k].append(np.concatenate([v, pad], axis=0))
        
        # 填充动作数据
        padded_actions = []
        for sample in samples:
            a = sample['action']
            T = a.shape[0]
            if T >= self.max_episode_len:
                padded_actions.append(a[:self.max_episode_len])
            else:
                last = a[-1:]
                pad_cnt = self.max_episode_len - T
                pad = np.repeat(last, pad_cnt, axis=0)
                padded_actions.append(np.concatenate([a, pad], axis=0))
        
        # 转换为torch张量
        obs_torch = {k: torch.stack([torch.from_numpy(x) for x in obs_batch[k]], dim=0) for k in obs_batch}
        action_torch = torch.stack([torch.from_numpy(x) for x in padded_actions], dim=0)
        
        return {
            'obs': obs_torch,
            'action': action_torch
        }


# 其余辅助函数保持不变
def _convert_actions(raw_actions, abs_action, rotation_transformer):
    actions = raw_actions
    if abs_action:
        is_dual_arm = False
        if raw_actions.shape[-1] == 14:
            # 双臂
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
        # 计算总步数
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

        # 保存低维数据
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
        
        def img_copy(img_arr, zarr_idx, hdf5_arr, hdf5_idx):
            try:
                img_arr[zarr_idx] = hdf5_arr[hdf5_idx]
                # 确保可以成功解码decode
                _ = img_arr[zarr_idx]
                return True
            except Exception as e:
                return False
        
        with tqdm(total=n_steps*len(rgb_keys), desc="Loading image data", mininterval=1.0) as pbar:
            # 每个线程一个块，不需要同步
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
    """从统计信息创建归一化器"""
    max_abs = np.maximum(stat['max'].max(), np.abs(stat['min']).max())
    scale = np.full_like(stat['max'], fill_value=1/max_abs)
    offset = np.zeros_like(stat['max'])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )
