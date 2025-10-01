"""
# 初始化数据集，默认返回整条episode
dataset = RobomimicReplayImageDatasetRTV1(
    shape_meta=shape_meta,
    dataset_path=dataset_path,
    n_obs_steps=2,
    use_sequence_sampler=False  # 默认值，可以省略
)

# 切换到返回多个片段的模式
dataset.set_mode(use_sequence_sampler=True)

# 切换回返回整条episode的模式
dataset.set_mode(use_sequence_sampler=False)
"""

from typing import Dict, List
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
            episodes_per_item=1,
            use_sequence_sampler=False  # 初始模式设置，False表示返回整条episode
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
                    # cache does not exists
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
        
        # 存储所有必要的组件，以便动态切换模式
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
        # 每个 dataset 索引返回的 episode 数量（仅在 episode 模式下生效）
        self.episodes_per_item = int(episodes_per_item)
        
        # 为两种模式准备所有必要的组件
        # 1. 序列采样器模式（返回多个片段）
        key_first_k = dict()
        if n_obs_steps is not None:
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        self.sequence_sampler = SequenceSampler(
            replay_buffer=replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k)
        
        # 2. Episode范围模式（返回整条episode）
        episode_ends = replay_buffer.episode_ends[:]
        episode_starts = [0] + episode_ends[:-1].tolist()
        
        self.episode_ranges = []
        for i in range(len(episode_starts)):
            if train_mask[i]:
                self.episode_ranges.append((episode_starts[i], episode_ends[i]))
        
        # 设置当前模式
        self.set_mode(use_sequence_sampler)

    def set_mode(self, use_sequence_sampler):
        """动态切换数据集模式
        
        Args:
            use_sequence_sampler: 
                True - 使用序列采样器模式，返回多个片段
                False - 使用episode范围模式，返回整条episode
        """
        self.use_sequence_sampler = use_sequence_sampler
        if use_sequence_sampler:
            self.sampler = self.sequence_sampler
            self.episode_ranges_current = None
        else:
            self.sampler = None
            self.episode_ranges_current = self.episode_ranges

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        
        # 根据当前模式设置验证数据集
        if self.use_sequence_sampler:
            # 使用序列采样器模式
            val_set.sequence_sampler = SequenceSampler(
                replay_buffer=self.replay_buffer, 
                sequence_length=self.horizon,
                pad_before=self.pad_before, 
                pad_after=self.pad_after,
                episode_mask=~self.train_mask
            )
            val_set.set_mode(True)  # 确保使用序列采样器模式
        else:
            # 使用episode范围模式
            episode_ends = self.replay_buffer.episode_ends[:]
            episode_starts = [0] + episode_ends[:-1].tolist()
            
            val_set.episode_ranges = []
            for i in range(len(episode_starts)):
                if not self.train_mask[i]:
                    val_set.episode_ranges.append((episode_starts[i], episode_ends[i]))
            val_set.set_mode(False)  # 确保使用episode范围模式
        
        val_set.train_mask = ~self.train_mask
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
        if self.use_sequence_sampler:
            return len(self.sampler)
        else:
            if getattr(self, 'episodes_per_item', 1) > 1:
                return math.ceil(len(self.episode_ranges_current) / self.episodes_per_item)
            return len(self.episode_ranges_current)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        
        if self.use_sequence_sampler:
            # 使用序列采样器模式 - 返回多个片段
            data = self.sampler.sample_sequence(idx)

            # 只返回前n_obs_steps个观测步骤
            T_slice = slice(self.n_obs_steps) if self.n_obs_steps is not None else slice(None)

            obs_dict = dict()
            for key in self.rgb_keys:
                # 移动通道维度并归一化
                obs_dict[key] = np.moveaxis(data[key][T_slice], -1, 1).astype(np.float32) / 255.
                del data[key]
            for key in self.lowdim_keys:
                obs_dict[key] = data[key][T_slice].astype(np.float32)
                del data[key]

            torch_data = {
                'obs': dict_apply(obs_dict, torch.from_numpy),
                'action': torch.from_numpy(data['action'].astype(np.float32))
            }
        else:
            # 使用episode范围模式 - 返回整条episode 或 多个 episode（由 episodes_per_item 控制）
            epi_per = getattr(self, 'episodes_per_item', 1)
            if epi_per <= 1:
                start_idx, end_idx = self.episode_ranges_current[idx]

                # 直接读取整个episode的数据
                data = {}
                for key in self.replay_buffer.keys():
                    data[key] = self.replay_buffer[key][start_idx:end_idx]

                # 处理观测数据
                T_slice = slice(self.n_obs_steps) if self.n_obs_steps is not None else slice(None)

                obs_dict = dict()
                for key in self.rgb_keys:
                    # 移动通道维度并归一化
                    obs_dict[key] = np.moveaxis(data[key][T_slice], -1, 1).astype(np.float32) / 255.
                for key in self.lowdim_keys:
                    obs_dict[key] = data[key][T_slice].astype(np.float32)

                torch_data = {
                    'obs': dict_apply(obs_dict, torch.from_numpy),
                    'action': torch.from_numpy(data['action'].astype(np.float32))
                }
            else:
                # 返回多个 episode，按 idx 进行分组
                start_group = idx * epi_per
                ranges = self.episode_ranges_current[start_group:start_group + epi_per]
                samples = []
                obs_lengths = []
                act_lengths = []
                for (s_idx, e_idx) in ranges:
                    data = {}
                    for key in self.replay_buffer.keys():
                        data[key] = self.replay_buffer[key][s_idx:e_idx]
                    T_slice = slice(self.n_obs_steps) if self.n_obs_steps is not None else slice(None)
                    obs_dict = {}
                    for key in self.rgb_keys:
                        obs_dict[key] = np.moveaxis(data[key][T_slice], -1, 1).astype(np.float32) / 255.
                    for key in self.lowdim_keys:
                        obs_dict[key] = data[key][T_slice].astype(np.float32)
                    action = data['action'].astype(np.float32)
                    obs_lengths.append(int(next(iter(obs_dict.values())).shape[0]))
                    act_lengths.append(int(action.shape[0]))
                    samples.append({'obs': obs_dict, 'action': action})

                # pad samples to same time length by repeating last frame/action
                max_obs = int(max(obs_lengths)) if len(obs_lengths) > 0 else 0
                max_act = int(max(act_lengths)) if len(act_lengths) > 0 else 0

                # pad obs modalities
                obs_batch = {k: [] for k in self.rgb_keys + self.lowdim_keys}
                for sample in samples:
                    for k in obs_batch.keys():
                        v = sample['obs'][k]
                        T = v.shape[0]
                        if T >= max_obs:
                            obs_batch[k].append(v[:max_obs])
                        else:
                            last = v[-1:]
                            pad_cnt = max_obs - T
                            pad = np.repeat(last, pad_cnt, axis=0)
                            obs_batch[k].append(np.concatenate([v, pad], axis=0))

                # pad actions
                padded_actions = []
                for sample in samples:
                    a = sample['action']
                    T = a.shape[0]
                    if T >= max_act:
                        padded_actions.append(a[:max_act])
                    else:
                        last = a[-1:]
                        pad_cnt = max_act - T
                        pad = np.repeat(last, pad_cnt, axis=0)
                        padded_actions.append(np.concatenate([a, pad], axis=0))

                # convert to torch and stack
                obs_torch = {k: torch.stack([torch.from_numpy(x) for x in obs_batch[k]], dim=0) for k in obs_batch.keys()}
                action_torch = torch.stack([torch.from_numpy(x) for x in padded_actions], dim=0)

                # print lengths for confirmation
                print(f"[Dataset.__getitem__ grouped] obs_lengths={obs_lengths}, action_lengths={act_lengths}")

                torch_data = {
                    'obs': obs_torch,
                    'action': action_torch
                }
        
        return torch_data

    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate function that pads a list of episode samples to the same time length by
        repeating the last timestep in obs modalities and actions. It prints per-episode
        lengths for inspection and returns a batch dict compatible with the training loop.

        Returned dict contains:
            'obs': {k: Tensor[B, T_max, ...]}
            'action': Tensor[B, T_max_action, D]
        """
        if len(batch) == 0:
            return batch

        # compute lengths per sample
        obs_lengths = []
        action_lengths = []
        for sample in batch:
            # obs time length: take any modality (they should be same length per sample)
            first_obs = next(iter(sample['obs'].values()))
            obs_len = first_obs.shape[0]
            obs_lengths.append(int(obs_len))
            action_lengths.append(int(sample['action'].shape[0]))

        # print lengths for debugging/confirmation
        print(f"[Dataset.collate_fn] obs_lengths={obs_lengths}, action_lengths={action_lengths}")

        max_obs_len = int(max(obs_lengths))
        max_act_len = int(max(action_lengths))

        # pad obs modalities
        obs_keys = list(batch[0]['obs'].keys())
        obs_batch = {}
        for k in obs_keys:
            padded = []
            for sample in batch:
                v = sample['obs'][k]
                # handle potential leading batch dim [1, T, ...]
                if v.ndim >= 2 and v.shape[0] == 1:
                    v = v.squeeze(0)
                T = v.shape[0]
                if T >= max_obs_len:
                    padded.append(v[:max_obs_len])
                else:
                    last = v[-1:]
                    pad_cnt = max_obs_len - T
                    pad = last.repeat(pad_cnt, *([1] * (last.ndim - 1)))
                    padded.append(torch.cat([v, pad], dim=0))
            obs_batch[k] = torch.stack(padded, dim=0)

        # pad actions
        padded_actions = []
        for sample in batch:
            a = sample['action']
            # squeeze leading batch dim if present
            if a.ndim == 3 and a.shape[0] == 1:
                a = a.squeeze(0)
            T = a.shape[0]
            if T >= max_act_len:
                padded_actions.append(a[:max_act_len])
            else:
                last = a[-1:]
                pad_cnt = max_act_len - T
                pad = last.repeat(pad_cnt, *([1] * (last.ndim - 1)))
                padded_actions.append(torch.cat([a, pad], dim=0))
        action_batch = torch.stack(padded_actions, dim=0)

        return {
            'obs': obs_batch,
            'action': action_batch
        }

# 其余辅助函数保持不变
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
            print(f"Episode {i} length: {episode_length}")  # 添加显示
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
