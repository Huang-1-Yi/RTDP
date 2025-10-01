from typing import Dict, List, Optional, Tuple
import os
import shutil
import copy
import json
import numpy as np
import h5py
import zarr
from tqdm import tqdm
import multiprocessing
from filelock import FileLock
from threadpoolctl import threadpool_limits

import torch
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
import hashlib
import time

register_codecs()


class RobomimicReplayImageDatasetRTV8(BaseImageDataset):
    """
    RTV8: 四步法实现（转换期物理化 + 每集映射 + 每 epoch 全局映射表文件化 + 运行期按表取数）

    关键点：
    - 在数据转换函数中，针对每个 episode 执行“首尾各复制 horizon-1 帧”的物理化填充；
      将填充后的序列直接写入 zarr（obs/action 与 rgb/low_dim），并并行生成 pos_info（与 obs 并列）
      和 episode_map（每集统计 real_len, real_len_new, buffer_pos_id_start/end, window_num 等）。
    - window_num_1 = real_len + horizon - 1；window_num_2 = real_len_new - horizon + 1；二者应恒等。
    - 每个 epoch 生成一个“全局二维映射总表”并保存到文件：形状为 [batch_size, max_cols, 2]，
      单元存 (episode_id, episode_get_id)。按列组织 batch：p=id%bs, q=id//bs。
    - 运行期按 epoch 读取映射表文件，__getitem__ 根据 (e, m) 计算绝对下标，直接切片 zarr 数组，返回
      obs(n_obs_steps)、action(horizon) 与 pos_info(horizon, 5)。
    """

    def __init__(
        self,
        shape_meta: dict,
        dataset_path: str,
        horizon: int = 1,
        n_obs_steps: Optional[int] = None,
        abs_action: bool = False,
        rotation_rep: str = 'rotation_6d',  # ignored when abs_action=False
        use_legacy_normalizer: bool = False,
        use_cache: bool = False,
        seed: int = 42,
        val_ratio: float = 0.0,
        n_demo: int = 100,
        batch_size: int = 1,
        mapping_dir: Optional[str] = None,
        build_epoch_mappings: int = 0,
        epoch_index: int = 0,
        num_epochs: Optional[int] = None,
    ):
        start_time = time.time()
        print(f"数据化初始化开始于: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.n_demo = n_demo
        self.horizon = int(horizon)
        self.n_obs_steps = int(n_obs_steps) if n_obs_steps is not None else 1
        self.abs_action = abs_action
        self.use_legacy_normalizer = use_legacy_normalizer
        self.batch_size = int(batch_size) if batch_size is not None else 1
        if self.batch_size <= 0:
            self.batch_size = 1

        rotation_transformer = RotationTransformer(from_rep='axis_angle', to_rep=rotation_rep)
        # 记录训练总 epoch 数（可用于预构建映射表）
        self.num_epochs = int(num_epochs) if num_epochs is not None else None

        # 1+2 步：转换阶段完成“物理化填充 + pos_info + episode_map”
        replay_buffer = None
        if use_cache:
            # 注：沿用原有缓存文件命名，以 n_demo 作为区分；若需严格隔离不同 horizon，可在此加入 h 标记。
            cache_zarr_path = dataset_path + f'.{n_demo}.' + '.zarr.zip'
            cache_lock_path = cache_zarr_path + '.rtv8.lock'
            print('Acquiring lock on cache (RTV8).')
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    try:
                        print('Cache does not exist. Creating (RTV8)!')
                        replay_buffer = _convert_robomimic_to_replay_v8(
                            store=zarr.MemoryStore(),
                            shape_meta=shape_meta,
                            dataset_path=dataset_path,
                            abs_action=abs_action,
                            rotation_transformer=rotation_transformer,
                            n_demo=n_demo,
                            horizon=self.horizon,
                        )
                        print('Saving RTV8 cache to disk.')
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(store=zip_store)
                    except Exception as e:
                        if os.path.exists(cache_zarr_path):
                            shutil.rmtree(cache_zarr_path)
                        raise e
                else:
                    print('Loading cached RTV8 ReplayBuffer from Disk.')
                    with zarr.ZipStore(cache_zarr_path, mode='r') as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore())
                    print('Loaded!')
        else:
            replay_buffer = _convert_robomimic_to_replay_v8(
                store=zarr.MemoryStore(),
                shape_meta=shape_meta,
                dataset_path=dataset_path,
                abs_action=abs_action,
                rotation_transformer=rotation_transformer,
                n_demo=n_demo,
                horizon=self.horizon,
            )

        # 解析 obs 键（rgb / low_dim）
        rgb_keys: List[str] = []
        lowdim_keys: List[str] = []
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            t = attr.get('type', 'low_dim')
            if t == 'rgb':
                rgb_keys.append(key)
            elif t == 'low_dim':
                lowdim_keys.append(key)

        # 划分训练/验证（按 episode 级）
        val_mask = get_val_mask(n_episodes=replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask

        # 保存属性
        self.replay_buffer = replay_buffer
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.train_mask = train_mask
        self.seed = seed

        # episode 映射表（存放于 meta/episode_map）
        # episode_map: [n_episodes, 6] = [episode_id, real_len, real_len_new, buf_start(1-based), buf_end(1-based), window_num]
        self.episode_map = np.asarray(self.replay_buffer.meta['episode_map'][:], dtype=np.int64)
        assert self.episode_map.shape[0] == self.replay_buffer.n_episodes

        # 提取训练集 episodes
        self.train_episode_ids: np.ndarray = np.nonzero(self.train_mask)[0].astype(np.int64)
        if self.train_episode_ids.size == 0:
            raise RuntimeError('No training episodes available!')

        # 校验 window_num_1 == window_num_2（转换时已等价，这里再次稳妥检查）
        # window_num 已存为列 5
        # 若需要详细核对，可按 real_len_new - h + 1 的公式重新验证。
        self._validate_window_num(self.train_episode_ids)

        # 全局映射目录
        if mapping_dir is None:
            mapping_dir = os.path.splitext(dataset_path)[0] + f'.rtdp_v8_maps.b{self.batch_size}'
        os.makedirs(mapping_dir, exist_ok=True)
        self.mapping_dir = mapping_dir

        # 若指定预先构建多个 epoch 的全局映射表，则批量生成文件；
        # 否则若提供了 num_epochs，则默认构建 num_epochs 份映射表。
        if build_epoch_mappings and build_epoch_mappings > 0:
            for epk in range(build_epoch_mappings):
                path = self._mapping_file_path(epk)
                if not os.path.exists(path):  # 添加映射表存在性检查​，避免重复构建已存在的映射表
                    self._build_and_save_epoch_mapping(epoch_index=epk, episode_ids=self.train_episode_ids)
                else:
                    print(f"Mapping for epoch {epk} already exists. Skipping.")
        elif self.num_epochs is not None and self.num_epochs > 0:
            for epk in range(self.num_epochs):
                path = self._mapping_file_path(epk)
                if not os.path.exists(path):
                    self._build_and_save_epoch_mapping(epoch_index=epk, episode_ids=self.train_episode_ids)
                else:
                    print(f"Mapping for epoch {epk} already exists. Skipping.")

        # 加载当前 epoch 的映射表（若不存在则生成后加载）
        self._epoch_index = int(epoch_index)
        self._load_or_build_epoch_mapping(self._epoch_index, self.train_episode_ids)
        self._write_current_epoch_file(self._epoch_index)
        print(f"数据化初始化结束在: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # ----------------- 公共 API -----------------
    def set_epoch(self, epoch_index: int):
        """切换到指定 epoch 的映射总表（文件不存在则自动生成）。"""
        self._epoch_index = int(epoch_index)
        self._load_or_build_epoch_mapping(self._epoch_index, self.train_episode_ids)
        self._write_current_epoch_file(self._epoch_index)

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.train_mask = ~self.train_mask
        val_set.train_episode_ids = np.nonzero(val_set.train_mask)[0].astype(np.int64)
        if val_set.train_episode_ids.size == 0:
            return val_set

        val_set._validate_window_num(val_set.train_episode_ids)

        # 使用独立的映射目录
        val_set.mapping_dir = self.mapping_dir + '.val'
        os.makedirs(val_set.mapping_dir, exist_ok=True)

        # 默认加载或构建第 0 个 epoch 的映射表
        val_set._load_or_build_epoch_mapping(0, val_set.train_episode_ids)
        return val_set

    # ----------------- Normalizer -----------------
    def get_normalizer(self, **kwargs) -> LinearNormalizer:
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

        # obs lowdim
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

        # obs image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    # ----------------- Dataloader 接口 -----------------
    def __len__(self):
        # 当前 epoch 对应的全局映射表长度
        return int(self.batch_size * self._max_cols)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        # 检查当前 epoch 是否发生变化（for persistent_workers=True 的多进程 DataLoader）
        new_epoch = self._read_current_epoch_file()
        if (new_epoch is not None) and (new_epoch != getattr(self, '_epoch_index', None)):
            self._epoch_index = new_epoch
            # 重新加载当前 epoch 的映射表
            self._load_or_build_epoch_mapping(self._epoch_index, self.train_episode_ids)
        B = self.batch_size
        p = idx % B
        q = idx // B
        if q >= self._max_cols:
            q = self._max_cols - 1

        e = int(self._global_pairs[p, q, 0])              # episode_id
        m = int(self._global_pairs[p, q, 1])              # episode_get_id (窗口起点)

        # 通过 episode_map 将 (e, m) 映射到“填充后全局数组”的绝对下标
        # episode_map 行: [episode_id, real_len, real_len_new, buf_start(1-based), buf_end(1-based), window_num]
        info = self.episode_map[e]
        buf_start_1b = int(info[3])
        # 绝对 0-based 起点
        abs_start = (buf_start_1b - 1) + m

        S = self.n_obs_steps
        h = self.horizon

        # 采样 obs/action/pos_info
        obs_dict = {}
        for key in self.rgb_keys:
            seq = self.replay_buffer[key][abs_start: abs_start + S]  # (S, H, W, C)
            seq = np.moveaxis(seq, -1, 1).astype(np.float32) / 255.0  # -> (S, C, H, W)
            obs_dict[key] = seq
        for key in self.lowdim_keys:
            seq = self.replay_buffer[key][abs_start: abs_start + S].astype(np.float32)
            obs_dict[key] = seq

        actions = self.replay_buffer['action'][abs_start: abs_start + h].astype(np.float32)
        pos_info = self.replay_buffer['pos_info'][abs_start: abs_start + h].astype(np.int64)

        out = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(actions),
            'window_info': torch.from_numpy(pos_info),  # [h, 5]
        }
        return out

    # ----------------- 内部：epoch 映射表 -----------------
    def _mapping_file_path(self, epoch_index: int) -> str:
        return os.path.join(self.mapping_dir, f'mapping_epoch_{epoch_index:05d}.npz')

    def _current_epoch_file_path(self) -> str:
        return os.path.join(self.mapping_dir, 'current_epoch.txt')

    def _load_or_build_epoch_mapping(self, epoch_index: int, episode_ids: np.ndarray):
        path = self._mapping_file_path(epoch_index)
        if not os.path.exists(path):
            print(f"Mapping for epoch {epoch_index} not found. Building...")
            self._build_and_save_epoch_mapping(epoch_index, episode_ids)
        
        data = np.load(path)
        current_fingerprint = hashlib.md5(
            str(self.train_episode_ids.tolist()).encode() + 
            str(self.episode_map.tobytes()).encode()
        ).hexdigest()
        
        saved_fingerprint = data.get('dataset_fingerprint', None)
        if saved_fingerprint is None or saved_fingerprint != current_fingerprint:
            print(f"Dataset changed. Rebuilding mapping for epoch {epoch_index}...")
            self._build_and_save_epoch_mapping(epoch_index, episode_ids)
            data = np.load(path)

        data = np.load(path)
        pairs = data['pairs']  # [B, max_cols, 2]
        B_file = int(data['batch_size'])
        max_cols = int(data['max_cols'])
        
        # 验证映射表是否匹配当前配置
        if B_file != self.batch_size:
            raise RuntimeError(f'Mapping file batch_size={B_file} mismatches dataset batch_size={self.batch_size}')
        
        # 验证映射表是否包含当前训练集的所有episode
        unique_ep_ids = np.unique(pairs[..., 0])
        if not np.all(np.isin(unique_ep_ids, episode_ids)):
            print(f"Warning: Mapping for epoch {epoch_index} contains episodes not in current training set. Rebuilding...")
            self._build_and_save_epoch_mapping(epoch_index, episode_ids)
            data = np.load(path)
            pairs = data['pairs']
            max_cols = int(data['max_cols'])
        
        self._global_pairs = pairs.astype(np.int32)
        self._max_cols = max_cols

    def _build_and_save_epoch_mapping(self, epoch_index: int, episode_ids: np.ndarray):
        start_time = time.time()
        print(f"Epoch mapping 创建开始于: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        # 计算数据集指纹
        dataset_fingerprint = hashlib.md5(
            str(self.train_episode_ids.tolist()).encode() + 
            str(self.episode_map.tobytes()).encode()
        ).hexdigest()
        
        rng = np.random.RandomState(self.seed + epoch_index)
        ep_ids = episode_ids.copy()
        rng.shuffle(ep_ids)

        B = self.batch_size
        rows: List[List[Tuple[int, int]]] = [[] for _ in range(B)]
        row_lens = np.zeros(B, dtype=np.int64)

        # 将每个 episode 的窗口 [0..window_num) 依次追加到“当前最短行”
        for e in ep_ids:
            win = int(self.episode_map[e, 5])  # window_num
            r = int(np.argmin(row_lens))
            rows[r].extend([(int(e), m) for m in range(win)])
            row_lens[r] += win

        # 对齐列长
        max_cols = int(row_lens.max()) if B > 0 else 0
        if max_cols == 0:
            raise RuntimeError('No columns in epoch mapping!')

        # 用“从 0 开始”的顺序重复若干 episode 来补齐短行
        def fill_row_to(r: int, target: int):
            deficit = target - len(rows[r])
            if deficit <= 0:
                return
            # 一次挑选若干 episode 顺序填入其 [0..window_num) 直到补满
            fill_ptr = 0
            while len(rows[r]) < target:
                # 任取一个 episode（循环使用 ep_ids 列表）
                e = int(ep_ids[fill_ptr % len(ep_ids)])
                win = int(self.episode_map[e, 5])
                # 依次追加 0..win-1，但不能超过 target
                need = target - len(rows[r])
                take = min(win, need)
                rows[r].extend([(e, m) for m in range(take)])
                fill_ptr += 1

        for r in range(B):
            fill_row_to(r, max_cols)

        pairs = np.zeros((B, max_cols, 2), dtype=np.int32)
        for r in range(B):
            for c, (e, m) in enumerate(rows[r]):
                pairs[r, c, 0] = e
                pairs[r, c, 1] = m

        path = self._mapping_file_path(epoch_index)
        # 保存时包含指纹
        np.savez_compressed(
            path, 
            pairs=pairs, 
            batch_size=self.batch_size, 
            max_cols=max_cols,
            dataset_fingerprint=dataset_fingerprint
        )
        print(f"Epoch mapping 创建结束在 {time.time() - start_time:.2f} seconds")
    
    def _write_current_epoch_file(self, epoch_index: int):
        try:
            with open(self._current_epoch_file_path(), 'w') as f:
                f.write(str(int(epoch_index)))
        except Exception:
            pass

    def _read_current_epoch_file(self) -> Optional[int]:
        try:
            with open(self._current_epoch_file_path(), 'r') as f:
                txt = f.read().strip()
            if txt:
                return int(txt)
        except Exception:
            return None
        return None

    def _validate_window_num(self, episode_ids: np.ndarray):
        # 验证 window_num_1 == window_num_2
        # window_num_1 = real_len + h - 1；window_num_2 = real_len_new - h + 1
        h = self.horizon
        bad: List[int] = []
        for e in episode_ids:
            real_len = int(self.episode_map[e, 1])
            real_len_new = int(self.episode_map[e, 2])
            win1 = real_len + h - 1
            win2 = real_len_new - h + 1
            if win1 != win2 or win1 != int(self.episode_map[e, 5]):
                bad.append(int(e))
        if len(bad) > 0:
            print('Window num mismatch for episode ids:', bad)
            raise RuntimeError('window_num_1 != window_num_2 for some episodes!')


# ----------------- 转换：物理化填充 + pos_info + episode_map -----------------
def _convert_robomimic_to_replay_v8(
    store,
    shape_meta,
    dataset_path,
    abs_action,
    rotation_transformer,
    n_workers=None,
    max_inflight_tasks=None,
    n_demo=100,
    horizon=1,
):
    start_time = time.time()
    print(f"数据化转换开始于：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    h = int(horizon)
    pad_b = max(h - 1, 0)
    pad_a = max(h - 1, 0)

    # 解析 obs 键
    rgb_keys: List[str] = []
    lowdim_keys: List[str] = []
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        t = attr.get('type', 'low_dim')
        if t == 'rgb':
            rgb_keys.append(key)
        elif t == 'low_dim':
            lowdim_keys.append(key)

    root = zarr.group(store)
    data_group = root.require_group('data', overwrite=True)
    meta_group = root.require_group('meta', overwrite=True)

    # 读取一次，统计每集长度与填充后长度
    with h5py.File(dataset_path, 'r') as file:
        demos = file['data']
        real_lens = []
        for i in range(n_demo):
            demo = demos[f'demo_{i}']
            real_lens.append(int(demo['actions'].shape[0]))
        real_lens = np.asarray(real_lens, dtype=np.int64)
        real_len_news = real_lens + 2 * (h - 1)
        total_padded = int(real_len_news.sum())

        # 创建 episode_map 数组（注意：应存放在 meta 组，避免违反 ReplayBuffer 对 data 组长度一致性的断言）
        episode_map = np.zeros((n_demo, 6), dtype=np.int64)
        episode_map_arr = meta_group.array(
            name='episode_map',
            data=np.zeros((n_demo, 6), dtype=np.int64),
            shape=(n_demo, 6),
            chunks=(n_demo, 6), # 一次性写入
            dtype=np.int64,
            compressor=None,
            overwrite=True
        )

        # 确保填充后的总长度与累计和一致
        assert total_padded == real_len_news.sum(), "Total padded length mismatch"
        episode_ends_new = np.cumsum(real_len_news)
        # 确保最后一个累计和等于总长度
        assert episode_ends_new[-1] == total_padded, "Episode ends mismatch with total padded length"
        _ = meta_group.array('episode_ends', episode_ends_new, dtype=np.int64, compressor=None, overwrite=True)

        # 分配 action / lowdim 数组
        # 注：chunks=full shape 仅为简单起见，action/lowdim 为整体写入；图像用 (1,H,W,C) chunk。
        # 动作
        action_dim = int(np.prod(shape_meta['action']['shape']))
        action_arr = data_group.array(
            name='action',
            data=np.empty((total_padded, action_dim), dtype=np.float32),
            shape=(total_padded, action_dim),
            chunks=(min(total_padded, max(1, 1024)), action_dim),
            compressor=None,
            dtype=np.float32,
            overwrite=True,
        )

        # 低维
        lowdim_arrs: Dict[str, zarr.Array] = {}
        for key in lowdim_keys:
            this_shape = (total_padded,) + tuple(shape_meta['obs'][key]['shape'])
            lowdim_arrs[key] = data_group.array(
                name=key,
                data=np.empty(this_shape, dtype=np.float32),
                shape=this_shape,
                chunks=(min(total_padded, max(1, 1024)),) + tuple(shape_meta['obs'][key]['shape']),
                compressor=None,
                dtype=np.float32,
                overwrite=True,
            )

        # 图像
        img_arrs: Dict[str, zarr.Array] = {}
        for key in rgb_keys:
            c, h_img, w_img = tuple(shape_meta['obs'][key]['shape'])
            this_compressor = Jpeg2k(level=50)
            img_arrs[key] = data_group.require_dataset(
                name=key,
                shape=(total_padded, h_img, w_img, c),
                chunks=(1, h_img, w_img, c),
                compressor=this_compressor,
                dtype=np.uint8,
            )

        # pos_info: [total_padded, 5]
        pos_info_arr = data_group.array(
            name='pos_info',
            data=np.empty((total_padded, 5), dtype=np.int64),
            shape=(total_padded, 5),
            chunks=(min(total_padded, max(1, 1024)), 5),
            compressor=None,
            dtype=np.int64,
            overwrite=True,
        )

        # episode_map: [n_demo, 6] = [episode_id, real_len, real_len_new, buf_start(1b), buf_end(1b), window_num]
        episode_map = np.zeros((n_demo, 6), dtype=np.int64)
        write_ptr = 0  # 全局 0-based 写指针

        # 遍历每集，生成填充写入
        for e in tqdm(range(n_demo), desc='RTV8: materialize padded episodes'):
            demo = demos[f'demo_{e}']
            real_len = real_lens[e]
            real_len_new = real_len_news[e]
            win_num = real_len + h - 1

            # 计算该集写入区间 [ofs, ofs+real_len_new)
            ofs = write_ptr
            ofs_end = ofs + real_len_new

            # 读取本集原始数据
            act_raw = demo['actions'][:].astype(np.float32)
            act_raw = _convert_actions_v8(act_raw, abs_action=abs_action, rotation_transformer=rotation_transformer)

            # 填充后的 action 写入
            # 前 pad_b 段
            if pad_b > 0:
                action_arr[ofs: ofs + pad_b] = act_raw[0:1].repeat(pad_b, axis=0)
            # 中间原始段
            action_arr[ofs + pad_b: ofs + pad_b + real_len] = act_raw
            # 尾 pad_a 段
            if pad_a > 0:
                action_arr[ofs + pad_b + real_len: ofs_end] = act_raw[-1: ].repeat(pad_a, axis=0)

            # 低维 obs 填充
            for key in lowdim_keys:
                arr = demo['obs'][key][:].astype(np.float32)
                if pad_b > 0:
                    lowdim_arrs[key][ofs: ofs + pad_b] = arr[0:1].repeat(pad_b, axis=0)
                lowdim_arrs[key][ofs + pad_b: ofs + pad_b + real_len] = arr
                if pad_a > 0:
                    lowdim_arrs[key][ofs + pad_b + real_len: ofs_end] = arr[-1: ].repeat(pad_a, axis=0)

            # 图像 obs 填充（逐帧复制）
            for key in rgb_keys:
                arr = demo['obs'][key]
                # 前 pad_b
                for t in range(pad_b):
                    img_arrs[key][ofs + t] = arr[0]
                # 中段原始
                for t in range(real_len):
                    img_arrs[key][ofs + pad_b + t] = arr[t]
                # 尾 pad_a
                for t in range(pad_a):
                    img_arrs[key][ofs + pad_b + real_len + t] = arr[-1]

            # pos_info 构建
            # 列说明：0=is_warmup, 1=episode_index, 2=episode_new_pos_id, 3=episode_id, 4=buffer_pos_id(1-based)
            pos = np.zeros((real_len_new, 5), dtype=np.int64)
            # is_warmup: 前 pad_b 段为 1
            if pad_b > 0:
                pos[:pad_b, 0] = 1
            # episode_index：前段=0，中段=0..real_len-1，尾段=real_len-1
            if real_len > 0:
                pos[pad_b: pad_b + real_len, 1] = np.arange(real_len, dtype=np.int64)
                if pad_b > 0:
                    pos[:pad_b, 1] = 0
                if pad_a > 0:
                    pos[pad_b + real_len:, 1] = real_len - 1
            # episode_new_pos_id：0..real_len_new-1
            pos[:, 2] = np.arange(real_len_new, dtype=np.int64)
            # episode_id
            pos[:, 3] = e
            # buffer_pos_id（1-based，全局）
            buf_start_1b = ofs + 1
            pos[:, 4] = buf_start_1b + pos[:, 2]
            # 写入 pos_info
            pos_info_arr[ofs: ofs_end] = pos

            # 更新episode_map数组（而不是最后一次性写入）
            episode_map[e, 0] = e
            episode_map[e, 1] = real_len
            episode_map[e, 2] = real_len_new
            episode_map[e, 3] = buf_start_1b
            episode_map[e, 4] = ofs_end  # 最后一个 1-based = ofs_end
            episode_map[e, 5] = win_num

            # 立即写入当前episode_map行（确保数据持久化）
            episode_map_arr[e] = episode_map[e]

            # 前进写指针
            write_ptr = ofs_end

        # 将最终 episode_map 写入 meta 组（data 组内仅包含时间维度一致的序列数据）
        episode_map_arr[:, :] = episode_map

    # 在转换函数的最后，创建 ReplayBuffer 之前添加：
    assert write_ptr == total_padded, f"Write pointer ({write_ptr}) does not match total padded length ({total_padded})"
    print(f"数据化转换结束在： {time.time() - start_time:.2f} seconds")
    # 创建ReplayBuffer（确保所有数据已写入）
    replay_buffer = ReplayBuffer(root)
    print(f"数据化转换后保存结束在： {time.time() - start_time:.2f} seconds")
    return replay_buffer


# ----------------- 工具：动作转换 -----------------
def _convert_actions_v8(raw_actions, abs_action, rotation_transformer):
    actions = raw_actions
    if abs_action:
        is_dual_arm = False
        if raw_actions.shape[-1] == 14:
            raw_actions = raw_actions.reshape(-1, 2, 7)
            is_dual_arm = True

        pos = raw_actions[..., :3]
        rot = raw_actions[..., 3:6]
        gripper = raw_actions[..., 6:]
        rot = rotation_transformer.forward(rot)
        raw_actions = np.concatenate([pos, rot, gripper], axis=-1).astype(np.float32)

        if is_dual_arm:
            raw_actions = raw_actions.reshape(-1, 20)
        actions = raw_actions
    return actions


def normalizer_from_stat(stat):
    max_abs = np.maximum(stat['max'].max(), np.abs(stat['min']).max())
    scale = np.full_like(stat['max'], fill_value=1 / max_abs)
    offset = np.zeros_like(stat['max'])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat,
    )
