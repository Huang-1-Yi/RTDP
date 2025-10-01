from typing import Dict, Optional, Tuple, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
try:
    import robomimic.models.base_nets as rmbn
    if not hasattr(rmbn, 'CropRandomizer'):
        raise ImportError("CropRandomizer is not in robomimic.models.base_nets")
except ImportError:
    import robomimic.models.obs_core as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
from diffusion_policy.model.vision.rot_randomizer import RotRandomizer


class SlidingWindowDiffusionPolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            crop_shape=(76, 76),
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            rot_aug=False,
            # 模型时间步：训练/推理统一使用固定常数时间步（不随位置变化）
            model_fixed_timestep: int = 1,
            window_min_weight: float = 0.02,
            # 滑动窗口相关参数
            window_loss_weights="exponential",  # 窗口内不同位置的损失权重
            window_exp_gamma: float = 0.2,       # 指数权重衰减系数
            parallel_value=1,  # 并行扩展的批次大小
            parallel_agg: str = "mean",         # P 聚合方式：mean/median
            **kwargs):
        super().__init__()

        # 解析shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # 获取原始robomimic配置
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')
        
        with config.unlocked():
            config.observation.modalities.obs = obs_config
            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # 初始化全局状态
        ObsUtils.initialize_obs_utils_with_config(config)

        # 加载模型
        policy: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=action_dim,
                device='cpu',
            )

        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
        
        if obs_encoder_group_norm:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
            )

        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )

        # 创建扩散模型
        obs_feature_dim = obs_encoder.output_shape()[0]
        print(f"Obs encoder output shape: {obs_encoder.output_shape()}")
        print(f"Obs feature dim: {obs_feature_dim}")

        input_dim = action_dim
        global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
        
        # 初始化noise_scheduler
        self.noise_scheduler = noise_scheduler

        # 预计算alpha_bar - 修改为基于horizon
        max_horizon = horizon  # 使用horizon作为最大时间步
        beta_schedule = self.noise_scheduler.config.beta_schedule
        if beta_schedule == "squaredcos_cap_v2":
            s = 0.008
            t = torch.arange(0, max_horizon + 1, dtype=torch.float32)
            alpha_bar = torch.cos((t / max_horizon + s) / (1 + s) * math.pi * 0.5) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
        else:
            beta_start = self.noise_scheduler.config.beta_start
            beta_end = self.noise_scheduler.config.beta_end
            betas = torch.linspace(beta_start, beta_end, max_horizon, dtype=torch.float32)
            alphas = 1.0 - betas
            alpha_bar = torch.cumprod(alphas, dim=0)
            alpha_bar = torch.cat([torch.tensor([1.0]), alpha_bar])
        self.register_buffer('alpha_bar', alpha_bar)

        self.normalizer = LinearNormalizer()
        self.rot_randomizer = RotRandomizer() if rot_aug else None
        self.rot_aug = rot_aug

        self.parallel_value = parallel_value    # 并行扩展的批次大小
        self.horizon = horizon                  # 滑动窗口参数
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.rot_aug = rot_aug
        self.kwargs = kwargs
        # 统一的模型时间步（与噪声强度无关，仅作为时间嵌入常量）
        self.model_fixed_timestep = int(model_fixed_timestep)
        self.parallel_agg = str(parallel_agg).lower()
        if self.parallel_agg not in ("mean", "median"):
            raise ValueError(f"Unsupported parallel_agg: {self.parallel_agg}")

        
        # 设置窗口内损失权重 (根据你的理念，从1开始衰减)
        if window_loss_weights == "linear":
            # weight_i = 1 - i/horizon for i in 0 to H-1
            self.window_weights = 1.0 - torch.arange(horizon, dtype=torch.float32) / horizon
        elif window_loss_weights == "exponential":
            self.window_weights = torch.exp(-torch.arange(horizon, dtype=torch.float32) * float(window_exp_gamma))  # 0.2
        elif window_loss_weights == "constant":
            self.window_weights = torch.ones(horizon)
        else:
            raise ValueError(f"Unknown window loss weights: {window_loss_weights}")
        # 应用最小权重下限，避免远期位置学习信号过弱
        if window_min_weight > 0.0:
            self.window_weights = torch.clamp(self.window_weights, min=float(window_min_weight))
        
        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
        
        # 推理状态
        self._inference_buffer = None
        self._inference_t_buffer = None
        self._inference_global_cond = None

        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        按照指定流程计算单个滑动窗口的损失：
        - obs / action 归一化，可选旋转增强
        - 使用前 n_obs_steps 的观测编码为全局条件
        - 按位置 1..H 生成自定义加噪 noisy_trajectory
        - 使用 prediction_type 决定目标（epsilon 或 sample）
        - 对时间维进行加权，再按特征和批次做归约
        返回：{'loss': 标量, 'pred_actions': 归一化下的预测动作x0}
        """
        # 1. 检查 batch 必备键
        assert 'obs' in batch
        assert 'action' in batch

        # 2. 归一化 obs / action，并设置轨迹为归一化后的动作
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])  # [B,H,D]
        trajectory = nactions

        # 3. 旋转增强（可选）
        if self.rot_aug and self.rot_randomizer is not None:
            nobs, nactions = self.rot_randomizer(nobs, nactions)
            trajectory = nactions

        # 4. 获取形状并检查窗口长度
        B, H, D = nactions.shape
        assert H == self.horizon, f"Window size H={H} must be equal to horizon H={self.horizon}"

        # 5. 并行倍数 P
        P = int(getattr(self, 'parallel_value', 1))
        P = max(P, 1)

        # 6. 条件与时间步（固定常数）
        local_cond = None
        timesteps = torch.full((B,), self.model_fixed_timestep, device=nactions.device, dtype=torch.long)
        if P > 1:
            timesteps = timesteps.repeat_interleave(P, dim=0)  # [BP]

        # 7. 观测编码作为全局条件编码（只取前 n_obs_steps 帧）
        this_nobs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        global_cond = nobs_features.reshape(B, -1)
        if P > 1:
            global_cond = global_cond.repeat_interleave(P, dim=0)  # [BP,G]

        # 8. 动作加噪，替代 scheduler.add_noise
        if P > 1:
            trajectory = trajectory.repeat_interleave(P, dim=0)  # [BP,H,D]
        noise = torch.randn(trajectory.shape, device=trajectory.device, dtype=trajectory.dtype)
        noise_levels = torch.arange(1, H + 1, device=nactions.device, dtype=torch.long)  # [1..H]
        alpha_bar_t = self.alpha_bar[noise_levels].reshape(1, H, 1)  # [1,H,1]
        noisy_trajectory = torch.sqrt(alpha_bar_t) * trajectory + torch.sqrt(1 - alpha_bar_t) * noise  # [BP,H,D]

    # 9. 前向预测
        pred_all = self.model(noisy_trajectory, timesteps, local_cond=local_cond, global_cond=global_cond)  # [BP,H,D]

        # 10. 如有并行，按 P 平均到 [B,H,D]
        if P > 1:
            pred_view = pred_all.view(B, P, H, D)
            if self.parallel_agg == 'mean':
                pred = pred_view.mean(dim=1)
            else:
                pred = pred_view.median(dim=1).values
        else:
            pred = pred_all

        # 15. 目标使用原始轨迹（sample 监督）
        if P > 1:
            traj_for_loss = trajectory.view(B, P, H, D).mean(dim=1)
        else:
            traj_for_loss = trajectory
        target = traj_for_loss
        pred_actions = pred

        # 16. MSE（不归约）
        loss_elem = F.mse_loss(pred, target, reduction='none')  # [B,H,D]

        # 17-18. 按时间加权并归约
        weights = self.window_weights.to(nactions.device).reshape(1, H, 1)  # [1,H,1]
        loss_weighted = (loss_elem * weights).sum(dim=1)  # [B,D]
        loss_per_sample = loss_weighted.mean(dim=-1)      # [B]
        loss = loss_per_sample.mean()                     # 标量

        return {
            'loss': loss,
            'pred_actions': pred_actions  # 归一化空间下的 x0 预测 [B,H,D]
        }
    
    def _initialize_inference_buffer(self, obs_dict: Dict[str, torch.Tensor]):
        """
        初始化推理缓冲区（考虑并行维度 P）。
        """
        B = next(iter(obs_dict.values())).shape[0]
        device = self.device
        dtype = self.dtype
        H = self.horizon
        D = self.action_dim
        P = int(getattr(self, 'parallel_value', 1))
        P = max(P, 1)
        
        # 创建基础高斯噪声
        if P > 1:
            base_noise = torch.randn(B*P, H, D, device=device, dtype=dtype)
        else:
            base_noise = torch.randn(B, H, D, device=device, dtype=dtype)
        
        # 为每个位置设置噪声级别
        noise_levels = torch.arange(1, H+1, device=device, dtype=torch.long)
        alpha_bar = self.alpha_bar[noise_levels].reshape(1, H, 1)
        
        # 应用噪声级别
        initialized_buffer = torch.sqrt(1 - alpha_bar) * base_noise
        
        # 如果使用并行，需要聚合
        if P > 1:
            init_view = initialized_buffer.view(B, P, H, D)
            if self.parallel_agg == 'mean':
                initialized_buffer = init_view.mean(dim=1)
            else:
                initialized_buffer = init_view.median(dim=1).values
        
        # 设置缓冲区
        self._inference_buffer = initialized_buffer
        
        # 编码当前观测作为全局条件
        nobs = self.normalizer.normalize(obs_dict)
        this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        self._inference_global_cond = nobs_features.reshape(B, -1)

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        推理与训练对齐：
        - 使用推理缓冲区作为 x0 估计，按位置 1..H 施加噪声，构造 noisy_trajectory
        - 前向预测得到 pred；若 prediction_type=='epsilon'，重建 x0 作为动作预测
        - 执行第一个动作，并将缓冲区左移以形成下次的前缀
        返回：{'action': 反归一化的第一步动作, 'action_pred': 反归一化的全窗口预测}
        """
        # 1. 批次大小与设备
        B = next(iter(obs_dict.values())).shape[0]
        device = self.device

        # 2. 初始化或更新推理缓冲区与全局条件
        if self._inference_buffer is None or self._inference_buffer.shape[0] != B:
            self._initialize_inference_buffer(obs_dict)
        else:
            nobs = self.normalizer.normalize(obs_dict)
            this_nobs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            self._inference_global_cond = nobs_features.reshape(B, -1)

        # 3. 形状与并行倍数
        H = self.horizon
        D = self.action_dim
        P = int(getattr(self, 'parallel_value', 1))
        P = max(P, 1)

        # 4. 构造 noisy_trajectory（以缓冲区作为 x0 估计）
        x0_guess = self._inference_buffer  # [B,H,D]
        if P > 1:
            x0_rep = x0_guess.repeat_interleave(P, dim=0)  # [BP,H,D]
        else:
            x0_rep = x0_guess
        noise = torch.randn_like(x0_rep)
        noise_levels = torch.arange(1, H + 1, device=device, dtype=torch.long)
        alpha_bar_t = self.alpha_bar[noise_levels].reshape(1, H, 1).to(device)
        noisy_trajectory = torch.sqrt(alpha_bar_t) * x0_rep + torch.sqrt(1 - alpha_bar_t) * noise

        # 5. 固定时间步与全局条件
        timesteps = torch.full((B,), self.model_fixed_timestep, device=device, dtype=torch.long)
        if P > 1:
            timesteps = timesteps.repeat_interleave(P, dim=0)
            global_cond = self._inference_global_cond.repeat_interleave(P, dim=0)
        else:
            global_cond = self._inference_global_cond

        # 6. 前向并根据 prediction_type 得到 x0 预测
        pred_all = self.model(noisy_trajectory, timesteps, local_cond=None, global_cond=global_cond)  # [BP,H,D]
        if P > 1:
            pred_view = pred_all.view(B, P, H, D)
            if self.parallel_agg == 'mean':
                pred = pred_view.mean(dim=1)
            else:
                pred = pred_view.median(dim=1).values
        else:
            pred = pred_all

        # 6. 直接将模型输出视为 x0 预测（与训练一致，监督为 sample）
        pred_actions = pred

        # 7. 取第一步动作并更新缓冲区（左移），并为最后一位注入最大噪声级别的噪声
        action_to_execute = pred_actions[:, 0:1, :]  # [B,1,D]
        if H > 1:
            self._inference_buffer[:, :H-1, :] = pred_actions[:, 1:, :]
        # 为下一次推理准备新的噪声输入到最后一位
        if P > 1:
            new_noise = torch.randn(B*P, 1, D, device=device, dtype=self.dtype)
            new_noise_view = new_noise.view(B, P, 1, D)
            if self.parallel_agg == 'mean':
                new_noise_mean = new_noise_view.mean(dim=1)
            else:
                new_noise_mean = new_noise_view.median(dim=1).values
        else:
            new_noise_mean = torch.randn(B, 1, D, device=device, dtype=self.dtype)
        alpha_bar_H = self.alpha_bar[H].reshape(1, 1, 1).to(device)
        new_noisy_action = torch.sqrt(1 - alpha_bar_H) * new_noise_mean
        self._inference_buffer[:, H-1:, :] = new_noisy_action

        # 8. 反归一化输出
        action_out = self.normalizer['action'].unnormalize(action_to_execute)
        full_pred_out = self.normalizer['action'].unnormalize(pred_actions)

        return {
            'action': action_out.reshape(B, self.n_action_steps, self.action_dim),
            'action_pred': full_pred_out
        }


    def reset(self):
        """重置推理状态"""
        self._inference_buffer = None
        self._inference_global_cond = None
    
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())