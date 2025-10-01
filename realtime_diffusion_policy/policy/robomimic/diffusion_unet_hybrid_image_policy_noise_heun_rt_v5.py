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
            parallel_value=1,  # 并行扩展的批次大小
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

        
        # 设置窗口内损失权重 (根据你的理念，从1开始衰减)
        if window_loss_weights == "linear":
            # weight_i = 1 - i/horizon for i in 0 to H-1
            self.window_weights = 1.0 - torch.arange(horizon, dtype=torch.float32) / horizon
        elif window_loss_weights == "exponential":
            self.window_weights = torch.exp(-torch.arange(horizon, dtype=torch.float32) * 0.2)
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
        计算单个滑动窗口的损失。
        此函数根据“预测真实值”的理念实现。
        它接收obs、action（可能部分缺失）、历史预测，在内部补全缺失位置，加噪预测。
        返回损失和预测动作。
        """
        # 1. 验证输入batch中必须包含obs、action和history_pred键
        assert 'obs' in batch
        assert 'action' in batch
        assert 'history_pred' in batch
        # 2. 对观测数据进行归一化处理
        nobs = self.normalizer.normalize(batch['obs'])
        
        # 3. 对动作数据进行归一化处理，作为监督目标
        nactions = self.normalizer['action'].normalize(batch['action'])  # [B, H, D]
        
        # 4. 获取历史预测数据（上一次预测的裁剪结果）
        history_pred = batch['history_pred']  # [B, horizon-1, D_act] 或 None

        # 5. 获取批次大小(B)、序列长度(T)和动作维度(D)
        B, T, D = nactions.shape
        
        # 6. 获取预设的滑动窗口长度(horizon)
        H = self.horizon
        
        # 7. 验证输入序列长度必须等于预设的horizon
        assert T == H, f"Window size T={T} must be equal to horizon H={H}"

        # 8. 获取并行重复倍数P（用于MC降噪），默认为1表示不并行
        P = int(getattr(self, 'parallel_value', 1))
        P = max(P, 1)

        # 9. 创建噪声级别数组，从1到H（对应扩散过程的时间步）
        noise_levels = torch.arange(1, H + 1, device=nactions.device, dtype=torch.long)  # [1..H]


        # 组装模型输入（模拟因果迭代去噪）：
        #    - 首个窗口（history_pred is None）：对真实动作按位置1..H施加增噪，作为输入
        #    - 随后窗口：前H-1步使用上一窗口预测值（视作当前去噪状态），最后一步用真实动作加最大噪声（级别H）
        
        # 10. 根据是否有历史预测，采用不同的输入构建策略
        if history_pred is None:
            # 位置相关加噪：x_t = sqrt(alpha_bar[t]) * x0 + sqrt(1-alpha_bar[t]) * eps
            # 批次扩展到 BP
        # 11. 对于首个窗口：对真实动作按位置1..H施加增噪
        # 12. 如果并行倍数P>1，扩展批次维度
            if P > 1:
                nactions_rep = nactions.repeat_interleave(P, dim=0)  # [BP,H,D]
            else:
                nactions_rep = nactions
            
            # 13. 生成随机噪声
            eps = torch.randn(nactions_rep.shape, device=nactions.device, dtype=nactions.dtype)
            
            # 14. 获取对应噪声级别的alpha_bar值
            alpha_bar_t = self.alpha_bar[noise_levels].reshape(1, T, 1)  # [1,H,1]
            
            # 15. 根据扩散公式计算带噪声的输入：x_t = sqrt(alpha_bar[t]) * x0 + sqrt(1-alpha_bar[t]) * eps
            model_input = torch.sqrt(alpha_bar_t) * nactions_rep + torch.sqrt(1 - alpha_bar_t) * eps  # [BP,H,D]
        else:
            # 16. 对于后续窗口：使用历史预测作为前缀
            # 17. 验证历史预测的形状正确
            assert history_pred.shape[1] == H - 1, \
                f"History pred shape {history_pred.shape} != expected [B, {H-1}, D]"
            
            # 18. 前H-1步使用上一窗口的预测结果
            prefix = history_pred  # [B, H-1, D]
            
            # 19. 获取最后一步的真实动作
            last_true = nactions[:, -1:, :]  # [B,1,D]
            
            # 20. 如果并行倍数P>1，扩展前缀和最后一步的真实动作
            if P > 1:
                prefix = prefix.repeat_interleave(P, dim=0)       # [BP,H-1,D]
                last_true_rep = last_true.repeat_interleave(P, 0) # [BP,1,D]
            else:
                last_true_rep = last_true
            
            # 21. 为最后一步生成随机噪声
            eps_last = torch.randn_like(last_true_rep)
            
            # 22. 获取最大噪声级别(H)对应的alpha_bar值
            alpha_bar_H = self.alpha_bar[noise_levels[-1]].reshape(1, 1, 1)  # [1,1,1]
            
            # 23. 对最后一步的真实动作添加最大噪声
            last_noisy = torch.sqrt(alpha_bar_H) * last_true_rep + torch.sqrt(1 - alpha_bar_H) * eps_last
            
            # 24. 将前缀和带噪声的最后一步拼接成完整输入
            model_input = torch.cat([prefix, last_noisy], dim=1)  # [BP,H,D]

        # 25. 准备全局条件（观测特征）
        # 26. 将观测数据重塑为(B*T, ...)的形状
        this_obs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
        
        # 27. 通过观测编码器提取特征
        obs_features = self.obs_encoder(this_obs)
        
        # 28. 将特征重塑为(B, -1)作为全局条件
        global_cond = obs_features.reshape(B, -1)
        
        # 29. 如果并行倍数P>1，扩展全局条件
        if P > 1:
            global_cond_rep = global_cond.repeat_interleave(P, dim=0)  # [BP,G]
        else:
            global_cond_rep = global_cond

        # 30. 使用模型预测真实动作值（模型输出直接回归x0，即真实动作）
        # 32. 创建固定时间步张量（所有样本使用相同的时间步）
        timesteps = torch.full((B,), self.model_fixed_timestep, device=nactions.device, dtype=torch.long)
        
        
        # 33. 如果并行倍数P>1，扩展时间步张量
        if P > 1:
            timesteps = timesteps.repeat_interleave(P, dim=0)  # [BP]
        
        # 34. 调用模型进行预测
        pred_actions_all = self.model(model_input, timesteps, global_cond=global_cond_rep)  # [BP,H,D]
        
        # 35. 如果并行倍数P>1，对并行维度的预测结果求平均
        if P > 1:
            pred_actions = pred_actions_all.view(B, P, H, D).mean(dim=1)
        else:
            pred_actions = pred_actions_all


        # 36. 计算加权损失
        # 37. 根据是否并行和是否有历史预测，选择不同的目标值计算MSE损失
        if P > 1 and history_pred is None:
            nactions_for_loss = nactions.repeat_interleave(P, dim=0)
            loss_per_element = F.mse_loss(pred_actions_all, nactions_for_loss, reduction='none')  # [BP,H,D]
        elif P > 1 and history_pred is not None:
            nactions_for_loss = nactions.repeat_interleave(P, dim=0)
            loss_per_element = F.mse_loss(pred_actions_all, nactions_for_loss, reduction='none')
        else:
            loss_per_element = F.mse_loss(pred_actions, nactions, reduction='none')  # [B,H,D]
        
        # 38. 在批次和特征维度上求平均，得到每个时间步的损失
        loss_per_timestep = loss_per_element.mean(dim=[0, 2])  # 形状从 [B,H,D] 变为 [H]
        
        # 39. 获取预设的时间步权重
        weights = self.window_weights.to(nactions.device)
        
        # 40. 在时间维度上进行加权平均，得到最终损失
        loss = (loss_per_timestep * weights).sum()

        # 41. 返回损失和预测动作
        ret = {
            'loss': loss,
            'pred_actions': pred_actions  # [B, H, D]
        }
        
        # 42. 如果并行倍数P>1，同时返回所有并行预测结果
        if P > 1:
            ret['pred_actions_all'] = pred_actions_all  # [B*P, H, D]
        return ret
    


    def _initialize_inference_buffer(self, obs_dict: Dict[str, torch.Tensor]):
        """
        初始化推理缓冲区，用纯高斯噪声填充。
        """
        B = next(iter(obs_dict.values())).shape[0]
        device = self.device
        dtype = self.dtype
        
        # 缓冲区用纯高斯噪声填充，代表了对未来的完全不确定性
        self._inference_buffer = torch.randn(
            B, self.horizon, self.action_dim, device=device, dtype=dtype)
        
        # 编码当前观测作为全局条件
        nobs = self.normalizer.normalize(obs_dict)
        this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        self._inference_global_cond = nobs_features.reshape(B, -1)

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        基于“预测真实值”的滑动窗口推理。
        """
        # 1. 获取批次大小
        B = next(iter(obs_dict.values())).shape[0]
        
        # 2. 获取设备信息
        device = self.device

        # 3. 初始化或更新推理缓冲区
        if self._inference_buffer is None or self._inference_buffer.shape[0] != B:
            # 4. 如果缓冲区不存在或批次大小不匹配，初始化缓冲区
            self._initialize_inference_buffer(obs_dict)
        else:
            # 5. 更新观测条件
            # 6. 对观测数据进行归一化
            nobs = self.normalizer.normalize(obs_dict)
            
            # 7. 将观测数据重塑为(B*T, ...)的形状
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            
            # 8. 通过观测编码器提取特征
            nobs_features = self.obs_encoder(this_nobs)
            
            # 9. 将特征重塑为(B, -1)作为全局条件
            self._inference_global_cond = nobs_features.reshape(B, -1)

        # 10. 进行模型预测
        # 11. 获取滑动窗口长度和动作维度
        H = self.horizon
        D = self.action_dim
        
        # 12. 获取缓冲区中的前缀（前H-1步的预测）
        prefix = self._inference_buffer[:, :H-1, :] if H > 1 else None
        
        # 13. 获取并行重复倍数P
        P = int(getattr(self, 'parallel_value', 1))
        P = max(P, 1)
        
        # 14. 获取最大噪声级别对应的alpha_bar值
        aH = self.alpha_bar[self.horizon].reshape(1, 1, 1).to(device)
        
        # 15. 根据并行倍数采用不同的处理方式
        if P > 1:
            # 16. 扩展前缀（如果存在）
            if prefix is not None:
                prefix_rep = prefix.repeat_interleave(P, dim=0)  # [BP,H-1,D]
            else:
                prefix_rep = None
            
            # 17. 为最后一步生成随机噪声
            eps = torch.randn(B*P, 1, D, device=device, dtype=self.dtype)
            
            # 18. 计算带噪声的最后一步输入
            last_noise = torch.sqrt(1 - aH) * eps
            
            # 19. 构建模型输入：拼接前缀和带噪声的最后一步
            model_input = last_noise if prefix_rep is None else torch.cat([prefix_rep, last_noise], dim=1)
            
            # 20. 扩展全局条件
            global_cond_rep = self._inference_global_cond.repeat_interleave(P, dim=0)  # [BP,G]
            
            # 21. 创建固定时间步张量并扩展
            timesteps = torch.full((B,), self.model_fixed_timestep, device=device, dtype=torch.long)
            timesteps = timesteps.repeat_interleave(P, dim=0)  # [BP]
            
            # 22. 调用模型进行预测
            pred_actions_all = self.model(model_input, timesteps, global_cond=global_cond_rep)  # [BP,H,D]
            
            # 23. 对并行维度的预测结果求平均
            pred_actions = pred_actions_all.view(B, P, H, D).mean(dim=1)
        else:
            # 24. 非并行模式：为最后一步生成随机噪声
            eps = torch.randn(B, 1, D, device=device, dtype=self.dtype)
            
            # 25. 计算带噪声的最后一步输入
            last_noise = torch.sqrt(1 - aH) * eps
            
            # 26. 构建模型输入：拼接前缀和带噪声的最后一步
            model_input = last_noise if prefix is None else torch.cat([prefix, last_noise], dim=1)
            
            # 27. 创建固定时间步张量
            timesteps = torch.full((B,), self.model_fixed_timestep, device=device, dtype=torch.long)
            
            # 28. 调用模型进行预测
            pred_actions = self.model(model_input, timesteps, global_cond=self._inference_global_cond)

        # 29. 提取要执行的动作（窗口的第一个动作，去噪最彻底）
        action_to_execute = pred_actions[:, 0:1, :] # Shape: [B, 1, D]

        # 30. 滑动窗口并准备下一次迭代
        if H > 1:
            # 31. 更新缓冲区：使用本次预测去掉已执行的第一个时间步
            self._inference_buffer[:, :H-1, :] = pred_actions[:, 1:, :]
        
        # 32. 末尾保持为空位（将由下一次再次填充噪声）

        # 33. 对预测结果进行反归一化
        unnormalized_action_to_execute = self.normalizer['action'].unnormalize(action_to_execute)
        unnormalized_full_prediction = self.normalizer['action'].unnormalize(pred_actions)
        
        # 34. 返回执行动作和完整预测
        return {
            'action': unnormalized_action_to_execute.reshape(B, self.n_action_steps, self.action_dim),
            'action_pred': unnormalized_full_prediction
        }


    def reset(self):
        """重置推理状态"""
        self._inference_buffer = None
        self._inference_global_cond = None
    
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())