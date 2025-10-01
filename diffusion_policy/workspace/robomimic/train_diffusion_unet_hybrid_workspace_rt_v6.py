if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
from termcolor import cprint
import tqdm
import numpy as np
import shutil
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.robomimic.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDiffusionUnetHybridWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch', 'optim_step', 'lr_step']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetHybridImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetHybridImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        # 在transformer中，优化器的配置是由 get_optimizer 方法提供的
        # 而在unet中，优化器通过 hydra.utils.instantiate 来直接根据配置文件进行初始化，
        # 并显式传递了 params=self.model.parameters()。
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())
        """
        *****适合适合需要灵活配置和替换优化器的场景，尤其是在实验中需要不断调整优化器超参数时
        将优化器的创建过程外部化，由外部的配置文件来处理优化器的配置，而模型本身仅提供参数。优化器实例化时需要显式传入模型参数。优势和劣势如下：
        优势：
            更高的灵活性：优化器的配置独立于模型，可以根据需要更改优化器，而无需修改模型代码。这对实验和调试时非常有用。
            配置文件中对优化器有清晰的控制，可以使用 hydra 等工具灵活地调整优化器的超参数。
            更便于调试和复用，因为优化器的配置完全在外部，不受模型内部的限制。
        劣势：
            需要明确传递参数：优化器实例化时必须显式传入模型的参数，这意味着需要在多个地方维护对模型参数的访问权。
            优化器的配置不在模型内部，可能导致在一些特殊情况下配置不一致或难以管理
        """

        # configure training state
        self.global_step = 0
        self.epoch = 0
        # 优化器步数（仅在执行 optimizer.step() 时递增），用于学习率调度对齐
        self.optim_step = 0
        # 学习率调度的 batch 级计数器（每处理完一个 dataloader 的 batch 增加一次）
        self.lr_step = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)
                self.epoch += 1
                self.global_step += 1

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler（按 dataloader 的 batch 步进：每个 batch 只更新一次 LR）
        # num_training_steps 以 batch 为单位计算
        current_lr_step = getattr(self, 'lr_step', 0)
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * cfg.training.num_epochs),
            # 使用 batch 级 lr_step 对齐 last_epoch
            last_epoch=current_lr_step-1    # last_epoch=self.global_step-1改为 current_lr_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env
        env_runner: BaseImageRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseImageRunner)

        # 引入了 termcolor 库来在控制台中输出带颜色的日志信息，使得日志更加易于阅读
        cfg.logging.name = str(cfg.logging.name)
        cprint("-----------------------------", "yellow")
        cprint(f"[WandB] group: {cfg.logging.group}", "yellow")
        cprint(f"[WandB] name: {cfg.logging.name}", "yellow")
        cprint("-----------------------------", "yellow")
        # configure logging
        print("开始设置wandb，并Logging to wandb")
        # 修改后 - 方案1：确保只传递一次 mode 参数
        # 创建 logging 配置的副本
        logging_config = OmegaConf.to_container(cfg.logging, resolve=True)
        if 'mode' in logging_config:
            logging_config.pop('mode')  # 删除 logging 配置中的 mode 参数

        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            mode="offline",
            **logging_config,  # 使用修改后的配置
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        # 创建独立的日志文件路径
        train_log_path = os.path.join(self.output_dir, 'train_logs.json.txt')
        val_log_path = os.path.join(self.output_dir, 'val_logs.json.txt')
        test_log_path = os.path.join(self.output_dir, 'test_logs.json.txt')
        with JsonLogger(train_log_path) as train_logger, \
             JsonLogger(val_log_path) as val_logger, \
             JsonLogger(test_log_path) as test_logger:
            while self.epoch < cfg.training.num_epochs:
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # ==================== START: MODIFIED TRAINING LOGIC ====================
                        
                        # 1. 获取整条 episode 的数据
                        actions = batch['action']  # Shape: [1, T_orig, D_act]
                        obs = batch['obs']         # Dict, each value shape: [1, T_obs, ...]

                        # 2. 从模型中获取参数
                        horizon = self.model.horizon
                        n_action_steps = self.model.n_action_steps # n_action_steps is 1 in your philosophy
                        n_obs_steps = self.model.n_obs_steps
                        
                        B, T_orig, D_act = actions.shape
                        T_obs = next(iter(obs.values())).shape[1]

                        # 3. 动作序列填充到长度 = T_obs + horizon（复制最后一个动作）
                        # 改为对齐更严格的总步数定义：min(len(obs), len(action)) + horizon
                        # 这样当动作序列短于观测长度时，不会额外多循环；若不足则在动作端做末帧填充
                        T_short = min(T_obs, T_orig)
                        target_len = T_short + horizon
                        pad_len = max(0, target_len - T_orig)
                        if pad_len > 0:
                            last_action = actions[:, -1:, :]
                            action_padding = last_action.repeat(1, pad_len, 1)
                            actions = torch.cat([actions, action_padding], dim=1)
                        T_padded = actions.shape[1]

                        # 4. 滑动窗口循环 (步长为 n_action_steps)
                        import math
                        # total steps equals min(T_obs, T_act) + horizon (warmup + realtime)
                        total_steps = T_short + horizon
                        # warmup: horizon + n_obs_steps - 1
                        warmup_steps = horizon + n_obs_steps - 1
                        # step stride (keep using model's n_action_steps)
                        step_stride = n_action_steps
                        num_windows = math.ceil(total_steps / step_stride)

                        for w in range(num_windows):
                            s = w * step_stride
                            if s >= total_steps:
                                break

                            # ---------- warmup phase ----------
                            if s < warmup_steps:
                                # obs: always repeat the first frame n_obs_steps times
                                window_obs = {}
                                for k, v in obs.items():
                                    # v: [B, T_obs, ...]
                                    first = v[:, 0:1]
                                    repeat_shape = [1, n_obs_steps] + [1] * (v.ndim - 2)
                                    window_obs[k] = first.repeat(*repeat_shape)

                                # actions: take available prefix actions up to index s (0..s),
                                # pad at front with copies of action[:,0] if needed to reach length=horizon
                                avail_len = min(s + 1, actions.shape[1])
                                avail_actions = actions[:, :avail_len, :]  # [B, avail_len, D_act]
                                if avail_len >= horizon:
                                    # take last horizon actions ending at index s
                                    window_actions = avail_actions[:, -horizon:, :]
                                else:
                                    pad_cnt = horizon - avail_len
                                    pad = actions[:, 0:1, :].repeat(1, pad_cnt, 1)
                                    window_actions = torch.cat([pad, avail_actions], dim=1)

                            # ---------- realtime phase ----------
                            else:
                                # realtime index j (0-based)
                                j = s - warmup_steps
                                # obs window: j .. j + n_obs_steps -1
                                start_obs_idx = j
                                end_obs_idx = j + n_obs_steps
                                window_obs = {k: v[:, start_obs_idx:end_obs_idx] for k, v in obs.items()}

                                # action window: immediately after obs window
                                start_act_idx = j + n_obs_steps
                                end_act_idx = start_act_idx + horizon
                                # actions were padded earlier to length >= min(T_obs, T_act) + horizon, so slicing is safe
                                window_actions = actions[:, start_act_idx:end_act_idx, :]

                            # 构建传递给 policy 的 batch（仅包含 obs 和 action）
                            window_batch = {
                                'obs': window_obs,
                                'action': window_actions,  # 包含真实和填充的动作
                            }

                            # 6. 计算损失并反向传播 (每个滑动步都是一个训练步)
                            raw_loss = self.model.compute_loss(window_batch)
                            loss = raw_loss / cfg.training.gradient_accumulate_every
                            loss.backward()

                            # 7. 更新优化器和EMA
                            if self.global_step % cfg.training.gradient_accumulate_every == 0:
                                # 可选梯度裁剪，稳定训练
                                clip_val = getattr(cfg.training, 'clip_grad_norm', 0.0)
                                if clip_val and clip_val > 0:
                                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_val)
                                self.optimizer.step()
                                self.optimizer.zero_grad()
                                self.optim_step += 1
                            
                            if cfg.training.use_ema:
                                ema.step(self.model)


                            # 9. 日志记录
                            raw_loss_cpu = raw_loss.item()
                            tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                            train_losses.append(raw_loss_cpu)
                            step_log = {
                                'train_loss': raw_loss_cpu,
                                'global_step': self.global_step,
                                'epoch': self.epoch,
                                'lr': lr_scheduler.get_last_lr()[0]
                            }
                            
                            # 避免在最后一个batch的最后一步重复记录
                            is_last_step_in_epoch = (batch_idx == (len(train_dataloader)-1)) and (w == num_windows - 1)
                            if not is_last_step_in_epoch:
                                wandb_run.log(step_log, step=self.global_step)
                                train_logger.log(step_log)  # 记录到训练日志

                            self.global_step += 1

                            if (cfg.training.max_train_steps is not None) and \
                                self.global_step >= cfg.training.max_train_steps:
                                break
                        
                        # 在完成当前 batch 的所有滑窗后，只进行一次学习率调度步进（按 batch 计数）
                        try:
                            lr_scheduler.step()
                        except Exception:
                            # 一些 scheduler 可能不允许在特定时刻调用 step；忽略并继续
                            pass
                        self.lr_step += 1

                        if (cfg.training.max_train_steps is not None) and \
                            self.global_step >= cfg.training.max_train_steps:
                            break
                    
                    if (cfg.training.max_train_steps is not None) \
                        and self.global_step >= cfg.training.max_train_steps:
                        break
                
                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()



                # # 在训练循环中，当需要rollout时切换到序列模式
                # dataset.set_mode(use_sequence_sampler=True)

                print("Running rollout...")
                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0:
                    runner_log = env_runner.run(policy)
                    # log all
                    step_log.update(runner_log)
                print("Rollout done.")
                
                # # 切换回compute_loss模式继续训练
                # dataset.set_mode(use_sequence_sampler=False)

                # run validation（简化：仅统计平均验证损失，不做反归一化作图与逐位置MSE）
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                # 验证阶段同样按滑动窗口计算平均损失
                                actions = batch['action']
                                obs = batch['obs']
                                horizon = self.model.horizon
                                n_action_steps = self.model.n_action_steps
                                n_obs_steps = self.model.n_obs_steps
                                B, T_orig, D_act = actions.shape
                                T_obs = next(iter(obs.values())).shape[1]

                                # 与训练相同，对动作做末帧填充到 min(T_obs, T_act) + horizon
                                T_short = min(T_obs, T_orig)
                                target_len = T_short + horizon
                                pad_len = max(0, target_len - T_orig)
                                if pad_len > 0:
                                    last_action = actions[:, -1:, :]
                                    action_padding = last_action.repeat(1, pad_len, 1)
                                    actions = torch.cat([actions, action_padding], dim=1)

                                import math
                                # mirror training: total_steps = min(T_obs, T_act) + horizon
                                total_steps = T_short + horizon
                                warmup_steps = horizon + n_obs_steps - 1
                                step_stride = n_action_steps
                                num_windows = math.ceil(total_steps / step_stride)
                                seq_losses = []
                                for w in range(num_windows):
                                    s = w * step_stride
                                    if s >= total_steps:
                                        break

                                    if s < warmup_steps:
                                        # warmup: repeat first obs n_obs_steps times
                                        window_obs = {}
                                        for k, v in obs.items():
                                            first = v[:, 0:1]
                                            repeat_shape = [1, n_obs_steps] + [1] * (v.ndim - 2)
                                            window_obs[k] = first.repeat(*repeat_shape)

                                        avail_len = min(s + 1, actions.shape[1])
                                        avail_actions = actions[:, :avail_len, :]
                                        if avail_len >= horizon:
                                            window_actions = avail_actions[:, -horizon:, :]
                                        else:
                                            pad_cnt = horizon - avail_len
                                            pad = actions[:, 0:1, :].repeat(1, pad_cnt, 1)
                                            window_actions = torch.cat([pad, avail_actions], dim=1)
                                    else:
                                        j = s - warmup_steps
                                        start_obs_idx = j
                                        end_obs_idx = j + n_obs_steps
                                        window_obs = {k: v[:, start_obs_idx:end_obs_idx] for k, v in obs.items()}
                                        start_act_idx = j + n_obs_steps
                                        end_act_idx = start_act_idx + horizon
                                        window_actions = actions[:, start_act_idx:end_act_idx, :]

                                    window_batch = {
                                        'obs': window_obs,
                                        'action': window_actions,
                                    }
                                    with torch.no_grad():
                                        loss = policy.compute_loss(window_batch)
                                        seq_losses.append(loss.item())


                                if len(seq_losses) > 0:
                                    val_losses.append(float(np.mean(seq_losses)))
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss
                            # 创建专门的验证日志条目
                            val_log_entry = {
                                'global_step': self.global_step,
                                'epoch': self.epoch,
                                'val_loss': val_loss
                            }
                            val_logger.log(val_log_entry)  # 记录到验证日志

                print("Validation done.")
                # 简化：不进行额外的采样、误差计算或作图保存

                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                
                # 不再使用单一的json_logger，而是分别记录
                train_logger.log(step_log)  # 记录完整的step_log到训练日志


                self.global_step += 1
                self.epoch += 1

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetHybridWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()

