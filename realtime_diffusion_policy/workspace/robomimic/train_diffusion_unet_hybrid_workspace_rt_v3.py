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
    include_keys = ['global_step', 'epoch', 'lr_step']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model: DiffusionUnetHybridImagePolicy = hydra.utils.instantiate(cfg.policy)
        self.ema_model: DiffusionUnetHybridImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())
        self.global_step = 0
        self.epoch = 0
        self.lr_step = 0        # 策略新增，区分预热和非预热阶段的步数

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)
                self.epoch += 1
                self.global_step += 1

        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler（按 dataloader 的 batch 步进：每个 batch 只更新一次 LR）
        # num_training_steps 以 batch 为单位计算
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * cfg.training.num_epochs),
            # 使用 batch 级 lr_step 对齐 last_epoch
            last_epoch=self.lr_step-1    # last_epoch由self.global_step-1改为self.lr_step-1
        )

        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

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
        print("开始设置wandb，并Logging to wandb")
        # 创建 logging 配置的副本
        logging_config = OmegaConf.to_container(cfg.logging, resolve=True)
        if 'mode' in logging_config:
            logging_config.pop('mode')  # 删除 logging 配置中的 mode 参数，确保只传递一次 mode 参数

        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            mode="offline",
            **logging_config,  # 使用修改后的配置
        )
        wandb.config.update({
                "output_dir": self.output_dir,
        })

        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

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
                        # 先提取 window_info 等元数据，再将剩余的数据搬到 GPU
                        if 'window_info' in batch:
                            window_info = batch.pop('window_info')# 保留在 CPU，如需使用可单独读取，避免将 window_info 等元数据搬到 GPU
                            warm_up_flags = window_info[:, 0, 0]  # [B]
                        
                        # 搬到 GPU
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch    # 用于采样监控训练进度
                        
                        # 计算每个样本的损失
                        raw_loss = self.model.compute_loss(batch) # 计算每个样本的损失，返回 [B] 张量

                        # 计算整个批次的平均损失
                        batch_loss = raw_loss.mean()  # 标量
                        
                        loss = batch_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        
                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        # logging
                        raw_loss_cpu = batch_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            wandb_run.log(step_log, step=self.global_step)
                            step_log = {
                                'train_loss': raw_loss_cpu,
                                'global_step': self.global_step,
                                'epoch': self.epoch,
                                'lr': lr_scheduler.get_last_lr()[0]
                            }
                        self.global_step += 1  # ✅ 移动到外层

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()


                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0:
                    print("Running rollout...")
                    runner_log = env_runner.run(policy)
                    
                    # 新增：提取测试分数并记录到测试日志
                    test_score = runner_log.get('test/mean_score', None)  # 注意键名是 'test/mean_score'
                    if test_score is not None:
                        test_log_entry = {
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'test_mean_score': test_score
                        }
                        test_logger.log(test_log_entry)
                        print(f"记录测试分数: {test_score} (epoch {self.epoch})")

                    # log all
                    step_log.update(runner_log)
                    print("Rollout done.")


                # run validation（仅统计平均验证损失，不做反归一化作图与逐位置MSE）
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        total_non_warmup_loss = 0.0
                        total_non_warmup_samples = 0
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                warm_up_flags = None
                                if 'window_info' in batch:
                                    window_info = batch.pop('window_info')# 保留在 CPU，如需使用可单独读取，避免将 window_info 等元数据搬到 GPU
                                    # 提取每个序列的第一个时间步的 warm_up_flag
                                    warm_up_flags = window_info[:, 0, 0]  # [B]

                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                realtime_losses = self.model.compute_loss(batch)  # [B]
                                
                                # 只记录非预热样本的损失，预热期间的损失会被自动滤除
                                non_warmup_losses = realtime_losses[~warm_up_flags.bool()]
                                val_losses.append(float(np.mean(non_warmup_losses))) 
                                if warm_up_flags is not None:
                                    # 确保 warm_up_flags 在 GPU 上
                                    warm_up_flags = warm_up_flags.to(device)
                                    
                                    # 创建非预热样本的掩码
                                    non_warmup_mask = (warm_up_flags == 0)
                                    
                                    # 提取非预热样本的损失
                                    non_warmup_losses = realtime_losses[non_warmup_mask]
                                else:
                                    # 如果没有 window_info，使用所有样本
                                    non_warmup_losses = realtime_losses
                                
                                # 累加损失和样本数量
                                if len(non_warmup_losses) > 0:
                                    total_non_warmup_loss += non_warmup_losses.sum().item()
                                    total_non_warmup_samples += len(non_warmup_losses)

                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        
                        if len(val_losses) > 0:
                            val_loss = total_non_warmup_loss / total_non_warmup_samples

                        else:
                            val_loss = 0.0
                            print("Warning: No non-warmup samples found in validation.")
                        
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


                # 在每个 epoch 结束时进行额外的采样、误差计算
                # 对训练集采样的第一个batch，使用当前 policy 进行动作预测，并与真实动作（gt_action）计算均方误差（MSE），将误差记录到日志（step_log['train_action_mse_error']）。
                # 监控模型在训练集上的动作预测效果。
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        obs_dict = batch['obs']
                        gt_action = batch['action']
                        
                        result = policy.predict_action(obs_dict)
                        pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse
                        # torch.cuda.empty_cache()

                # checkpoint save
                if (self.epoch % cfg.training.checkpoint_every) == 0:
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

                wandb_run.log(step_log, step=self.global_step)
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

