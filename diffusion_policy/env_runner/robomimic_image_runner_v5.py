import os
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import h5py
import math
import dill
import wandb.sdk.data_types.video as wv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.model.common.rotation_transformer import RotationTransformer

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
# 可选引入绘图库（若不存在则仅保存NPY）
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception:
    plt = None
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils


def create_env(env_meta, shape_meta, enable_render=True):
    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta['obs'].items():
        modality_mapping[attr.get('type', 'low_dim')].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, 
        render_offscreen=enable_render,
        use_image_obs=enable_render, 
    )
    return env


class RobomimicImageRunner(BaseImageRunner):
    """
    Robomimic envs already enforces number of steps.
    """

    def __init__(self, 
            output_dir,
            dataset_path,
            shape_meta:dict,
            n_train=10,
            n_train_vis=3,
            train_start_idx=0,
            n_test=22,
            n_test_vis=6,
            test_start_seed=10000,
            max_steps=400,
            n_obs_steps=2,
            n_action_steps=8,
            render_obs_key='agentview_image',
            fps=10,
            crf=22,
            past_action=False,
            abs_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None,
            # debug_save_actions: bool = False,
            debug_save_actions: bool = True,
            debug_save_details: bool = False,
        ):
        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test

        # assert n_obs_steps <= n_action_steps
        dataset_path = os.path.expanduser(dataset_path)
        robosuite_fps = 20
        steps_per_render = max(robosuite_fps // fps, 1)

        # read from dataset
        env_meta = FileUtils.get_env_metadata_from_dataset(
            dataset_path)
        # disable object state observation
        env_meta['env_kwargs']['use_object_obs'] = False

        rotation_transformer = None
        if abs_action:
            env_meta['env_kwargs']['controller_configs']['control_delta'] = False
            rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

        def env_fn():
            robomimic_env = create_env(
                env_meta=env_meta, 
                shape_meta=shape_meta
            )
            # Robosuite's hard reset causes excessive memory consumption.
            # Disabled to run more envs.
            # https://github.com/ARISE-Initiative/robosuite/blob/92abf5595eddb3a845cd1093703e5a3ccd01e77e/robosuite/environments/base.py#L247-L248
            robomimic_env.env.hard_reset = False
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    RobomimicImageWrapper(
                        env=robomimic_env,
                        shape_meta=shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )
        
        # For each process the OpenGL context can only be initialized once
        # Since AsyncVectorEnv uses fork to create worker process,
        # a separate env_fn that does not create OpenGL context (enable_render=False)
        # is needed to initialize spaces.
        def dummy_env_fn():
            robomimic_env = create_env(
                    env_meta=env_meta, 
                    shape_meta=shape_meta,
                    enable_render=False
                )
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    RobomimicImageWrapper(
                        env=robomimic_env,
                        shape_meta=shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()

        # train
        with h5py.File(dataset_path, 'r') as f:
            for i in range(n_train):
                train_idx = train_start_idx + i
                enable_render = i < n_train_vis
                init_state = f[f'data/demo_{train_idx}/states'][0]

                def init_fn(env, init_state=init_state, 
                    enable_render=enable_render):
                    # setup rendering
                    # video_wrapper
                    assert isinstance(env.env, VideoRecordingWrapper)
                    env.env.video_recoder.stop()
                    env.env.file_path = None
                    if enable_render:
                        filename = pathlib.Path(output_dir).joinpath(
                            'media', wv.util.generate_id() + ".mp4")
                        filename.parent.mkdir(parents=False, exist_ok=True)
                        filename = str(filename)
                        env.env.file_path = filename

                    # switch to init_state reset
                    assert isinstance(env.env.env, RobomimicImageWrapper)
                    env.env.env.init_state = init_state

                env_seeds.append(train_idx)
                env_prefixs.append('train/')
                env_init_fn_dills.append(dill.dumps(init_fn))
        
        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, 
                enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # switch to seed reset
                assert isinstance(env.env.env, RobomimicImageWrapper)
                env.env.env.init_state = None
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns, dummy_env_fn=dummy_env_fn)
        # env = SyncVectorEnv(env_fns)


        self.env_meta = env_meta
        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.rotation_transformer = rotation_transformer
        self.abs_action = abs_action
        self.tqdm_interval_sec = tqdm_interval_sec
        self.debug_save_actions = debug_save_actions
        self.debug_save_details = debug_save_details

    def run(self, policy: BaseImagePolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env
        
        # 一次性检查并提示 policy 与 runner 的时序设置
        pol_n_act = getattr(policy, 'n_action_steps', None)
        pol_n_obs = getattr(policy, 'n_obs_steps', None)
        if (pol_n_act is not None) and (pol_n_act != self.n_action_steps):
            print(f"[warn] policy.n_action_steps={pol_n_act} != runner.n_action_steps={self.n_action_steps}. Will align actions by tiling/truncation.")
        if (pol_n_obs is not None) and (pol_n_obs != self.n_obs_steps):
            print(f"[warn] policy.n_obs_steps={pol_n_obs} != runner.n_obs_steps={self.n_obs_steps}. Ensure config alignment to avoid conditioning mismatch.")
        print(f"[info] abs_action={self.abs_action}, max_steps={self.max_steps}, n_envs={len(self.env_fns)}") 

        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()

            env_name = self.env_meta['env_name']
            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval {env_name}Image {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            
            done = False
            warned_action_len = False
            # 调试：按chunk收集每步准备送入环境的动作
            if self.debug_save_actions:
                print("[debug] rollout action recording is enabled; artifacts will be saved under 'rollout_debug'.")
                chunk_actions = []  # list of np.ndarray [n_envs, n_action_steps, act_dim]
                l2_curve = []      # list of scalar（跨env和步长平均的范数）
            if self.debug_save_details:
                # 更细的 per-env 指标（只记录每轮的第1子步，便于定位）
                firststep_l2_model = []      # [T, n_envs]
                firststep_l2_env = []        # [T, n_envs]
                firststep_l2_model_pos = []  # [T, n_envs]
                firststep_l2_env_pos = []    # [T, n_envs]
                firststep_l2_model_rot = []  # [T, n_envs]
                firststep_l2_env_rot = []    # [T, n_envs]
                firststep_l2_model_grp = []  # [T, n_envs]
                firststep_l2_env_grp = []    # [T, n_envs]
                obs_mean_per_env = []        # [T, n_envs]
            while not done:
                # create obs dict
                np_obs_dict = dict(obs)
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))

                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action']
                # 对齐 runner 期望的 n_action_steps（如不一致，进行平铺或截断）
                if action.shape[1] != self.n_action_steps:
                    print("对齐 runner 期望的 n_action_steps（如不一致，进行平铺或截断）")
                    if not warned_action_len:
                        print(f"[warn] action length from policy ({action.shape[1]}) != runner.n_action_steps ({self.n_action_steps}). Will adjust.")
                        warned_action_len = True
                    if action.shape[1] < self.n_action_steps:
                        reps = int(math.ceil(self.n_action_steps / action.shape[1]))
                        action = np.tile(action, (1, reps, 1))[:, :self.n_action_steps, :]
                    else:
                        action = action[:, :self.n_action_steps, :]

                if self.debug_save_details:
                    try:
                        # 模型输出（对齐步长后）的第1子步 L2（每环境）及分量
                        l2_model = np.linalg.norm(action[:, 0, :], axis=-1)
                        D = action.shape[-1]
                        # 约定：pos(0:3), rot(3:D-1) if D>4, gripper(-1)
                        pos_model = np.linalg.norm(action[:, 0, :3], axis=-1)
                        grp_model = np.linalg.norm(action[:, 0, -1:], axis=-1)
                        if D > 4:
                            rot_model = np.linalg.norm(action[:, 0, 3:-1], axis=-1)
                        else:
                            rot_model = np.zeros_like(pos_model)
                        firststep_l2_model.append(l2_model.copy())
                        firststep_l2_model_pos.append(pos_model.copy())
                        firststep_l2_model_rot.append(rot_model.copy())
                        firststep_l2_model_grp.append(grp_model.copy())
                    except Exception:
                        pass

                # 简单的动作范数诊断（batch 首个样本）
                try:
                    l2 = np.linalg.norm(action[0], axis=-1)
                    if np.all(l2 < 1e-4):
                        print("[warn] Predicted action L2 is near zero across steps; policy may be outputting (almost) zeros.")
                except Exception:
                    pass
                if not np.all(np.isfinite(action)):
                    print(action)
                    raise RuntimeError("Nan or Inf action")
                
                # step env
                env_action = action
                if self.abs_action:
                    env_action = self.undo_transform_action(action)

                # 调试：保存准备送入环境的动作（记录实际将送入环境的 env_action）
                if self.debug_save_actions:
                    try:
                        chunk_actions.append(env_action.copy())
                        # 计算L2范数（先对最后一维求范数-> [n_envs, n_action_steps]，再整体平均到标量）
                        l2 = np.linalg.norm(env_action, axis=-1).mean()
                        l2_curve.append(float(l2))
                    except Exception as e:
                        print(f"[warn] failed to record env actions: {e}")

                if self.debug_save_details:
                    try:
                        # env_action 的第1子步 L2（每环境）及分量
                        l2_env = np.linalg.norm(env_action[:, 0, :], axis=-1)
                        D = env_action.shape[-1]
                        pos_env = np.linalg.norm(env_action[:, 0, :3], axis=-1)
                        grp_env = np.linalg.norm(env_action[:, 0, -1:], axis=-1)
                        if D > 4:
                            rot_env = np.linalg.norm(env_action[:, 0, 3:-1], axis=-1)
                        else:
                            rot_env = np.zeros_like(pos_env)
                        firststep_l2_env.append(l2_env.copy())
                        firststep_l2_env_pos.append(pos_env.copy())
                        firststep_l2_env_rot.append(rot_env.copy())
                        firststep_l2_env_grp.append(grp_env.copy())
                    except Exception:
                        pass
                    try:
                        key = self.env_meta.get('render_obs_key', None) or 'agentview_image'
                        if key in np_obs_dict:
                            img = np_obs_dict[key]
                            if img.ndim >= 3:
                                last = img[:, -1]
                                obs_mean = last.reshape(last.shape[0], -1).mean(axis=1)
                                obs_mean_per_env.append(obs_mean.copy())
                    except Exception:
                        pass

                obs, reward, done, info = env.step(env_action)
                done = np.all(done)
                past_action = action

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            # 调试：按chunk落盘动作序列与L2曲线图
            if self.debug_save_actions:
                try:
                    debug_dir = pathlib.Path(self.output_dir).joinpath('rollout_debug')
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    # 保存动作序列
                    # 形状: [T_chunk, n_envs, n_action_steps, act_dim]
                    if len(chunk_actions) > 0:
                        actions_arr = np.stack(chunk_actions, axis=0)
                        np.save(str(debug_dir.joinpath(f'chunk_{chunk_idx+1:02d}_actions.npy')), actions_arr)
                        # 保存L2曲线图
                        if plt is not None:
                            plt.figure()
                            plt.plot(l2_curve, label='mean L2 over envs/steps')
                            plt.title(f"chunk {chunk_idx+1}/{n_chunks}")
                            plt.xlabel('control loop step')
                            plt.ylabel('L2(action)')
                            plt.grid(True, alpha=0.3)
                            plt.legend()
                            plt.savefig(str(debug_dir.joinpath(f'chunk_{chunk_idx+1:02d}_action_l2.png')), dpi=150, bbox_inches='tight')
                            plt.close()
                except Exception as e:
                    print(f"[warn] failed to save rollout debug artifacts: {e}")

            if self.debug_save_details:
                try:
                    debug_dir = pathlib.Path(self.output_dir).joinpath('rollout_debug')
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    def _maybe_save(name, lst):
                        if len(lst) > 0:
                            arr = np.stack(lst, axis=0)
                            np.save(str(debug_dir.joinpath(f'chunk_{chunk_idx+1:02d}_{name}.npy')), arr)
                    _maybe_save('firststep_l2_model', firststep_l2_model)
                    _maybe_save('firststep_l2_env', firststep_l2_env)
                    _maybe_save('firststep_l2_model_pos', firststep_l2_model_pos)
                    _maybe_save('firststep_l2_env_pos', firststep_l2_env_pos)
                    _maybe_save('firststep_l2_model_rot', firststep_l2_model_rot)
                    _maybe_save('firststep_l2_env_rot', firststep_l2_env_rot)
                    _maybe_save('firststep_l2_model_grp', firststep_l2_model_grp)
                    _maybe_save('firststep_l2_env_grp', firststep_l2_env_grp)
                    _maybe_save('obs_mean_agentview_image', obs_mean_per_env)
                except Exception as e:
                    print(f"[warn] failed to save per-env debug metrics: {e}")

            # collect data for this round，但是无法收集完整的奖励
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
        # clear out video buffer
        _ = env.reset()
        
        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video
        
        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data


    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1,2,10)

        d_rot = action.shape[-1] - 4
        pos = action[...,:3]
        rot = action[...,3:3+d_rot]
        gripper = action[...,[-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([
            pos, rot, gripper
        ], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction
