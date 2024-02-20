import argparse

import hydra

# from teachDRL.spinup.screipt_utils.run_utils import setup_logger_kwargs
# from teachDRL.spinup.algos.sac.sac import sac
# from teachDRL.spinup.algos.sac import core
import gym
import teachDRL.gym_flowers
from teachDRL.teachers.teacher_controller import TeacherController
from collections import OrderedDict
import os
import numpy as np
from my_scripts.screipt_utils import create_log_directory, get_callbacks
import random
from omegaconf import OmegaConf
from utils.env_utils import create_producer
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO, SAC, DQN
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm


alg_dict = {
    'sac': SAC,
    'ppo': PPO,
    'dqn': DQN,
}
# Argument definition
# parser = argparse.ArgumentParser()

# parser.add_argument('--exp_name', type=str, default='test')
# parser.add_argument('--seed', '-s', type=int, default=0)

# Deep RL student arguments, so far only works with SAC
# TODO solve this
# parser.add_argument('--hid', type=int, default=-1)  # number of neurons in hidden layers
# TODO solve this
# parser.add_argument('--l', type=int, default=1)  # number of hidden layers

# parser.add_argument('--gamma', type=float, default=0.99)
# parser.add_argument('--epochs', type=int, default=100)
# parser.add_argument('--gpu_id', type=int, default=-1)  # default is no GPU
# parser.add_argument('--ent_coef', type=float, default=0.005)
# parser.add_argument('--max_ep_len', type=int, default=2000)
# parser.add_argument('--steps_per_ep', type=int, default=200000)  # nb of env steps per epochs (stay above max_ep_len)
# parser.add_argument('--buf_size', type=int, default=2000000)
# parser.add_argument('--nb_test_episodes', type=int, default=50)
# parser.add_argument('--lr', type=float, default=1e-3)
# parser.add_argument('--train_freq', type=int, default=10)
# parser.add_argument('--batch_size', type=int, default=1000)

# Parameterized bipedal walker arguments, so far only works with bipedal-walker-continuous-v0
# parser.add_argument('--env', type=str, default="bipedal-walker-continuous-v0")

# Choose student (walker morphology)
# parser.add_argument('--leg_size', type=str, default="default")  # choose walker type ("short", "default" or "quadru")


# Selection of parameter space
# So far 3 choices: "--max_stump_h 3.0 --max_obstacle_spacing 6.0" (aka Stump Tracks) or "-hexa" (aka Hexagon Tracks)
# or "-seq" (untested experimental env)
# parser.add_argument('--max_stump_h', type=float, default=None)
# parser.add_argument('--max_stump_w', type=float, default=None)
# parser.add_argument('--max_stump_r', type=float, default=None)
# parser.add_argument('--roughness', type=float, default=None)
# parser.add_argument('--max_obstacle_spacing', type=float, default=None)
# parser.add_argument('--max_gap_w', type=float, default=None)
# parser.add_argument('--step_h', type=float, default=None)
# parser.add_argument('--step_nb', type=float, default=None)
# parser.add_argument('--hexa_shape', '-hexa', action='store_true')
# parser.add_argument('--stump_seq', '-seq', action='store_true')

# Reward coefficient arguments:
# parser.add_argument('--nb_reward_coeff', type=int, default=None)
# parser.add_argument('--init_reward_coeff_mode', type=str, default="target")  # choose walker type ("target" or "random")
# for now the reward components are [touch the puck, x-speed of the puck, y-speed of the puck, target] # TODO

# Teacher-specific arguments:
# parser.add_argument('--teacher', type=str, default="ALP-GMM")  # ALP-GMM, Covar-GMM, RIAC, Oracle, Random

# ALPGMM (Absolute Learning Progress - Gaussian Mixture Model) related arguments
# parser.add_argument('--gmm_fitness_fun', '-fit', type=str, default=None)
# parser.add_argument('--nb_em_init', type=int, default=None)
# parser.add_argument('--min_k', type=int, default=None)
# parser.add_argument('--max_k', type=int, default=None)
# parser.add_argument('--fit_rate', type=int, default=None)
# parser.add_argument('--weighted_gmm', '-wgmm', action='store_true')
# parser.add_argument('--alp_max_size', type=int, default=None)

# CovarGMM related arguments
# parser.add_argument('--absolute_lp', '-alp', action='store_true')

# RIAC related arguments
# parser.add_argument('--max_region_size', type=int, default=None)
# parser.add_argument('--alp_window_size', type=int, default=None)

# args = parser.parse_args()
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    if 'environment' not in cfg:
        print('Specify an environment')
        return

    env_args = cfg['environment']
    alg_args = cfg['algorithm']
    teacher_args = {key: value for key, value in cfg['teacher'].items() if value is not None}
    print("hey")
    seed = cfg['seed'] if cfg['seed'] and cfg['seed']!=0 else random.randint(0, 999999)

    log_dir = create_log_directory(cfg, seed)
    cfg['log_dir'] = log_dir

    tb_log_dir = os.path.join(log_dir, 'tb_logs')

    # logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    # Bind this run to specific GPU if there is one
    # if args.gpu_id != -1:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # TODO Set up Student's DeepNN architecture if provided
    # ac_kwargs = dict()
    # if args.hid != -1:
    #     ac_kwargs['hidden_sizes'] = [args.hid] * args.l

    # Set bounds for environment's parameter space format:[min, max, nb_dimensions] (if no nb_dimensions, assumes only 1)
    param_env_bounds = OrderedDict()
    # if args.max_stump_h is not None:
    #     param_env_bounds['stump_height'] = [0, args.max_stump_h]
    # if args.max_stump_w is not None:
    #     param_env_bounds['stump_width'] = [0, args.max_stump_w]
    # if args.max_stump_r is not None:
    #     param_env_bounds['stump_rot'] = [0, args.max_stump_r]
    # if args.max_obstacle_spacing is not None:
    #     param_env_bounds['obstacle_spacing'] = [0, args.max_obstacle_spacing]
    # if args.hexa_shape:
    #     param_env_bounds['poly_shape'] = [0, 4.0, 12]
    # if args.stump_seq:
    #     param_env_bounds['stump_seq'] = [0, 6.0, 10]
    if cfg["nb_reward_coeff"] is not None:
        param_env_bounds['reward_coefficients'] = [0, 1.0, cfg['nb_reward_coeff']]
    # Set Teacher hyperparameters
    params = {}

    # TODO define oracle right
    # if args.teacher == "Oracle":
    #     if 'stump_height' in param_env_bounds and 'obstacle_spacing' in param_env_bounds:
    #         params['window_step_vector'] = [0.1, -0.2]  # order must match param_env_bounds construction
    #     elif 'poly_shape' in param_env_bounds:
    #         params['window_step_vector'] = [0.1] * 12
    #         print('hih')
    #     elif 'stump_seq' in param_env_bounds:
    #         params['window_step_vector'] = [0.1] * 10
    #     else:
    #         print('Oracle not defined for this parameter space')
    #         exit(1)


    print(f'Configuration:\n {OmegaConf.to_yaml(cfg)}')

    # dumps configuration
    with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))

    env_producer = create_producer(seed=seed, **env_args)
    env = make_vec_env(env_producer,
                       seed=seed,
                       n_envs=cfg['parallel'],
                       vec_env_cls=SubprocVecEnv,
                       monitor_dir=log_dir,
                       )


    alg_cls = alg_dict[cfg.algorithm.alg]

    # remove alg name from arguments
    alg_args = {k:v for k,v in alg_args.items() if k != 'alg'}
    if 'train_freq' in alg_args:
        alg_args['train_freq'] = tuple(alg_args['train_freq']) # tuple is required and not list

    env_init = {}
    env_init['init_reward_coeff_mode'] = cfg["init_reward_coeff_mode"]

    teacher = TeacherController(teacher_args["alg"], cfg["nb_test_episodes"], param_env_bounds, cfg["env"],
                                seed=cfg["seed"], teacher_params=params)
    callback_list = get_callbacks(cfg, teacher, env_init)

    model = alg_cls(env=env, seed=seed, **alg_args)
    model.learn(**cfg.learn, tb_log_name=tb_log_dir, callback=callback_list)
    model.save(os.path.join(log_dir, "model.zip"))

    if isinstance(model, OffPolicyAlgorithm):
        model.save_replay_buffer(os.path.join(log_dir, "replay_buffer"))



    # cfg = compose(config_name="config.yaml")

    # Initialize teacher


    # Launch Student training
    sac(env_f, actor_critic=core.mlp_actor_critic, ac_kwargs=ac_kwargs, gamma=args.gamma, seed=args.seed,
        epochs=args.epochs,
        logger_kwargs=logger_kwargs, alpha=args.ent_coef, max_ep_len=args.max_ep_len, steps_per_epoch=args.steps_per_ep,
        replay_size=args.buf_size, env_init=env_init, env_name=args.env, nb_test_episodes=args.nb_test_episodes,
        lr=args.lr,
        train_freq=args.train_freq, batch_size=args.batch_size, Teacher=Teacher)


if __name__ == '__main__':
    main()
