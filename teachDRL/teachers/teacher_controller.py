import numpy as np
import pickle
import copy
from teachDRL.teachers.algos.riac import RIAC
from teachDRL.teachers.algos.alp_gmm import ALPGMM
from teachDRL.teachers.algos.covar_gmm import CovarGMM
from teachDRL.teachers.algos.random_teacher import RandomTeacher
from teachDRL.teachers.algos.oracle_teacher import OracleTeacher
from teachDRL.teachers.algos.lp_ucb import LP_UCB
from teachDRL.teachers.algos.reward_ucb import REWARD_UCB
from teachDRL.teachers.utils.test_utils import get_test_set_name
from collections import OrderedDict

def param_vec_to_param_dict(param_env_bounds, param):
    param_dict = OrderedDict()
    cpt = 0
    for i,(name, bounds) in enumerate(param_env_bounds.items()):
        if len(bounds) == 2:
            param_dict[name] = param[i]
            cpt += 1
        elif len(bounds) == 3:  # third value is the number of dimensions having these bounds
            nb_dims = bounds[2] + 1
            param_dict[name] = param[i:i+nb_dims]
            cpt += nb_dims
    #print('reconstructed param vector {}\n into {}'.format(param, param_dict)) #todo remove
    return param_dict

def param_dict_to_param_vec(param_env_bounds, param_dict):  # needs param_env_bounds for order reference
    param_vec = []
    for name, bounds in param_env_bounds.items():
        #print(param_dict[name])
        param_vec.append(param_dict[name])
    return np.array(param_vec, dtype=np.float32)



class TeacherController(object):
    def __init__(self, teacher, nb_test_episodes, param_env_bounds, env_name, seed=None, teacher_params={}):
        self.teacher = teacher
        self.nb_test_episodes = nb_test_episodes
        self.test_ep_counter = 0
        self.eps= 1e-03
        self.param_env_bounds = copy.deepcopy(param_env_bounds)
        self.env_name = env_name
        self.reward_scale_mode = teacher_params['reward_scale_mode']
        # figure out parameters boundaries vectors
        mins, maxs = [], []
        for name, bounds in param_env_bounds.items():
            if "component" in name:
                continue
            if len(bounds) == 2:
                mins.append(bounds[0])
                maxs.append(bounds[1])
            elif len(bounds) == 3:  # third value is the number of dimensions having these bounds
                mins.extend([bounds[0]] * bounds[2])
                maxs.extend([bounds[1]] * bounds[2])
            else:
                print("ill defined boundaries, use [min, max, nb_dims] format or [min, max] if nb_dims=1")
                exit(1)

        # setup tasks generator
        if teacher == 'Oracle':
            self.task_generator = OracleTeacher(mins, maxs, teacher_params['window_step_vector'], seed=seed)
        elif teacher == 'Random':
            self.task_generator = RandomTeacher(mins, maxs, seed=seed)
        elif teacher == 'RIAC':
            self.task_generator = RIAC(mins, maxs, seed=seed, params=teacher_params)
        elif teacher == 'ALP-GMM':
            self.task_generator = ALPGMM(mins, maxs, seed=seed, params=teacher_params)
        elif teacher == 'Covar-GMM':
            self.task_generator = CovarGMM(mins, maxs, seed=seed, params=teacher_params)
        elif teacher == 'LP-UCB':
            self.task_generator = LP_UCB(mins, maxs, seed=seed, params=teacher_params)
        elif teacher == 'REWARD-UCB':
            self.task_generator = REWARD_UCB(mins, maxs, seed=seed, params=teacher_params)
        else:
            print('Unknown teacher')
            raise NotImplementedError

        if self.env_name != "air_hockey":
            self.test_mode = "fixed_set"
            if self.test_mode == "fixed_set":
                name = get_test_set_name(self.param_env_bounds)
                self.test_env_list = pickle.load( open("teachDRL/teachers/test_sets/"+name+".pkl", "rb" ) )
                print('fixed set of {} tasks loaded: {}'.format(len(self.test_env_list),name))
        else:
            self.test_mode = "target" # test on what? #TODO
        #data recording
        self.env_params_train = []
        self.env_train_rewards = []
        self.env_train_norm_rewards = []
        self.env_train_len = []

        self.env_params_test = []
        self.env_test_rewards = []
        self.env_test_len = []

    def record_train_episode(self, reward, ep_len):
        self.env_train_rewards.append(reward)
        self.env_train_len.append(ep_len)
        if self.reward_scale_mode == 'log':
            if reward < 0:
                reward = -np.log1p(-reward)
            else:
                reward = np.log1p(reward)
            reward = np.interp(reward, (-np.log(300), np.log(2000)), (0, 1))
            self.env_train_norm_rewards.append(reward)
        elif self.reward_scale_mode == 'linear':
            if self.teacher != 'Oracle':
                reward = np.interp(reward, (-300, 2000), (0, 1))
                self.env_train_norm_rewards.append(reward)
        elif self.reward_scale_mode == 'no_scale':
            self.env_train_norm_rewards.append(reward)
        else:
            print("not implemented yet")
            exit(10)
        self.task_generator.update(self.env_params_train[-1], reward)

    def record_test_episode(self, reward, ep_len):
        self.env_test_rewards.append(reward)
        self.env_test_len.append(ep_len)

    def dump(self, filename):
        with open(filename, 'wb') as handle:
            dump_dict = {'env_params_train': self.env_params_train,
                         'env_train_rewards': self.env_train_rewards,
                         'env_train_len': self.env_train_len,
                         'env_params_test': self.env_params_test,
                         'env_test_rewards': self.env_test_rewards,
                         'env_test_len': self.env_test_len,
                         'env_param_bounds': list(self.param_env_bounds.items())}
            dump_dict = self.task_generator.dump(dump_dict)
            pickle.dump(dump_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def hyperspherical_to_cartesian(self, angles):
        """
        Convert hyperspherical coordinates to Cartesian coordinates.

        Parameters:
            r (float): Radius of the hypersphere.
            angles (list): List of angles corresponding to each dimension.

        Returns:
            numpy.array: Cartesian coordinates.
        """
        r = 1
        dimensions = len(angles) + 1
        cartesian_coords = [r * np.prod(np.sin(angles[:i])) * np.cos(angles[i]) for i in range(dimensions - 1)]
        cartesian_coords.append(r * np.prod(np.sin(angles[:dimensions - 1])))
        return np.array(cartesian_coords,dtype=np.float32)
    def set_env_params(self):
        params_spheral = copy.copy(self.task_generator.sample_task())
        params = self.hyperspherical_to_cartesian(params_spheral)
        assert type(params[0]) == np.float32
        self.env_params_train.append(params_spheral)
        param_dict = param_vec_to_param_dict(self.param_env_bounds, params)
        return param_dict

    def set_test_env_params(self, test_env):
        self.test_ep_counter += 1
        if self.test_mode == "fixed_set":
            test_param_dict = self.test_env_list[self.test_ep_counter-1]

            # removing legacy parameters from test_set, don't pay attention
            legacy = ['tunnel_height', 'gap_width', 'step_height', 'step_number']
            keys = test_param_dict.keys()
            for env_param in legacy:
                if env_param in keys:
                    del test_param_dict[env_param]

        elif self.test_mode == "target":
            # test on target task
            params_vec = np.array([0.0, 0.0, 0.0, 1.0])
            test_param_dict = param_vec_to_param_dict(self.param_env_bounds, params_vec)
        else:
            raise NotImplementedError

        #print('test param dict is: {}'.format(test_param_dict))
        test_param_vec = param_dict_to_param_vec(self.param_env_bounds, test_param_dict)
        #print('test param vector is: {}'.format(test_param_vec))

        self.env_params_test.append(test_param_vec)
        test_env.wrapped_env.set_environment(**test_param_dict)

        if self.test_ep_counter == self.nb_test_episodes:
            self.test_ep_counter = 0