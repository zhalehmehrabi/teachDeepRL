import numpy as np
import pickle
import copy
from teachDRL.teachers.algos.riac import RIAC
from teachDRL.teachers.algos.alp_gmm import ALPGMM
from teachDRL.teachers.algos.alp_learning_gmm import ALPLearningGMM
from teachDRL.teachers.algos.alp_gmm_new_formula import ALPGMMNewFormula
from teachDRL.teachers.algos.covar_gmm import CovarGMM
from teachDRL.teachers.algos.random_teacher import RandomTeacher
from teachDRL.teachers.algos.oracle_teacher import OracleTeacher
from teachDRL.teachers.utils.test_utils import get_test_set_name
from collections import OrderedDict


class CDataset:
    """
    A simple FIFO experience buffer for teacher.

    C_buf is used to store the values of coefficients.

    Value_buf is used to store a list which the first element is the reward feature S, and the second element is the
    S * gradient of log probability of policy

    W_buf is used to store the values of weights which will be caluculated for soft argmin.
    """

    def __init__(self, C_dim, size):
        self.C_buf = np.zeros([size, C_dim], dtype=np.float32)
        self.Values_buf = [([], []) for _ in range(size)]
        self.W_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, C, S, S_multiple_grad_log_p):
        self.C_buf[self.ptr] = C
        self.Values_buf[self.ptr] = (S, S_multiple_grad_log_p)
        self.W_buf[self.ptr] = None
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def CalculateSoftArgMin(self, newC):
        print("calculations")

def param_vec_to_param_dict(param_env_bounds, param):
    param_dict = OrderedDict()
    cpt = 0
    for i, (name, bounds) in enumerate(param_env_bounds.items()):
        if len(bounds) == 2:
            param_dict[name] = param[i]
            cpt += 1
        elif len(bounds) == 3:  # third value is the number of dimensions having these bounds
            nb_dims = bounds[2]
            param_dict[name] = param[i:i + nb_dims]
            cpt += nb_dims
    # print('reconstructed param vector {}\n into {}'.format(param, param_dict)) #todo remove
    return param_dict


def param_dict_to_param_vec(param_env_bounds, param_dict):  # needs param_env_bounds for order reference
    param_vec = []
    for name, bounds in param_env_bounds.items():
        # print(param_dict[name])
        param_vec.append(param_dict[name])
    return np.array(param_vec, dtype=np.float32)


class TeacherController(object):
    def __init__(self, teacher, nb_test_episodes, param_env_bounds, alpha, n_c_updates, step_size,learning_radio,
                 beta=None, seed=None, new_formula=False, teacher_params={}):
        self.teacher = teacher
        self.nb_test_episodes = nb_test_episodes
        self.test_ep_counter = 0
        self.eps = 1e-03
        self.param_env_bounds = copy.deepcopy(param_env_bounds)

        # figure out parameters boundaries vectors
        mins, maxs = [], []
        for name, bounds in param_env_bounds.items():
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
        elif teacher == 'ALP-GMM' and not new_formula:
            self.task_generator = ALPGMM(mins, maxs, seed=seed, params=teacher_params)
        elif teacher == 'ALP-GMM' and new_formula:
            self.task_generator = ALPGMMNewFormula(mins, maxs, beta=beta, seed=seed, params=teacher_params, alpha=alpha,
                                                 n_c_updates=n_c_updates, step_size=step_size,
                                                 learning_radio=learning_radio)
        elif teacher == 'ALP-Learning-GMM':
            self.task_generator = ALPLearningGMM(mins, maxs, beta=beta, seed=seed, params=teacher_params, alpha=alpha,
                                                 n_c_updates=n_c_updates, step_size=step_size,
                                                 learning_radio=learning_radio)
        elif teacher == 'Covar-GMM':
            self.task_generator = CovarGMM(mins, maxs, seed=seed, params=teacher_params)
        else:
            print('Unknown teacher')
            raise NotImplementedError

        self.test_mode = "Target_task"
        if self.test_mode == "fixed_set":
            name = get_test_set_name(self.param_env_bounds)
            self.test_env_list = pickle.load(open("teachDRL/teachers/test_sets/" + name + ".pkl", "rb"))
            print('fixed set of {} tasks loaded: {}'.format(len(self.test_env_list), name))

        # data recording
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
        if self.teacher != 'Oracle' and self.teacher != 'ALP-Learning-GMM':
            reward = np.interp(reward, (-150, 350), (0, 1))
            self.env_train_norm_rewards.append(reward)

    def update_episodes(self, reward):
        self.task_generator.update(self.env_params_train[-1], reward)

    def record_test_episode(self, reward, ep_len):
        self.env_test_rewards.append(reward)
        self.env_test_len.append(ep_len)

    def record_gradiend_components(self, C, S, Grad):
        self.teacher.append_Dataset(C, S, Grad)

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

    def set_env_params(self, env):
        params = np.float32(copy.copy(self.task_generator.sample_task()))
        assert type(params[0]) == np.float32 # FIXME ino bastam ta bbinam chi mishe
        self.env_params_train.append(params)
        param_dict = param_vec_to_param_dict(self.param_env_bounds, params)
        env.env.set_environment(**param_dict)
        return params

    def set_test_env_params(self, test_env):
        self.test_ep_counter += 1
        if self.test_mode == "fixed_set":
            test_param_dict = self.test_env_list[self.test_ep_counter - 1]

            # removing legacy parameters from test_set, don't pay attention
            legacy = ['tunnel_height', 'gap_width', 'step_height', 'step_number']
            keys = test_param_dict.keys()
            for env_param in legacy:
                if env_param in keys:
                    del test_param_dict[env_param]
        elif self.test_mode == "Target_task":
            test_params = np.concatenate([np.zeros(test_env.env.number_C - 1), np.ones(1)])
            test_param_dict = param_vec_to_param_dict(self.param_env_bounds, test_params)
        else:
            raise NotImplementedError

        # print('test param dict is: {}'.format(test_param_dict))
        test_param_vec = param_dict_to_param_vec(self.param_env_bounds, test_param_dict)
        # print('test param vector is: {}'.format(test_param_vec))

        self.env_params_test.append(test_param_vec)
        test_env.env.set_environment(**test_param_dict)

        if self.test_ep_counter == self.nb_test_episodes:
            self.test_ep_counter = 0

    def record_grads(self, C, average_return, S, s_multiplied_grad_log_p_pi):
        self.task_generator.dataset_alps.store(C, average_return,S, s_multiplied_grad_log_p_pi)
