from sklearn.mixture import GaussianMixture as GMM
import numpy as np
from gym.spaces import Box
from teachDRL.teachers.utils.dataset import BufferedDataset
import pandas as pd
import tensorflow as tf
import scipy

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
        self.S_star_buf = np.zeros([size, C_dim], dtype=np.float32)
        self.S_buf = np.zeros([size, C_dim], dtype=np.float32)
        self.S_multiply_grad_log_p_star_buf = np.zeros([size, C_dim], dtype=np.float32)
        self.S_multiply_grad_log_p_buf = np.zeros([size, C_dim], dtype=np.float32)

        self.average_out_return_buf = np.zeros(size, dtype=np.float32)
        self.W_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.number_C = C_dim

    def store(self, C, average_outed_return, averaged_out_S, averaged_out_S_multiply_grad_log_p):
        self.C_buf[self.ptr] = C
        self.Values_buf[self.ptr] = (averaged_out_S, averaged_out_S_multiply_grad_log_p)

        # creating s_star which is actually multiplying s with c_star which is [0,0, ... , 0 , 1], means last element
        # for having s_star as array for future use, we put it in zeros array so we have [0,0, ..., 0, s_star]
        self.S_star_buf[self.ptr][-1] = np.sum(averaged_out_S[:, -1])

        # the same for s * grad(log(p))
        self.S_multiply_grad_log_p_star_buf[self.ptr][-1] = np.sum(averaged_out_S_multiply_grad_log_p[:, -1])

        # Calculating the S_buf which is multiplication of S into its own C.
        self.S_buf[self.ptr] = np.sum(np.multiply(C, averaged_out_S), axis=0)

        # The same for c * s * grad(log(p))
        self.S_multiply_grad_log_p_buf[self.ptr] = np.sum(np.multiply(C, averaged_out_S_multiply_grad_log_p), axis=0)

        self.average_out_return_buf[self.ptr] = average_outed_return
        self.W_buf[self.ptr] = None
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

def proportional_choice(v, eps=0.):
    if np.sum(v) == 0 or np.random.rand() < eps:
        return np.random.randint(np.size(v))
    else:
        probas = np.array(v) / np.sum(v)
        return np.where(np.random.multinomial(1, probas) == 1)[0][0]


# Absolute Learning Progress (ALP) computer object
# It uses a buffered kd-tree to efficiently implement a k-nearest-neighbor algorithm
class EmpiricalALPLearningComputer():
    def __init__(self, beta, dataset, n_C, alpha):
        self.alp_knn = dataset
        self.beta = beta
        self.number_C = n_C
        self.alpha = alpha
        # Creating the computational graph for computing gradients
        self.task_tensor_placeholder = tf.placeholder(dtype=tf.float32, shape=(self.number_C,))
        self.C_buf_tensor_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, self.number_C))
        self.reward_placeholder = tf.placeholder(dtype=tf.float32, shape=(None,))

        with tf.GradientTape() as tape:
            tape.watch([self.task_tensor_placeholder, self.C_buf_tensor_placeholder, self.reward_placeholder])

            # 1 - Compute distance between current C, (task), and all other C's
            distance = tf.norm(self.task_tensor_placeholder - self.C_buf_tensor_placeholder, axis=1)

            # 2 - Compute a weight for each C in dataset, here is the calculation of soft-argmin
            a = tf.exp(-self.beta * distance)
            b = tf.reduce_sum(tf.exp(-self.beta * distance))
            self.weights = a / b

            # 3 - Compute the C' as weighted average of some C.
            C_prime = tf.squeeze(tf.matmul(tf.expand_dims(self.weights, axis=0), self.C_buf_tensor_placeholder))

            # 4 - Compute the reward of the C' found, which is weighted average of some C.
            # TODO compute reward but interpolated into 0-1,
            self.c_prime_reward = tf.math.reduce_sum(tf.math.multiply(self.weights,
                                                                      self.reward_placeholder))
        self.grad = tape.gradient(C_prime, self.task_tensor_placeholder)
        # self.grad = tape.gradient(self.weights, a)

    def compute_alp(self, task, reward):
        alp = 0
        grad_alp = 0
        if self.alp_knn.size > 5:
            # Compute absolute learning progress for new task

            indices_including_task = np.linalg.norm(self.alp_knn.C_buf, axis=1) != 0
            indices = np.logical_and(np.linalg.norm(task - self.alp_knn.C_buf, axis=1) != 0, indices_including_task)
            index_current_C = np.logical_xor(indices, indices_including_task)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                feed_dict = {
                    self.task_tensor_placeholder: task,
                    self.C_buf_tensor_placeholder: self.alp_knn.C_buf[indices],
                    self.reward_placeholder: self.alp_knn.average_out_return_buf[indices]
                }

                # Compute Gradient of C' with respect to C using the computational graph
                gradCprime_wrt_C, weights, c_prime_reward = sess.run(
                    [self.grad, self.weights, self.c_prime_reward], feed_dict=feed_dict)

            self.alp_knn.W_buf[indices] = weights

            # 1 - Compute alp as absolute difference in reward for shaped system and sparse system
            lp1 = reward - c_prime_reward

            # Calculating the reward with C*, which means taking only the last element of reward feature(S)
            reward_c_star_task = np.sum(self.alp_knn.S_star_buf[index_current_C])
            reward_c_star_c_prime = np.sum(self.alp_knn.W_buf[indices][np.newaxis, :] @ self.alp_knn.S_star_buf[indices])

            lp2 = reward_c_star_task - reward_c_star_c_prime
            alp = self.alpha * (np.abs(lp1)) + (1 - self.alpha) * np.abs(lp2)

            # 2 - compute grad ALP with respect to C, which is task here.

            # 2.1 - calculate the first element of first part of derivation
            s_multiply_grad_p_pi = self.alp_knn.S_multiply_grad_log_p_buf[index_current_C]
            # 2.2 - calculate the second element of first part of derivation
            tmp = np.multiply(self.alp_knn.C_buf[indices], self.alp_knn.S_multiply_grad_log_p_buf[indices])
            tmp2 = self.alp_knn.W_buf[indices][np.newaxis, :] @ tmp
            c_prime_s_multiply_grad_p_pi_grad_c_wrt_c_prime = np.multiply(np.squeeze(tmp2), gradCprime_wrt_C)

            grad_alp_part1 = s_multiply_grad_p_pi - c_prime_s_multiply_grad_p_pi_grad_c_wrt_c_prime
            # 2.3 - calculate the first element of second part of derivation

            # For creating an array which has the following shape [0, 0, ..., 0, 1]
            c_star = np.concatenate([np.zeros(self.number_C - 1), np.array([1.0])])

            c_star_s_multiply_grad_p_pi = np.multiply(c_star,self.alp_knn.S_multiply_grad_log_p_buf[index_current_C])

            # 2.4 - Calculating the second element of second part of derivative
            c_star_s_multiply_grad_p_pi_grad_c_wrt_c_prime = np.multiply(c_star,
                                                                         c_prime_s_multiply_grad_p_pi_grad_c_wrt_c_prime)
            grad_alp_part2 = c_star_s_multiply_grad_p_pi - c_star_s_multiply_grad_p_pi_grad_c_wrt_c_prime

            grad_alp = self.alpha * np.abs(grad_alp_part1) + (1 - self.alpha) * np.abs(grad_alp_part2)

            # c' zarb dar s_multi_grad_p_pi and zarb grad c' wrt c
        # Add to database
        # self.alp_knn.add_xy(reward, task)
        return alp, grad_alp



# Absolute Learning Progress - Gaussian Mixture Model
# mins / maxs are vectors defining task space boundaries (ex: mins=[0,0,0] maxs=[1,1,1])
class ALPLearningGMM():
    def __init__(self, mins, maxs, beta, alpha, n_c_updates, step_size,  seed=None, params=dict()):
        self.seed = seed
        if not seed:
            self.seed = np.random.randint(42, 424242)
        np.random.seed(self.seed)

        # Task space boundaries
        self.mins = np.array(mins)
        self.maxs = np.array(maxs)

        # Range of number of Gaussians to try when fitting the GMM
        self.potential_ks = np.arange(2, 11, 1) if "potential_ks" not in params else params["potential_ks"]
        # Restart new fit by initializing with last fit
        self.warm_start = False if "warm_start" not in params else params["warm_start"]
        # Fitness criterion when selecting best GMM among range of GMMs varying in number of Gaussians.
        self.gmm_fitness_fun = "aic" if "gmm_fitness_fun" not in params else params["gmm_fitness_fun"]
        # Number of Expectation-Maximization trials when fitting
        self.nb_em_init = 1 if "nb_em_init" not in params else params['nb_em_init']
        # Number of episodes between two fit of the GMM
        self.fit_rate = 250 if "fit_rate" not in params else params['fit_rate']
        self.nb_random = self.fit_rate  # Number of bootstrapping episodes

        # Ratio of randomly sampled tasks VS tasks sampling using GMM
        self.random_task_ratio = 0.2 if "random_task_ratio" not in params else params["random_task_ratio"]
        self.random_task_generator = Box(self.mins, self.maxs, dtype=np.float32)

        # Maximal number of episodes to account for when computing ALP
        alp_max_size = None if "alp_max_size" not in params else params["alp_max_size"]
        alp_buffer_size = 500 if "alp_buffer_size" not in params else params["alp_buffer_size"]

        self.tasks = []
        self.alps = []
        self.tasks_alps = []
        self.grad_alps = []

        # Init GMMs
        self.potential_gmms = [self.init_gmm(k) for k in self.potential_ks]

        # Boring book-keeping
        self.bk = {'weights': [], 'covariances': [], 'means': [], 'tasks_alps': [], 'episodes': []}

        """ A dataset to save the components needed for caluculating the ALP Learning GMM and its derivations"""

        self.number_C = 4
        self.dataset_alps = CDataset(self.number_C, alp_buffer_size)
        self.n_c_updates = n_c_updates
        self.GMM_or_Learning = 'GMM'
        self.step_size = step_size
        # Init ALP computer
        self.alp_computer = EmpiricalALPLearningComputer(beta=beta, dataset=self.dataset_alps, n_C=self.number_C,
                                                         alpha=alpha)

    def init_gmm(self, nb_gaussians):
        return GMM(n_components=nb_gaussians, covariance_type='full', random_state=self.seed,
                   warm_start=self.warm_start, n_init=self.nb_em_init)

    def get_nb_gmm_params(self, gmm):
        # assumes full covariance
        # see https://stats.stackexchange.com/questions/229293/the-number-of-parameters-in-gaussian-mixture-model
        nb_gmms = gmm.get_params()['n_components']
        d = len(self.mins)
        params_per_gmm = (d * d - d) / 2 + 2 * d + 1
        return nb_gmms * params_per_gmm - 1

    def update(self, task, reward):
        self.tasks.append(task)
        # TODO inja vaghti reward miad normal shode tavasot teacher_controller
        # Compute corresponding ALP
        alp, grad_alp = self.alp_computer.compute_alp(task, reward)
        self.alps.append(alp)
        self.grad_alps.append(grad_alp)
        # Concatenate task vector with ALP dimension
        self.tasks_alps.append(np.array(task.tolist() + [self.alps[-1]]))

        if len(self.tasks) >= self.nb_random:  # If initial bootstrapping is done
            if (len(self.tasks) % self.fit_rate) == 0:  # Time to fit
                # 1 - Retrieve last <fit_rate> (task, reward) pairs
                cur_tasks_alps = np.array(self.tasks_alps[-self.fit_rate:])

                # 2 - Fit batch of GMMs with varying number of Gaussians
                self.potential_gmms = [g.fit(cur_tasks_alps) for g in self.potential_gmms]

                # 3 - Compute fitness and keep best GMM
                fitnesses = []
                if self.gmm_fitness_fun == 'bic':  # Bayesian Information Criterion
                    fitnesses = [m.bic(cur_tasks_alps) for m in self.potential_gmms]
                elif self.gmm_fitness_fun == 'aic':  # Akaike Information Criterion
                    fitnesses = [m.aic(cur_tasks_alps) for m in self.potential_gmms]
                elif self.gmm_fitness_fun == 'aicc':  # Modified AIC
                    n = self.fit_rate
                    fitnesses = []
                    for l, m in enumerate(self.potential_gmms):
                        k = self.get_nb_gmm_params(m)
                        penalty = (2 * k * (k + 1)) / (n - k - 1)
                        fitnesses.append(m.aic(cur_tasks_alps) + penalty)
                else:
                    raise NotImplementedError
                    exit(1)
                self.gmm = self.potential_gmms[np.argmin(fitnesses)]

                # book-keeping
                self.bk['weights'].append(self.gmm.weights_.copy())
                self.bk['covariances'].append(self.gmm.covariances_.copy())
                self.bk['means'].append(self.gmm.means_.copy())
                self.bk['tasks_alps'] = self.tasks_alps
                self.bk['episodes'].append(len(self.tasks))

    def sample_task(self): # TODO sample bardari bayad ba yek arg injuri beshe ke har bar ke az GMM migirim, chand bar az learning begirim??
        if (len(self.tasks) < self.nb_random) or (np.random.random() < self.random_task_ratio):
            # Random task sampling
            new_task = self.random_task_generator.sample()
            new_task = np.float32(scipy.special.softmax(new_task))
        elif self.GMM_or_Learning == 'GMM':
            # ALP-based task sampling

            # 1 - Retrieve the mean ALP value of each Gaussian in the GMM
            self.alp_means = []
            for pos, _, w in zip(self.gmm.means_, self.gmm.covariances_, self.gmm.weights_):
                self.alp_means.append(pos[-1])

            # 2 - Sample Gaussian proportionally to its mean ALP
            idx = proportional_choice(self.alp_means, eps=0.0)

            # 3 - Sample task in Gaussian, without forgetting to remove ALP dimension
            new_task = np.random.multivariate_normal(self.gmm.means_[idx], self.gmm.covariances_[idx])[:-1]

            #new_task = np.clip(new_task, self.mins, self.maxs).astype(np.float32)
            # instead of clipping, we do the softmax to take everything between 0 and 1 with the sum to 1.
            new_task = np.float32(scipy.special.softmax(new_task))
            self.GMM_or_Learning = 'Learning'
        else:
            new_task = self.tasks[-1] + self.n_c_updates * self.step_size * self.grad_alps[-1]
            new_task = np.float32(scipy.special.softmax(new_task))
            self.GMM_or_Learning = 'GMM'
        return new_task

    def dump(self, dump_dict):
        dump_dict.update(self.bk)
        return dump_dict

    # def append_Dataset(self, C, S, Grad):
    #     new = pd.DataFrame({'C': [C], 'S': [S], 'S*Grad_log_pi': [Grad]})
    #     self.dataset = pd.concat([self.dataset, new])
