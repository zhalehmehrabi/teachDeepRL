import numpy as np
from gym.spaces import Box
from teachDRL.teachers.utils.dataset import BufferedDataset
import itertools
from SMPyBandits.Policies import UCB
from randomist.algorithms.optimist import OPTIMIST


def proportional_choice(v, eps=0.):
    if np.sum(v) == 0 or np.random.rand() < eps:
        return np.random.randint(np.size(v))
    else:
        minimum = np.min(v)
        shifted_v = v - minimum
        probas = np.array(shifted_v) / np.sum(shifted_v)
        return np.where(np.random.multinomial(1, probas) == 1)[0][0]

# Absolute Learning Progress (ALP) computer object
# It uses a buffered kd-tree to efficiently implement a k-nearest-neighbor algorithm
class EmpiricalALPComputer():
    def __init__(self, task_size, max_size=None, buffer_size=500):
        self.alp_knn = BufferedDataset(1, task_size, buffer_size=buffer_size, lateness=0, max_size=max_size)

    def compute_alp(self, task, reward):
        alp = 0
        if len(self.alp_knn) > 5:
            # Compute absolute learning progress for new task
            
            # 1 - Retrieve closest previous task
            dist, idx = self.alp_knn.nn_y(task)
            
            # 2 - Retrieve corresponding reward
            closest_previous_task_reward = self.alp_knn.get_x(idx[0])

            # 3 - Compute alp as absolute difference in reward
            lp = reward - closest_previous_task_reward
            # alp = np.abs(lp)
            alp = lp # FIXME for now, I want teacher to work as LP

        # Add to database
        self.alp_knn.add_xy(reward, task)
        return alp

# Absolute Learning Progress - Gaussian Mixture Model
# mins / maxs are vectors defining task space boundaries (ex: mins=[0,0,0] maxs=[1,1,1])
class LP_OPTIMIST():
    def __init__(self, mins, maxs, seed=None, params=dict()):
        self.seed = seed
        if not seed:
            self.seed = np.random.randint(42,424242)
        np.random.seed(self.seed)

        # Task space boundaries
        self.mins = np.array(mins)
        self.maxs = np.array(maxs)

        self.nb_random = 10 if "nb_random" not in params else params['nb_random']

        # Ratio of randomly sampled tasks VS tasks sampling using GMM
        self.random_task_ratio = 0.2 if "random_task_ratio" not in params else params["random_task_ratio"]

        # Maximal number of episodes to account for when computing ALP
        alp_max_size = None if "alp_max_size" not in params else params["alp_max_size"]
        alp_buffer_size = 500 if "alp_buffer_size" not in params else params["alp_buffer_size"]

        # Init ALP computer
        self.alp_computer = EmpiricalALPComputer(len(mins), max_size=alp_max_size, buffer_size=alp_buffer_size)

        self.tasks = []
        self.alps = []
        self.tasks_alps = []

        # Boring book-keeping
        self.bk = {'weights': [], 'covariances': [], 'means': [], 'tasks_alps': [], 'episodes': []}

        # Define the number of thetas and discretization values
        # Number of thetas
        num_thetas = len(mins)

        # Number of discrete values for each theta
        m_discrete_values = 5 if "nb_discretization" not in params else params["nb_discretization"]

        # Discretize each theta into m values between min and max
        theta_values = np.linspace(self.mins[0], self.maxs[0], m_discrete_values)

        # Generate all combinations of theta values for the arms
        self.all_combinations = list(itertools.product(theta_values, repeat=num_thetas))
        num_arms = len(self.all_combinations)
        # if params["MAB_configuration"]["arm_type"] == "Bernoulli"
        self.policy = OPTIMIST(nbArms=num_arms)
        self.rewards = [[] for _ in range(num_arms)]
        self.chosen_arm = -1
    def update(self, task, reward):
        self.tasks.append(task)

        # Compute corresponding ALP
        self.alps.append(self.alp_computer.compute_alp(task, reward))


        # Update the associated arm
        if len(self.tasks) >= self.nb_random:
            self.policy.getReward(self.chosen_arm, self.alps[-1])
            self.rewards[self.chosen_arm].append(self.alps[-1])

    def sample_task(self):
        if (len(self.tasks) < self.nb_random) or (np.random.random() < self.random_task_ratio):
            # Randomly taking an arm
            self.chosen_arm = np.random.randint(len(self.all_combinations))
        else:
            # taking an arm
            self.chosen_arm = self.policy.choice()
            print(self.chosen_arm)

        # find the related coefficients
        new_task = self.all_combinations[self.chosen_arm]

        return new_task


    def dump(self, dump_dict):
        dump_dict.update(self.bk)
        return dump_dict