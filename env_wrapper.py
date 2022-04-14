from gym import Wrapper, spaces
from torch.optim import Adam
from nn_builder.pytorch.NN import NN
import torch.nn.functional as F
import random
import numpy as np
from stable_baselines3 import SAC
import torch
from torch import nn
from discriminator.vcl_nn import DiscriminativeVCL

LR = 0.001
INITIAL_POSTERIOR_VAR = 1e-3

N_CLASSES = 10
LAYER_WIDTH = 100
N_HIDDEN_LAYERS = 3 # bigger net for complex env
N_TASKS = 3
TASK_SIZE = 100
MULTIHEADED = False
CORESET_SIZE = 200
EPOCHS = 100
BATCH_SIZE = 256
TRAIN_FULL_CORESET = True
TRAIN_NUM_SAMPLES = 10

device = torch.device("cpu")
print("Running on device", device)

class DIAYN_Pretrained_Wrapper(Wrapper):
    def __init__(self, env, skill_choosen):
        Wrapper.__init__(self, env)
        self.skill_choosen = skill_choosen
        self.state_size = env.observation_space.shape[0]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_size + 1,), dtype=np.float32)

    def reset(self, **kwargs):
        # one skill
        observation = self.env.reset(**kwargs)
        return self.observation(observation)

    def observation(self, observation):
        # concatenate skill with state
        return np.concatenate((np.array(observation), np.array([self.skill_choosen])))

    def step(self, action):
        next_state, reward, done, _ = self.env.step(action)
        return self.observation(next_state), reward, done, _


class DIAYN_Skill_Wrapper(Wrapper):
    def __init__(self, env, num_skills):
        Wrapper.__init__(self, env)
        self.num_skills = num_skills
        self.current_step = 0
        self.skill = random.randint(0, self.num_skills - 1)
        self.state_size = env.observation_space.shape[0]
        print("the shape of observation is : {}".format(env.observation_space.shape))
        self.hidden_size = 128
        # discriminator負責state到skill的映射
        self.discriminator = NN(input_dim=self.state_size,
                                layers_info=[self.hidden_size, self.hidden_size, self.num_skills],
                                hidden_activations="relu",
                                output_activation='none',
                                )

        self.discriminator_optimizer = Adam(self.discriminator.parameters(), lr=LR)
        # union distribution for skill
        self.prior_probability_of_skill = 1.0 / self.num_skills

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_size + 1,), dtype=np.float32)

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.skill = random.randint(0, self.num_skills - 1)
        return self.observation(observation)

    def observation(self, observation):
        return np.concatenate((np.array(observation), np.array([self.skill])))

    def step(self, action):
        next_state, reward, done, _ = self.env.step(action)
        new_reward, discriminator_outputs = self.calculate_new_reward(next_state)
        reward = new_reward
        self.discriminator_learn(self.skill, discriminator_outputs)
        self.current_step += 1
        return self.observation(next_state), reward, done, _

    def calculate_new_reward(self, next_state):
        probability_correct_skill, disciminator_outputs = self.get_predicted_probability_of_skill(self.skill,
                                                                                                  next_state)
        new_reward = np.log(probability_correct_skill + 1e-8) - np.log(self.prior_probability_of_skill)

        return new_reward, disciminator_outputs

    def discriminator_learn(self, skill, discriminator_outputs):
        loss = nn.CrossEntropyLoss()(discriminator_outputs, torch.Tensor([skill]).long())

        self.discriminator_optimizer.zero_grad()
        loss.backward()
        self.discriminator_optimizer.step()

    def get_predicted_probability_of_skill(self, skill, next_state):
        predicted_probabilities_unnormalised = self.discriminator(torch.Tensor(np.array(next_state)).unsqueeze(0))
        probability_of_correct_skill = F.softmax(predicted_probabilities_unnormalised, dim=-1)[:, skill]
        return probability_of_correct_skill.item(), predicted_probabilities_unnormalised


class DIAYN_VIC_Skill_Wrapper(Wrapper):
    def __init__(self, env, num_skills):
        Wrapper.__init__(self, env)
        self.num_skills = num_skills
        self.skill = random.randint(0, self.num_skills - 1)
        # print(env.observation_space['pov'].shape)
        self.state_size = env.observation_space.shape[0]
        print("the shape of observation is : {}".format(env.observation_space.shape))
        # self.hidden_size = 100
        self.hidden_size = 128
        # record the curr iteration for variational uodate, every 100 steps there is a update
        self.curr_steps = 0
        # record the curr tasks for variational uodate, there are total 10 different ceil/tasks/head for
        # prior/posterior store
        self.curr_task = 0
        # change the original discriminator to vcl discriminative model
        self.discriminator = DiscriminativeVCL(
            in_size=self.state_size, out_size=self.num_skills,
            layer_width=self.hidden_size,
            n_hidden_layers=N_HIDDEN_LAYERS,
            n_heads=(N_TASKS if MULTIHEADED else 1),
            initial_posterior_var=INITIAL_POSTERIOR_VAR).to(device)
        self.discriminator_optimizer = Adam(self.discriminator.parameters(), lr=LR)
        self.prior_probability_of_skill = 1.0 / self.num_skills
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_size + 1,), dtype=np.float32)


    def reset(self, **kwargs):
        # reset the prior and posterior every episodic
        self.curr_task += 1
        self.discriminator.reset_for_new_task(0)
        observation = self.env.reset(**kwargs)
        self.skill = random.randint(0, self.num_skills - 1)
        return self.observation(observation)

    def observation(self, observation):

        return np.concatenate((np.array(observation), np.array([self.skill])))

    def step(self, action):

        next_state, _, done, _ = self.env.step(action)

        new_reward, discriminator_outputs = self.calculate_new_reward(next_state, 0)

        self.disciminator_learn(self.skill, discriminator_outputs, next_state, 0)

        return self.observation(next_state), new_reward, done, _

    def calculate_new_reward(self, next_state, head):

        probability_correct_skill, disciminator_outputs = self.get_predicted_probability_of_skill(self.skill,
                                                                                                  next_state, head)
        # calculate new rewards
        new_reward = np.log(probability_correct_skill + 1e-8) - np.log(self.prior_probability_of_skill)
        return new_reward, disciminator_outputs

    def disciminator_learn(self, skill, discriminator_outputs, next_state, task_idx):
        """
        one step gradient descent
        :param skill: the label of next_state is the skill be chosed at the start of episodic
        :param discriminator_outputs: the prediction of the output
        :param next_state: the input
        :param task_idx: the current id of task
        :return:
        """
        optimizer = self.discriminator_optimizer
        optimizer.zero_grad()
        head = task_idx if MULTIHEADED else 0

        x = torch.Tensor(np.array(next_state)).unsqueeze(0)
        x = x.to(device)
        y_true = torch.Tensor([skill]).long()
        y_true = y_true.to(device)

        loss = self.discriminator.vcl_loss(x, y_true, head, TRAIN_NUM_SAMPLES)
        
        loss.backward()
        optimizer.step()

    def get_predicted_probability_of_skill(self, skill, next_state, head):
        predicted_probabilities_unnormalised = self.discriminator(torch.Tensor(np.array(next_state)).unsqueeze(0), head)
        probability_of_correct_skill = F.softmax(predicted_probabilities_unnormalised, dim=-1)[:, skill]
        return probability_of_correct_skill.item(), predicted_probabilities_unnormalised


class DIAYN_Test_Wrapper(Wrapper):
    def __init__(self, env, skill_idx):
        Wrapper.__init__(self, env)
        # skill_idx means the current choosed skill
        self.skill = skill_idx
        self.state_size = env.observation_space.shape[0]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_size + 1,), dtype=np.float32)

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self.observation(observation)

    def observation(self, observation):
        return np.concatenate((np.array(observation), np.array([self.skill])))

    def step(self, action):
        # the input action is obtained by the trained model (model.predict)
        next_state, reward, done, _ = self.env.step(action)

        return self.observation(next_state), reward, done, _
