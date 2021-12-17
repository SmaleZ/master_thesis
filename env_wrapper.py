import gym
import pybulletgym
from gym import Wrapper, spaces
from torch.optim import Adam
from nn_builder.pytorch.NN import NN
import torch.nn.functional as F
import random
import numpy as np
from stable_baselines3 import SAC
import torch
from torch import nn


class DIAYN_Skill_Wrapper(Wrapper):
    def __init__(self, env, num_skills):
        Wrapper.__init__(self, env)
        self.num_skills = num_skills
        # print(env.observation_space['pov'].shape)
        # self.state_size = env.observation_space.shape[0]
        self.state_size = env.observation_space.shape[0] * env.observation_space.shape[1] * env.observation_space.shape[
            2]
        self.hidden_size = 128

        # discriminator負責state到skill的映射
        self.discriminator = NN(input_dim=self.state_size,
                                layers_info=[self.hidden_size, self.hidden_size, self.num_skills],
                                hidden_activations="relu",
                                output_activation='none',
                                )
        # TODO: from nn_builder.pytorch.CNN import CNN
        #
        # # PyTorch CNNs in nn_builder specifies layers in the following format:
        # # ["conv", channels, kernel size, stride, padding]
        # # ["maxpool", kernel size, stride, padding]
        # # ["avgpool", kernel size, stride, padding]
        # # ["linear", units]
        # # where channels, kernel size, stride, padding and units must be specified as integers
        #
        #
        # # This builds a CNN with 5 layers. The first layer is a convolutional layer with
        # # 32 units, a kernel size of 3, stride of 1 and 0 padding. The next layer is a maxpool
        # # layer with kernel size 2, stride of 2 and 0 padding. The next layer is a conv layer with
        # # 64 units, kernel size 3, stride of 1 and 2 padding. The next layer is an average pooling layer
        # # with kernel size 2, stride 2 and 0 padding. The final layer is a linear layer with 10 units.
        # # The output activation is softmax, there is no dropout or batch norm and the initialiser used is xavier
        # model = CNN(input_dim=(3, 64, 64),
        #             layers_info=[["conv", 32, 3, 1, 0], ["maxpool", 2, 2, 0],
        #                          ["conv", 64, 3, 1, 2], ["avgpool", 2, 2, 0],
        #                          ["linear", 10]],
        #             hidden_activations="relu", output_activation="softmax", dropout=0.0,
        #             initialiser="xavier", batch_norm=True)
        #
        # # Note that a pytorch CNN expects input data in the form:  (batch, channels, height, width)
        self.discriminator_optimizer = Adam(self.discriminator.parameters())

        # skill的概率分布為均勻分布
        self.prior_probability_of_skill = 1.0 / self.num_skills

        # 在原本的狀態維度多加一個維度代表skill
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_size + 1,), dtype=np.float32)

    def reset(self, **kwargs):
        # 隨機採樣一個skill
        observation = self.env.reset(**kwargs)
        self.skill = random.randint(0, self.num_skills - 1)
        return self.observation(observation)

    def observation(self, observation):
        # 狀態跟skill組合起來返回
        return np.concatenate((np.array(observation), np.array([self.skill])))

    def step(self, action):
        # print("enter into step")
        # print("the current action we choose is :")
        # print(action)
        # 這裡不使用原生reward
        next_state, _, done, _ = self.env.step(action)
        # print("obs shape:")
        # print(next_state.shape)
        # 使用一種技巧計算reward
        new_reward, discriminator_outputs = self.calculate_new_reward(next_state)
        print("next_state is:", next_state)
        print("the shape of obs is:", next_state.shape)
        print("the pseudo reward is", new_reward)
        # discriminator 學習預測正確的skill
        self.disciminator_learn(self.skill, discriminator_outputs)
        return self.observation(next_state), new_reward, done, _

    def calculate_new_reward(self, next_state):
        # 取得disciminator輸出以及在正確skill上的數值
        probability_correct_skill, disciminator_outputs = self.get_predicted_probability_of_skill(self.skill,
                                                                                                  next_state)
        # 獎勵計算方式參考原始論文
        new_reward = np.log(probability_correct_skill + 1e-8) - np.log(self.prior_probability_of_skill)
        return new_reward, disciminator_outputs

    def disciminator_learn(self, skill, discriminator_outputs):
        # print("enter into learn")
        # print("shape of discriminator :", self.discriminator.hidden_layers)
        # 計算disciminator輸出對上正確skill的交叉商
        loss = nn.CrossEntropyLoss()(discriminator_outputs, torch.Tensor([skill]).long())
        # 把梯度求出來
        loss.backward()
        # 更新disciminator
        self.discriminator_optimizer.step()
        # 梯度清空
        self.discriminator_optimizer.zero_grad()

    def get_predicted_probability_of_skill(self, skill, next_state):
        # discriminator 根據next_state預測可能的skill
        predicted_probabilities_unnormalised = self.discriminator(torch.Tensor(np.array(next_state)).unsqueeze(0))
        print("predicted_probabilities_unnormalised :", predicted_probabilities_unnormalised)
        # 正確的skill的概率
        probability_of_correct_skill = F.softmax(predicted_probabilities_unnormalised, dim=-1)[:, skill]
        print("probability_of_correct_skill:", probability_of_correct_skill)
        return probability_of_correct_skill.item(), predicted_probabilities_unnormalised
