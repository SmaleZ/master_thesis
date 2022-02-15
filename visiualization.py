import csv
import matplotlib.pyplot as plt
from env_wrapper import DIAYN_Skill_Wrapper, DIAYN_VIC_Skill_Wrapper, DIAYN_Test_Wrapper
import gym
from stable_baselines3 import SAC
import seaborn as sns
import numpy as np


# loss_file = open("train_loss.csv")
# loss_file_reader = csv.reader(loss_file)
# data = list(loss_file_reader)
# losses = list()
# time_steps = list()
# for i in range(len(data)):
#     time_steps.append(i)
#     losses.append(data[i][0])
#
# plt.plot(time_steps, losses)
# plt.show()
def plot_distribution():
    model = SAC.load('trainedmodel/Hopper-v3-250000.zip')
    # print(env.observation_space.shape)
    skill_reward = []
    for skill_idx in range(50):
        env = gym.make('Hopper-v3')
        env = DIAYN_Test_Wrapper(env, skill_idx=skill_idx)
        # print(skill_idx)
        # print(env.observation_space.shape)
        done = False
        episode_reward = []
        obs = env.reset()

        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            episode_reward.append(reward)
        env.close()

        # print("the choosen skill : {} 's reward is : {}".format(skill_idx, np.sum(episode_reward)))
        skill_reward.append(np.sum(episode_reward))

    sns.set(style="darkgrid")
    sns.kdeplot(skill_reward)
    plt.show()


def plot_stack_distribution():
    # model = SAC.load('trainedmodel/InvertedPendulum-v2-400000.zip')
    small_rewards = []
    big_rewards = []
    # epochs = ['100000', '250000', '500000', '750000', '1000000']
    epochs = ['1000','50000', '100000', '200000', '300000', '400000']
    for epoch in epochs:
        model = SAC.load('trainedmodel/DIAYN_VCL-InvertedPendulum-v2-'+epoch+'.zip')
        small_reward = 0
        big_reward = 0
        for skill_idx in range(50):
            env = gym.make('InvertedPendulum-v2')
            env = DIAYN_Test_Wrapper(env, skill_idx=skill_idx)
            # print(skill_idx)
            # print(env.observation_space.shape)
            done = False
            episode_reward = []
            obs = env.reset()

            while not done:
                action, _state = model.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                episode_reward.append(reward)
            env.close()
            if np.sum(episode_reward) < 500:
                small_reward += 1
            elif np.sum(episode_reward) >= 500:
                big_reward += 1
            print("the choosen skill : {} 's reward is : {}".format(skill_idx, np.sum(episode_reward)))
        small_rewards.append(small_reward)
        big_rewards.append(big_reward)

    plt.bar(epochs, small_rewards, color="green", label="reward<500")
    plt.bar(epochs, big_rewards, color="yellow", bottom=np.array(small_rewards),label="reward>=500")
    print(small_rewards)
    print(big_rewards)
    plt.ylabel('num.skills')
    plt.xlabel('total timesteps')
    plt.legend()
    plt.show()

def plot_stack_distribution_mcc():
    # model = SAC.load('trainedmodel/InvertedPendulum-v2-400000.zip')
    level1_rewards = []
    level2_rewards = []
    level3_rewards = []
    level4_rewards = []
    # epochs = ['100000', '250000', '500000', '750000', '1000000']
    epochs = ['1000','50000', '100000', '200000', '300000', '400000']
    for epoch in epochs:
        model = SAC.load('trainedmodel/MountainCarContinuous-v0-'+epoch+'.zip')
        level1_reward = 0
        level2_reward = 0
        level3_reward = 0
        level4_reward = 0
        for skill_idx in range(50):
            env = gym.make('MountainCarContinuous-v0')
            env = DIAYN_Test_Wrapper(env, skill_idx=skill_idx)
            # print(skill_idx)
            # print(env.observation_space.shape)
            done = False
            episode_reward = []
            obs = env.reset()

            while not done:
                action, _state = model.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                episode_reward.append(reward)
            env.close()
            total_episode_reward = np.sum(episode_reward)
            if total_episode_reward < -50:
                level1_reward += 1
            elif -50 <= total_episode_reward < -20:
                level2_reward += 1
            elif -20 <= total_episode_reward < 0:
                level3_reward += 1
            elif total_episode_reward >= 0:
                level4_reward += 1
            #print("the choosen skill : {} 's reward is : {}".format(skill_idx, np.sum(episode_reward)))
        level1_rewards.append(level1_reward)
        level2_rewards.append(level2_reward)
        level3_rewards.append(level3_reward)
        level4_rewards.append(level4_reward)

    plt.bar(epochs, np.array(level1_rewards), label="reward<-50")
    plt.bar(epochs, level2_rewards, bottom=np.array(level1_rewards), label="reward<-20 & >= -50")
    plt.bar(epochs, level3_rewards, bottom=np.array(level2_rewards) + np.array(level1_rewards), label="reward<0 & >= -20")
    plt.bar(epochs, level4_rewards, bottom=np.array(level3_rewards) + np.array(level2_rewards) + np.array(level1_rewards), label="reward>=0")
    # plt.hist([level1_rewards, level2_rewards, level3_rewards, level4_rewards], stacked=True)

    print(level1_rewards)
    print(level2_rewards)
    print(level3_rewards)
    print(level4_rewards)

    plt.ylabel('num.skills')
    plt.xlabel('total timesteps')
    plt.legend()
    plt.show()

def plot_hist():
    model = SAC.load('trainedmodel/InvertedPendulum-v2-100000.zip')
    # print(env.observation_space.shape)
    skill_reward = []
    for skill_idx in range(50):
        env = gym.make('InvertedPendulum-v2')
        env = DIAYN_Test_Wrapper(env, skill_idx=skill_idx)
        # print(skill_idx)
        # print(env.observation_space.shape)
        done = False
        episode_reward = []
        obs = env.reset()

        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            episode_reward.append(reward)
        env.close()

        # print("the choosen skill : {} 's reward is : {}".format(skill_idx, np.sum(episode_reward)))
        skill_reward.append(np.sum(episode_reward))

    plt.hist(skill_reward)
    plt.ylabel('num.skills')
    plt.xlabel('rewards')
    plt.show()

# plot_hist()
# plot_stack_distribution()
plot_stack_distribution_mcc()