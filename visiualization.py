import csv
import matplotlib.pyplot as plt
from env_wrapper import DIAYN_Skill_Wrapper, DIAYN_VIC_Skill_Wrapper, DIAYN_Test_Wrapper
import gym
from stable_baselines3 import SAC
import seaborn as sns
import numpy as np

import argparse

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

AVAILABLES_MODES = ['distribution', 'stack', 'hist', 'stack_mcc']

ENV_NAMES = {'MountainCarContinuous-v0': ['15000', '50000', '100000', '200000', '300000', '400000', '500000'],
             'InvertedPendulum-v2': ['50000', '100000', '200000', '300000', '400000'],
             'Hopper-v3': ['100000', '250000', '500000', '750000', '1000000'],
             'HalfCheetah-v2': ['100000', '250000', '500000', '750000', '1000000'],
             'Ant-v2': ['100000', '250000', '500000', '750000', '1000000']}


def plot_distribution():
    model = SAC.load('trainedmodel/HalfCheetah-v2-100000.zip')
    # print(env.observation_space.shape)
    skill_reward = []
    for skill_idx in range(50):
        env = gym.make('HalfCheetah-v2')
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


def plot_stack_distribution(env_name, agent):
    # model = SAC.load('trainedmodel/InvertedPendulum-v2-400000.zip')
    level1_rewards = []
    level2_rewards = []
    level3_rewards = []
    level4_rewards = []
    level5_rewards = []
    level6_rewards = []
    epochs = ENV_NAMES.get(env_name)
    # epochs = ['1000','50000', '100000', '200000', '300000', '400000']
    for epoch in epochs:
        if agent == 'DIAYN_VCL':
            model = SAC.load('trainedmodel/{}-{}-{}.zip'.format(agent, env_name, epoch))
        else:
            model = SAC.load('trainedmodel/{}-'.format(env_name) + epoch + '.zip')
        level1_reward = 0
        level2_reward = 0
        level3_reward = 0
        level4_reward = 0
        level5_reward = 0
        level6_reward = 0
        for skill_idx in range(50):
            env = gym.make(env_name)
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
            if np.sum(episode_reward) < 100:
                level1_reward += 1
            elif 100 <= np.sum(episode_reward) < 200:
                level2_reward += 1
            elif 200 <= np.sum(episode_reward) < 300:
                level3_reward += 1
            elif 300 <= np.sum(episode_reward) < 400:
                level4_reward += 1
            elif 400 <= np.sum(episode_reward) < 1000:
                level5_reward += 1
            elif 100 <= np.sum(episode_reward):
                level6_reward += 1
            # print("the choosen skill : {} 's reward is : {}".format(skill_idx, np.sum(episode_reward)))
        # small_rewards.append(small_reward)
        # big_rewards.append(big_reward)
        level1_rewards.append(level1_reward)
        level2_rewards.append(level2_reward)
        level3_rewards.append(level3_reward)
        level4_rewards.append(level4_reward)
        level5_rewards.append(level5_reward)
        level6_rewards.append(level6_reward)

    plt.bar(epochs, np.array(level1_rewards), label="reward < 100")
    plt.bar(epochs, level2_rewards, bottom=np.array(level1_rewards), label="100 <= reward < 200")
    plt.bar(epochs, level3_rewards, bottom=np.array(level2_rewards) + np.array(level1_rewards),
            label="200 <= reward < 300")
    plt.bar(epochs, level4_rewards,
            bottom=np.array(level3_rewards) + np.array(level2_rewards) + np.array(level1_rewards),
            label="300 <= reward < 400")
    plt.bar(epochs, level5_rewards,
            bottom=np.array(level3_rewards) + np.array(level2_rewards) + np.array(level1_rewards) + np.array(level4_rewards),
            label="400 <= reward < 1000")
    plt.bar(epochs, level6_rewards,
            bottom=np.array(level3_rewards) + np.array(level2_rewards) + np.array(level1_rewards) + np.array(
                level4_rewards) + np.array(level5_rewards),
            label="reward >= 1000")

    plt.ylabel('num.skills')
    plt.xlabel('total timesteps')
    plt.title('training dynamics on {}'.format(env_name))
    plt.legend()
    if agent == 'DIAYN_VCL':
        plt.title('DIAYN_VCL training dynamics on {}'.format(env_name))
        plt.savefig(fname='results/DIAYN_VCL_' + 'training_dynamic_' + env_name + '.png')
    else:
        plt.title('DIAYN training dynamics on {}'.format(env_name))
        plt.savefig(fname='results/'+ 'training_dynamic_' + env_name + '.png')
    plt.show()


def plot_stack_distribution_mcc(env_name, agent):
    # model = SAC.load('trainedmodel/InvertedPendulum-v2-400000.zip')
    level1_rewards = []
    level2_rewards = []
    level3_rewards = []
    level4_rewards = []
    # epochs = ['100000', '250000', '500000', '750000', '1000000']
    epochs = ['1000', '50000', '100000', '200000', '300000', '400000']
    for epoch in epochs:
        if agent == 'DIAYN_VCL':
            model = SAC.load('trainedmodel/{}-{}-{}.zip'.format(agent, env_name, epoch))
        else:
            model = SAC.load('trainedmodel/{}-'.format(env_name) + epoch + '.zip')
        level1_reward = 0
        level2_reward = 0
        level3_reward = 0
        level4_reward = 0
        for skill_idx in range(50):
            env = gym.make(env_name)
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
            # print("the choosen skill : {} 's reward is : {}".format(skill_idx, np.sum(episode_reward)))
        level1_rewards.append(level1_reward)
        level2_rewards.append(level2_reward)
        level3_rewards.append(level3_reward)
        level4_rewards.append(level4_reward)

    plt.bar(epochs, np.array(level1_rewards), label="reward<-50")
    plt.bar(epochs, level2_rewards, bottom=np.array(level1_rewards), label="reward<-20 & >= -50")
    plt.bar(epochs, level3_rewards, bottom=np.array(level2_rewards) + np.array(level1_rewards),
            label="reward<0 & >= -20")
    plt.bar(epochs, level4_rewards,
            bottom=np.array(level3_rewards) + np.array(level2_rewards) + np.array(level1_rewards), label="reward>=0")
    # plt.hist([level1_rewards, level2_rewards, level3_rewards, level4_rewards], stacked=True)

    # print(level1_rewards)
    # print(level2_rewards)
    # print(level3_rewards)
    # print(level4_rewards)
    # plt.title('training dynamics on {}'.format(env_name))
    plt.ylabel('num.skills')
    plt.xlabel('total timesteps')
    plt.legend()
    if agent == 'DIAYN_VCL':
        plt.title('DIAYN_VCL training dynamics on {}'.format(env_name))
        plt.savefig(fname='results/DIAYN_VCL_' + 'training_dynamic_' + env_name + '.png')
    else:
        plt.title('DIAYN training dynamics on {}'.format(env_name))
        plt.savefig(fname='results/'+ 'training_dynamic_' + env_name + '.png')
    plt.show()


def plot_hist(env_name, agent):
    # random_path = 'trainedmodel/Ant-v2-1000.zip'
    # agent = ''
    # training_steps = training_steps
    # env_name = env_name
    training_steps = ENV_NAMES.get(env_name)
    for training_step in training_steps:

        if agent == 'DIAYN_VCL':
            random_path = 'trainedmodel/{}-{}-1000.zip'.format(agent, env_name)
            path = 'trainedmodel/{}-{}-{}.zip'.format(agent, env_name, training_step)
        else:
            random_path = 'trainedmodel/{}-1000.zip'.format(env_name)
            path = 'trainedmodel/{}-{}.zip'.format(env_name, training_step)
        random_model = SAC.load(random_path)
        model = SAC.load(path)
        # print(env.observation_space.shape)
        random_skill_reward = []
        skill_reward = []
        for skill_idx in range(50):
            env = gym.make(env_name)
            env = DIAYN_Test_Wrapper(env, skill_idx=skill_idx)
            # print(skill_idx)
            # print(env.observation_space.shape)
            done = False
            random_done = False
            episode_reward = []
            random_episode_reward = []
            obs = env.reset()

            while not done:
                action, _state = model.predict(obs, deterministic=True)

                obs, reward, done, _ = env.step(action)

                episode_reward.append(reward)

            random_obs = env.reset()

            while not random_done:
                random_action, random_state = random_model.predict(random_obs, deterministic=True)

                random_obs, random_reward, random_done, _ = env.step(random_action)

                random_episode_reward.append(random_reward)

            env.close()

            # print("the choosen skill : {} 's reward is : {}".format(skill_idx, np.sum(episode_reward)))
            skill_reward.append(np.sum(episode_reward))
            random_skill_reward.append(np.sum(random_episode_reward))

        plt.hist(skill_reward, alpha=0.5, label="learned_skills")
        plt.hist(random_skill_reward, alpha=0.5, label="random_skills")
        # plt.hist([skill_reward, random_skill_reward], alpha=0.5, label=["skill_reward", "random_skills"])
        plt.ylabel('num.skills')
        plt.xlabel('rewards')
        plt.legend(loc='upper right')
        if agent == 'DIAYN_VCL':
            plt.title(agent + '-' + env_name + '-' + training_step)
            plt.savefig(fname='results/' + agent + '-' + env_name + '-' + training_step + '.png')
        else:
            plt.title(env_name + '-' + training_step)
            plt.savefig(fname='results/' + env_name + '-' + training_step + '.png')
        plt.close()
        # plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',
                        type=str,
                        choices=AVAILABLES_MODES,
                        default='hist')
    parser.add_argument('--agent_path',
                        type=str,
                        default=None)
    parser.add_argument('--env_name',
                        type=str,
                        default='HalfCheetah-v2')
    parser.add_argument('--training_steps',
                        type=str,
                        default='100000')
    parser.add_argument('--agent',
                        type=str,
                        default='')
    args = parser.parse_args()
    return args


def run_experiments(args):
    mode = args.mode
    if mode == 'distribution':
        plot_distribution()
    elif mode == 'stack':
        plot_stack_distribution(args.env_name, args.agent)
    elif mode == 'hist':
        plot_hist(args.env_name, args.agent)
    elif mode == 'stack_mcc':
        plot_stack_distribution_mcc(args.env_name, args.agent)

if __name__ == '__main__':
    args = parse_args()
    run_experiments(args)
