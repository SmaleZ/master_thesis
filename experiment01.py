from env_wrapper import DIAYN_Skill_Wrapper, DIAYN_VIC_Skill_Wrapper, DIAYN_Test_Wrapper
import gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import numpy as np

# total_training_timesteps = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000]
total_training_timesteps = [100000, 200000, 300000, 400000]
# total_training_timesteps = [15000]
num_skills = 50
# env_names = ['MountainCarContinuous-v0', 'InvertedPendulum-v2']
# env_names = ["BipedalWalker-v3"]
env_names = ['MountainCarContinuous-v0']
# env_names = ['InvertedPendulum-v2']
# env = gym.make('MountainCarContinuous-v0')
# env = gym.make('InvertedPendulum-v2')
def train(num_skills, total_training_timesteps, env_names):
    for env_name in env_names:
        env = gym.make(env_name)
        env = DIAYN_Skill_Wrapper(env, num_skills=num_skills)
        # env = DummyVecEnv([lambda: env])
        # normalized_vec_env = VecNormalize(env)
        for total_training_timestep in total_training_timesteps:
            agent = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard").learn(
                total_timesteps=total_training_timestep)
            agent.save("trainedmodel/{}-{}".format(env_name, total_training_timestep))


def eval(num_skills, env_name, agent_path):
    model = SAC.load(agent_path)
    #print(env.observation_space.shape)
    skill_reward = []
    for skill_idx in range(num_skills):
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

        print("the choosen skill : {} 's reward is : {}".format(skill_idx, np.sum(episode_reward)))
        skill_reward.append(np.mean(episode_reward))


    print("the skill reward of {} is {}: ".format(env_name, np.mean(skill_reward)))



## run experiment:
## fist step : train the model on two gym_env with different training scale
# train(num_skills=num_skills, total_training_timesteps=total_training_timesteps, env_names=env_names)

# 2nd step : eval
# for env_name in env_names:
#     for total_training_timestep in total_training_timesteps:
#         agent_path = "trainedmodel/{}-{}.zip".format(env_name, total_training_timestep)
#         eval(num_skills, env_name, agent_path)

# eval(num_skills=num_skills, env_name='InvertedPendulum-v2', agent_path='InvertedPendulum-v2-90000.zip')
# eval(num_skills=num_skills, env_name='MountainCarContinuous-v0', agent_path='MountainCarContinuous-v0-90000.zip')
# eval(num_skills=num_skills, env_name="BipedalWalker-v3", agent_path='BipedalWalker-v3-30000.zip')
model = SAC.load('trainedmodel/MountainCarContinuous-v0-100000.zip')
for skill_idx in range(num_skills):
    print("choose skill : {}".format(skill_idx))
    env = gym.make('MountainCarContinuous-v0')
    env = DIAYN_Test_Wrapper(env, skill_idx=skill_idx)
    obs = env.reset()
    done = False
    action = env.action_space.sample()
    while not done:
        env.render(mode='human')
        obs, reward, done, _ = env.step(action)
        if done:
            break
        action, _state = model.predict(obs)
    env.close()
