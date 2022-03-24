from env_wrapper import DIAYN_Skill_Wrapper, DIAYN_Pretrained_Wrapper
import gym
import pybulletgym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
# import minerl
import malmoenv
from pathlib import Path
import time

# xml = Path('/home/zilizhang/DIAYN/mobchase_single_agent.xml').read_text()
# env = malmoenv.make()
# #
# env.init(xml, 9000)
# # print("before wrapper")
# #print(env.observation_space.shape)
total_timesteps = 1000000
total_unsupervised_steps = 1000000
total_supervised_steps = 200000
num_skills = 10

# env = gym.make('Walker2DMuJoCoEnv-v0')
# env = gym.make('MountainCarContinuous-v0')
# env = gym.make('InvertedPendulum-v2')
# env = gym.make('HalfCheetah-v2')
# env = gym.make('Hopper-v3')

# env = gym.make('Walker2d-v3')
env = gym.make('Ant-v2')
# env = gym.make('Ant-v2')
# env = gym.make('Humanoid-v2')
# env = gym.make("MineRLNavigateDense-v0")
# pretrain_env = DIAYN_Skill_Wrapper(env, num_skills=num_skills)
# env = DummyVecEnv([lambda: env])
# env = DummyVecEnv([lambda: env])
# normalized_vec_env = VecNormalize(env)
print(env.observation_space.shape)
# agent = SAC("MlpPolicy", pretrain_env, verbose=1, tensorboard_log="./tensorboard/test").learn(total_timesteps=total_timesteps)
# agent.save("test_diayn_initialization_halfcheetah")
# for skill in [2,3,4,5,6,7,8,9]:
#     continuetrain_env = DIAYN_Pretrained_Wrapper(env, skill_choosen=skill)
#     continue_train_agent = SAC.load("trainedmodel/pretrained_Hopper-v3-1000000.zip", env=continuetrain_env)
#     continue_train_agent.learn(total_timesteps=total_supervised_steps)
#     continutrained_path = "trainedmodel/continualtrained_{}-{}-skill{}".format("Hopper-v3", total_unsupervised_steps, skill)
#     continue_train_agent.save(continutrained_path)


# test git
# test commit
