from env_wrapper import DIAYN_Skill_Wrapper
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
total_timesteps = 5
num_skills = 3

# env = gym.make('Walker2DMuJoCoEnv-v0')
# env = gym.make('MountainCarContinuous-v0')
# env = gym.make('InvertedPendulum-v2')
# env = gym.make('HalfCheetah-v2')
# env = gym.make('Walker2d-v3')
# env = gym.make('Ant-v2')
env = gym.make('Humanoid-v2')
# env = gym.make("MineRLNavigateDense-v0")
env = DIAYN_Skill_Wrapper(env, num_skills=num_skills)
# env = DummyVecEnv([lambda: env])
env = DummyVecEnv([lambda: env])
normalized_vec_env = VecNormalize(env)

agent = SAC("MlpPolicy", normalized_vec_env, verbose=1, tensorboard_log="./tensorboard").learn(total_timesteps=total_timesteps)
agent.save("sac_Walker2DMuJoCoEnv-v0")

# test git
# test commit
