from env_wrapper import DIAYN_Skill_Wrapper
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
import malmoenv
import gym

from pathlib import Path
xml = Path('/home/zilizhang/DIAYN/mobchase_single_agent.xml').read_text()
env = malmoenv.make()

env.init(xml, 9000)
total_timesteps = 3000
num_skills = 3
print(env.reward_range)
env = DIAYN_Skill_Wrapper(env, num_skills=num_skills)
#
# #check_env(env)
# obs = env.reset()
# env = gym.make('Walker2DMuJoCoEnv-v0')
n_steps = 1000
obs = env.reset()
done = False
for _ in range(n_steps):
    # Random action
    if not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(reward)
    else:
        print("has done")
    # action = env.action_space.sample()
    # obs, reward, done, info = env.step(action)
    # print(reward)
