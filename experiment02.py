from env_wrapper import DIAYN_Skill_Wrapper, DIAYN_VIC_Skill_Wrapper, DIAYN_Test_Wrapper, DIAYN_Pretrained_Wrapper
import gym
from stable_baselines3 import SAC
import numpy as np
import argparse

num_skills = 50
env_names = ['InvertedPendulum-v2']
AVAILABLES_MODES = ['train', 'train_vcl', 'eval', 'view']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',
                        type=str,
                        choices=AVAILABLES_MODES,
                        default='train')
    parser.add_argument('--env_name',
                        type=str,
                        default='HalfCheetah-v2')
    parser.add_argument('--total_pretraining_timesteps',
                        type=int,
                        default=100000)
    parser.add_argument('--total_training_timesteps',
                        type=int,
                        default=1000000)
    parser.add_argument('--agent_path',
                        type=str,
                        default=None)
    parser.add_argument('--num_skills',
                        type=int,
                        default=10)
    args = parser.parse_args()
    return args


def train(num_skills, total_pretraining_timesteps, total_training_timesteps, env_name):
    """

    :param num_skills: total number of skills
    :param total_pretraining_timesteps:  timesteps of pretraining
    :param total_training_timesteps: timesteps of continual training
    :param env_name: the envirorment choosed
    :return:
    """
    env = gym.make(env_name)
    pretrained_env = DIAYN_Skill_Wrapper(env, num_skills)
    pretrained_agent = SAC("MlpPolicy",
                           pretrained_env,
                           verbose=1,
                           tensorboard_log="./tensorboard/pretrained_model-{}-{}".format(env_name,
                                                                                         total_pretraining_timesteps)).learn(
        total_timesteps=total_pretraining_timesteps)
    pretrained_agent_path = "trainedmodel/pretrained_{}-{}".format(env_name, total_pretraining_timesteps)
    pretrained_agent.save(pretrained_agent_path)

    for i in range(num_skills):
        continutrained_env = DIAYN_Pretrained_Wrapper(skill_choosen=i, env=env)
        continutrained_agent = SAC.load(pretrained_agent_path, env=continutrained_env)
        continutrained_path = "trainedmodel/continualtrained_{}-{}-skill{}".format(env_name,
                                                                                   total_pretraining_timesteps, i)
        continutrained_agent.learn(total_timesteps=total_training_timesteps)
        continutrained_agent.save(continutrained_path)


def train_vcl(num_skills, total_pretraining_timesteps, total_training_timesteps, env_name):
    """

    :param num_skills: total number of skills
    :param total_pretraining_timesteps:  timesteps of pretraining
    :param total_training_timesteps: timesteps of continual training
    :param env_name: the envirorment choosed
    :return:
    """
    env = gym.make(env_name)
    pretrained_env = DIAYN_VIC_Skill_Wrapper(env, num_skills)
    pretrained_agent = SAC("MlpPolicy",
                           pretrained_env,
                           verbose=1,
                           tensorboard_log="./tensorboard/VCL_pretrained_model-{}-{}".format(env_name,
                                                                                             total_pretraining_timesteps)).learn(
        total_timesteps=total_pretraining_timesteps)
    pretrained_agent_path = "trainedmodel/VCL_pretrained_{}-{}".format(env_name, total_pretraining_timesteps)
    pretrained_agent.save(pretrained_agent_path)

    for i in range(num_skills):
        continutrained_env = DIAYN_Pretrained_Wrapper(skill_choosen=i, env=env)
        continutrained_agent = SAC.load(pretrained_agent_path, env=continutrained_env)
        continutrained_path = "trainedmodel/VCL_continualtrained_{}-{}-skill{}".format(env_name,
                                                                                       total_pretraining_timesteps, i)
        continutrained_agent.learn(total_timesteps=total_training_timesteps)
        continutrained_agent.save(continutrained_path)


def run_experiments(args):
    mode = args.mode
    if mode == 'train':
        train(
            num_skills=args.num_skills,
            total_pretraining_timesteps=args.total_pretraining_timesteps,
            total_training_timesteps=args.total_training_timesteps,
            env_name=args.env_name
        )
    elif mode == 'train_vcl':
        train_vcl(
            num_skills=args.num_skills,
            total_pretraining_timesteps=args.total_pretraining_timesteps,
            total_training_timesteps=args.total_training_timesteps,
            env_name=args.env_name
        )


if __name__ == '__main__':
    args = parse_args()
    run_experiments(args)
