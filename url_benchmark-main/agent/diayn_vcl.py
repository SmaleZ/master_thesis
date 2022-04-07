import math
from collections import OrderedDict

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dm_env import specs

import utils
from agent.ddpg import DDPGAgent

from discriminator.vcl_nn import DiscriminativeVCL
from discriminator.discriminator_learn import run_point_estimate_initialisation, run_task
from util.operations import class_accuracy

LR = 0.001
INITIAL_POSTERIOR_VAR = 1e-3

N_CLASSES = 10
LAYER_WIDTH = 100
N_HIDDEN_LAYERS = 2
N_TASKS = 3
TASK_SIZE = 100
MULTIHEADED = False
CORESET_SIZE = 200
EPOCHS = 100
BATCH_SIZE = 256
TRAIN_FULL_CORESET = True
TRAIN_NUM_SAMPLES = 10


class DIAYNVCLAgent(DDPGAgent):
    def __init__(self, update_skill_every_step, skill_dim, diayn_scale,
                 update_encoder, **kwargs):
        self.skill_dim = skill_dim
        self.update_skill_every_step = update_skill_every_step
        self.diayn_scale = diayn_scale
        self.update_encoder = update_encoder
        self.hidden_size = 128
        self.n_hidden_layers = 2
        # increase obs shape to include skill dim
        kwargs["meta_dim"] = self.skill_dim
        self.device  = kwargs['device']
        self.steps_for_point_estimate = 10000

        # create actor and critic
        super().__init__(**kwargs)

        self.diayn_vcl = DiscriminativeVCL(
            in_size = self.obs_dim - self.skill_dim,
            out_size = self.skill_dim,
            layer_width = kwargs['hidden_dim'],
            n_hidden_layers = self.n_hidden_layers,
            n_heads = (N_TASKS if MULTIHEADED else 1),
            initial_posterior_var = INITIAL_POSTERIOR_VAR).to(self.device)

        # loss criterion
        # self.diayn_criterion = nn.CrossEntropyLoss()
        # self.diayn_vcl_criterion = self.diayn_vcl.vcl_loss()
        # optimizers
        self.diayn_vcl_opt = torch.optim.Adam(self.diayn_vcl.parameters(), lr=self.lr)

        self.diayn_vcl.train()

    def get_meta_specs(self):
        return (specs.Array((self.skill_dim,), np.float32, 'skill'),)

    def init_meta(self):
        skill = np.zeros(self.skill_dim, dtype=np.float32)
        skill[np.random.choice(self.skill_dim)] = 1.0
        meta = OrderedDict()
        meta['skill'] = skill
        #reset the posterior and prior
        # if global_step % (self.update_skill_every_step * 10 ) == 0:
        #     self.diayn_vcl.reset_for_new_task(0)
        return meta

    def update_meta(self, meta, global_step, time_step):
        if global_step % self.update_skill_every_step == 0:
            #randomly choose a skill at the begining of the epsisode
            # if global_step % (self.update_skill_every_step * 1000) == 0 and global_step > self.steps_for_point_estimate:
            #     self.diayn_vcl.reset_for_new_task(0)
            return self.init_meta()
        return meta

    def update_diayn_vcl(self, skill, next_obs, step):
        metrics = dict()
        
        #here the loss is calculated by vcl_loss()
        loss, df_accuracy = self.compute_diayn_vcl_loss(next_obs, skill, step)

        self.diayn_vcl_opt.zero_grad()
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.diayn_vcl_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['diayn_loss'] = loss.item()
            metrics['diayn_acc'] = df_accuracy

        return metrics

    def compute_intr_reward(self, skill, next_obs, step):
        z_hat = torch.argmax(skill, dim=1)
        d_pred = self.diayn_vcl(next_obs, 0)
        d_pred_log_softmax = F.log_softmax(d_pred, dim=1)
        _, pred_z = torch.max(d_pred_log_softmax, dim=1, keepdim=True)
        reward = d_pred_log_softmax[torch.arange(d_pred.shape[0]),
                                    z_hat] - math.log(1 / self.skill_dim)
        reward = reward.reshape(-1, 1)

        return reward * self.diayn_scale

    def compute_diayn_vcl_loss(self, next_state, skill, step):
        """
        # DF Loss
        VCL LOSS
        """
        z_hat = torch.argmax(skill, dim=1)
        head = 0
        # x = torch.Tensor(np.array(next_state)).unsqueeze(0)
        # x = x.to(device)
        x = next_state
        x = x.to(self.device)
        # y_true = torch.Tensor([skill]).long()
        # y_true = y_true.to(device)
        y_true = z_hat
        y_true = y_true.to(self.device)
        
        if step <= self.steps_for_point_estimate:
            # print ("point estimate")
            d_loss = self.diayn_vcl.point_estimate_loss(x, y_true)
        else:
            d_loss = self.diayn_vcl.vcl_loss(x, y_true, head, TRAIN_NUM_SAMPLES)
        # print(next_state.size())
        # print(len((next_state).unsqueeze(0)))
        y_pred = self.diayn_vcl.prediction((next_state).unsqueeze(0), head)
        y_pred = y_pred.to(self.device)
        df_accuracy = class_accuracy(y_pred, y_true)
        
        return d_loss, df_accuracy

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)

        obs, action, extr_reward, discount, next_obs, skill = utils.to_torch(
            batch, self.device)

        # augment and encode
        obs = self.aug_and_encode(obs)
        next_obs = self.aug_and_encode(next_obs)

        if self.reward_free:
            metrics.update(self.update_diayn_vcl(skill, next_obs, step))

            with torch.no_grad():
                intr_reward = self.compute_intr_reward(skill, next_obs, step)

            if self.use_tb or self.use_wandb:
                metrics['intr_reward'] = intr_reward.mean().item()
            reward = intr_reward
        else:
            reward = extr_reward

        if self.use_tb or self.use_wandb:
            metrics['extr_reward'] = extr_reward.mean().item()
            metrics['batch_reward'] = reward.mean().item()

        if not self.update_encoder:
            obs = obs.detach()
            next_obs = next_obs.detach()

        # extend observations with skill
        obs = torch.cat([obs, skill], dim=1)
        next_obs = torch.cat([next_obs, skill], dim=1)

        # update critic
        metrics.update(
            self.update_critic(obs.detach(), action, reward, discount,
                               next_obs.detach(), step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
