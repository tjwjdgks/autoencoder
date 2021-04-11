from __future__ import print_function

import numpy as np

import math

#from scipy.misc import logsumexp


import torch
import torch.utils.data
import torch.nn as nn
from torch.nn import Linear
from torch.autograd import Variable
from torch.nn.functional import normalize

from utils.distributions import log_Bernoulli, log_Normal_diag, log_Normal_standard, log_Logistic_256, log_Softmax
from utils.nn import he_init, GatedDense, NonLinear

from models.Model import Model
import os


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Pytorch implementation of the baseline: Mult-VAE (https://arxiv.org/abs/1802.05814)
#=======================================================================================================================
class VAE(Model):
    def __init__(self, args):
        super(VAE, self).__init__(args)

        # encoder: q(z | x)
        #input size -> hidden_size
        modules = [nn.Dropout(p=0.5),
                   NonLinear(np.prod(self.args.input_size), self.args.hidden_size, gated=self.args.gated, activation=nn.Tanh())]
        #hidden size * (layer 갯수 -1)
        for _ in range(0, self.args.num_layers - 1):
            modules.append(NonLinear(self.args.hidden_size, self.args.hidden_size, gated=self.args.gated, activation=nn.Tanh()))
        self.q_z_layers = nn.Sequential(*modules)


        self.q_z_mean = Linear(self.args.hidden_size, self.args.z1_size)
        self.q_z_logvar = NonLinear(self.args.hidden_size, self.args.z1_size, activation=nn.Hardtanh(min_val=-12.,max_val=4.))

        # decoder: p(x | z)
        modules = [NonLinear(self.args.z1_size, self.args.hidden_size, gated=self.args.gated, activation=nn.Tanh())]
        for _ in range(0, self.args.num_layers - 1):
            modules.append(NonLinear(self.args.hidden_size, self.args.hidden_size, gated=self.args.gated, activation=nn.Tanh()))
        self.p_x_layers = nn.Sequential(*modules)
        #수정
        self.q_z1_layers_x = nn.Sequential(
            nn.Dropout(p=0.5),
            NonLinear(np.prod(self.args.input_size), self.args.hidden_size, gated=self.args.gated, activation=nn.Tanh())
        )
        # 수정
        modules = [
            NonLinear(2 * self.args.hidden_size, self.args.hidden_size, gated=self.args.gated, activation=nn.Tanh()), ]
        for _ in range(0, self.args.num_layers - 2):
            modules.append(
                NonLinear(self.args.hidden_size, self.args.hidden_size, gated=self.args.gated, activation=nn.Tanh()))
        self.q_z1_layers_joint = nn.Sequential(*modules)

        self.p_x_mean = NonLinear(self.args.hidden_size, np.prod(self.args.input_size), activation=None)

        self.epoch = -1
        self.batch_idx = -1
        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                he_init(m)

    # AUXILIARY METHODS
    #input x 들어옴
    def calculate_loss(self, x, beta=1., average=False,epoch = -1 , batch_idx = -1):
        '''
        :param x: input image(s)
        :param beta: a hyperparam for warmup
        :param average: whether to average loss or not
        :return: value of a loss function
        '''
        self.epoch = epoch
        self.batch_idx = batch_idx
        # pass through VAE
        x_mean, x_logvar, z_q, z_q_mean, z_q_logvar = self.forward(x)

        # RE
        RE = log_Softmax(x, x_mean, dim=1) #! Actually not Reconstruction Error but Log-Likelihood

        # KL
        log_p_z = self.log_p_z(z_q)
        log_q_z = log_Normal_diag(z_q, z_q_mean, z_q_logvar, dim=1)
        KL = -(log_p_z - log_q_z)
        # reconstruction error - kl
        loss = - RE + beta * KL

        if average:
            loss = torch.mean(loss)
            RE = torch.mean(RE)
            KL = torch.mean(KL)

        return loss, RE, KL

    # ADDITIONAL METHODS
    def reconstruct_x(self, x):
        x_mean, _, _, _, _ = self.forward(x)
        return x_mean

    # THE MODEL: VARIATIONAL POSTERIOR
    def q_z(self, x):
        x = self.q_z_layers(x)

        z_q_mean = self.q_z_mean(x)
        z_q_logvar = self.q_z_logvar(x)

        return z_q_mean, z_q_logvar

    # THE MODEL: GENERATIVE DISTRIBUTION
    # 수정
    def p_x(self,x,z):
        # 수정
        x = self.q_z1_layers_x(x)
        z = self.p_x_layers(z)
        # 수정
        h = torch.cat((x, z), 1)
        h = self.q_z1_layers_joint(h)
        if self.epoch%20 == 1:
            additionalTestPath = "/home/ryuck/0014_junghan/additionalResearch/array"
            xArray = np.array(x.tolist())
            np.save(os.path.join(additionalTestPath, "xArray"+str(self.epoch)+"_"+str(self.batch_idx)+".npy"), xArray)

            z2Array = np.array(z.tolist())
            np.save(os.path.join(additionalTestPath, "zArray"+str(self.epoch)+"_"+str(self.batch_idx)+".npy"), z2Array)

            hArray = np.array(h.tolist())
            np.save(os.path.join(additionalTestPath, "hArray"+str(self.epoch)+"_"+str(self.batch_idx)+".npy"), hArray)

        x_mean = self.p_x_mean(h)
        x_logvar = 0.

        return x_mean, x_logvar

    # the prior
    def log_p_z(self, z):
        # standard normal prior
        log_prior = log_Normal_standard(z, dim=1)

        return log_prior

    # THE MODEL: FORWARD PASS
    def forward(self, x):
        # input normalization & dropout
        # x 정규화
        x = normalize(x, dim=1)

        # z ~ q(z | x)
        z_q_mean, z_q_logvar = self.q_z(x)
        z_q = self.reparameterize(z_q_mean, z_q_logvar) #! train/test distinction -> built into reparameterize function

        # x_mean = p(x|z)
        # 수정
        x_mean, x_logvar = self.p_x(x,z_q)

        return x_mean, x_logvar, z_q, z_q_mean, z_q_logvar
