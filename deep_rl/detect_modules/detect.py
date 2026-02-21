#######################################################################
# Copyright (C) 2025 Saptarshi Nath, Christos Peridis,                #
# Eseoghene Benjamin, Xinran Liu, Soheil Kolouri, Andrea Soltoggio    #
# Licensed under the Apache License, Version 2.0                      #
# http://www.apache.org/licenses/LICENSE-2.0                          #
#######################################################################

import numpy as np
import ot
import torch
#from plotting import *
from torch.utils.data import DataLoader, SubsetRandomSampler
from scipy.spatial.distance import mahalanobis, cdist, pdist, squareform
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import KernelDensity
from scipy.stats import wasserstein_distance
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import random
import torch.optim as optim
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

class Detect:
    def __init__(self, reference_num, input_dim, action_dim, num_samples, reference=200,  device='cuda', title = '', num_iter=10, one_hot=True, normalized=True, demo=True):
        assert reference is not None, f'Reference not found.'
        self.ref = None
        self.device = device
        self.num_samples = num_samples
        self.num_iter = num_iter
        self.oh = one_hot
        self.normalized = normalized
        self.demo = demo
        self.title = title
        self.input_dim = input_dim
        self.reference_num = reference_num
        self.action_dim = action_dim

        self._b = None
        self._a_cache = {}  # batch_size -> a
        
        self.embeddings = []
        self.rewards = []

    def set_input_dim(self, an_input_dim):
        '''A setter method, for manually setting and setting the input dimensionality of the detect
        module.'''
        self.input_dim = an_input_dim

    def get_input_dim(self):
        '''A getter method for accessing the input dimensionality of the detect module.'''
        return self.input_dim

    def set_reference(self, a_task_observation_dim, some_reference_num, some_action_dim):
        '''A setter method, for manually setting and updating the reference for calculating
        the tasks embeddings.'''
        torch.manual_seed(98)
        reference = torch.rand(some_reference_num, (a_task_observation_dim + some_action_dim + 1), device=self.device)#Plus one which is the reward.
        self.ref = reference.float()
        ref_size = self.ref.shape[0]
        self._b = torch.full((ref_size,), 1.0 / ref_size, device=self.device)

    def get_reference(self):
        '''A getter method for accessing the reference which is used to calculate the task
        embeddings'''
        return self.ref

    def _get_a(self, n):
        a = self._a_cache.get(n)
        if a is None:
            a = torch.full((n,), 1.0 / n, device=self.device)
            self._a_cache[n] = a
        return a
        
    def set_num_samples(self, a_num_samples):
        '''A setter method for manually setting the num of samples'''
        self.num_samples = a_num_samples

    def get_num_samples(self):
        '''A getter method for retreiving the detect sample size.'''
        return self.num_samples

    def precalculate_embedding_size(self, a_reference_num, an_inputdim, some_action_dim):
        '''A method for calculating the embedding dimension '''
        pre_calc_embedding_size = a_reference_num * (an_inputdim + some_action_dim + 1)#Plus one which is the reward.
        return pre_calc_embedding_size

    def preprocess_dataset(self, X, action_space_size):
        # X can be numpy or torch; keep it torch
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X)

        # Optional subsample (no DataLoader needed)
        if self.num_samples is not None and X.shape[0] > self.num_samples:
            idx = torch.randperm(X.shape[0], device=X.device)[:self.num_samples]
            X = X.index_select(0, idx)

        X = X.to(self.device).float()

        img = X[:, :self.input_dim]
        act = X[:, self.input_dim:-1]
        reward = X[:, -1:].view(-1, 1)

        if self.normalized:
            # (Your original normalizes globally; keeping that behavior)
            mean = img.mean()
            std = img.std().clamp_min(1e-8)
            img = (img - mean) / std

        if self.oh:
            # assume discrete actions stored as scalar in act
            act = act.view(-1).long()
            act = F.one_hot(act, num_classes=action_space_size).float().to(self.device)

        return torch.cat((img, act, reward), dim=1)

    @torch.no_grad()
    def lwe(self, X, action_space_size, reg=0.05, numItermax=2000):
        X = self.preprocess_dataset(X, action_space_size)

        # Ensure ref is on device
        ref = self.ref
        ref_size = ref.shape[0]

        # Cost matrix on GPU (torch.cdist is fast)
        C = ot.dist(X, ref, p=2)

        # torch weights on same device/dtype
        a = torch.full((X.shape[0],), 1.0 / X.shape[0], device=X.device, dtype=X.dtype)
        b = torch.full((ref_size,), 1.0 / ref_size, device=ref.device, dtype=ref.dtype)

        gamma = ot.emd(a, b, C, numItermax=700_000)

        # Barycentric projection-style map (keep your formula)
        f = (ref_size * gamma).T @ X
        f = (f - ref) / (ref_size ** 0.5)

        return f.reshape(-1)  # 1D tensor on GPU
      
    def calculate_lwes_distance(self, lwe1, lwe2):
        '''Calculates the Euclidian Distance of the old vs the new embedding
        It returns a 2D Tensor of size (num * Data-Batch sample size)'''
        eu_dist = (lwe1 - lwe2).pow(2).ravel()
        return eu_dist

    def pwdist(self, tasks_dict):
        '''Computes the Distance of the Ebeddings of Different Taks & create a similarity Matrix for all different Tasks'''
        num_tasks = len(tasks_dict)
        tasks = tasks_dict.values()
        task_ids = list(tasks_dict.keys())

        #initialize pairwise distance matrix
        dist = torch.zeros((num_tasks,num_tasks))

        for k in tqdm(range(int(self.num_iter/2))):
          task_vecs = []
          task_vecs_ = []
          #emb_list = []
          for task in tasks:
            vec = self.lwe(task)
            task_vecs.append(vec)
          for i in range(num_tasks):
            for j in range(i+1,num_tasks):
                dist[i,j]+=torch.linalg.vector_norm(task_vecs[i]-task_vecs[j])/2.
          for task in tasks:
            vec = self.lwe(task)
            task_vecs_.append(vec)
          for i in range(num_tasks):
            dist[i,i]+=torch.linalg.vector_norm(task_vecs[i]-task_vecs_[i])
          for i in range(num_tasks):
            for j in range(i+1,num_tasks):
                dist[i,j]+=torch.linalg.vector_norm(task_vecs_[i]-task_vecs_[j])/2.

    def emb_distance(self, current_embedding, new_calculated_embedding):
        '''Computes the Distance of the newlly calculated embedding and the one that is 
        stored for the current task the agent is solving.'''

        distance  = torch.linalg.vector_norm(current_embedding - new_calculated_embedding)

        return distance