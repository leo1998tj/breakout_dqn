import math
import random
import numpy as np
import os
from collections import deque
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import model as m
from atari_wrappers import wrap_deepmind, make_atari
import torch.multiprocessing as mp
from torch.autograd import Variable
from torch.distributions import Categorical
import scipy.linalg as sp
from sklearn.utils.extmath import randomized_svd
import warnings

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

os.environ['OMP_NUM_THREADS'] = '1'
warnings.simplefilter("ignore", UserWarning)

# 1. GPU settings
os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # if gpu is to be used

# 3. environment reset
env_name = 'Breakout'
#env_name = 'SpaceInvaders'
# env_name = 'Riverraid'
#env_name = 'Seaquest'
#env_name = 'MontezumaRevenge'
env_raw = make_atari('{}NoFrameskip-v4'.format(env_name))
env = wrap_deepmind(env_raw, frame_stack=False, episode_life=True, clip_rewards=True)

c,h,w = m.fp(env.reset()).shape
n_actions = env.action_space.n
print(n_actions)

# 4. Network reset
policy_net = m.DQN(h, w, n_actions, device).to(device)
target_net = m.DQN(h, w, n_actions, device).to(device)
policy_net.apply(policy_net.init_weights)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# 5. DQN hyperparameters
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1.
EPS_END = 0.1
EPS_DECAY = 1000000
TARGET_UPDATE = 10000
NUM_STEPS = 50000000
M_SIZE = 1000000
POLICY_UPDATE = 4
EVALUATE_FREQ = 200000
optimizer = optim.Adam(policy_net.parameters(), lr=0.0000625, eps=1.5e-4)

def cal_fi_score(state, shared_model, epsilon = 1e-3):
    state = Variable(state, requires_grad=True)
    prob = shared_model(state)
    logp = F.log_softmax(prob, dim=-1)
    a = torch.argmax(logp, axis=1)
    dist = Categorical(logp)
    log_prob = dist.log_prob(a)
    log_prob2 = torch.log(prob[:,0])
    log_prob3 = torch.log(prob[:,1])
    log_prob4 = torch.log(prob[:, 2])
    log_prob5 = torch.log(prob[:, 3])
    loss = torch.sum(log_prob)
    loss2 = torch.sum(log_prob2)
    loss3 = torch.sum(log_prob3)
    loss4 = torch.sum(log_prob4)
    loss5 = torch.sum(log_prob5)
    loss.backward(retain_graph = True)
    grad = state.grad.cpu().numpy().copy()
    #梯度清零
    state.grad.zero_()
    loss2.backward(retain_graph = True)
    grad2 = state.grad.cpu().numpy().copy()
    state.grad.zero_()

    loss3.backward(retain_graph=True)
    grad3 = state.grad.cpu().numpy().copy()
    state.grad.zero_()

    loss4.backward(retain_graph=True)
    grad4 = state.grad.cpu().numpy().copy()
    state.grad.zero_()

    loss5.backward(retain_graph=True)
    grad5 = state.grad.cpu().numpy().copy()
    state.grad.zero_()

    logits = logp.detach().cpu().numpy()
    grad = grad.reshape(1, -1)
    grad2 = grad2.reshape(1, -1)
    grad3 = grad3.reshape(1, -1)
    grad4 = grad4.reshape(1, -1)
    grad5 = grad5.reshape(1, -1)
    grad_mat = np.array([grad2, grad3, grad4, grad5])
    res = []
    for i in range(len(grad)):
        # 对应的概率
        L0 = grad_mat[:, i, :].T @ np.diag(((logits[i, :]) ** 0.5 + 1e-10) ** -1)
        # L0 = grad_mat[:, i, :].T * (logits[i] ** 0.5)  old 计算方式
        # B0, D_L0, A0 = sp.svd(L0, full_matrices=False)
        B0, D_L0, A0 = randomized_svd(L0, n_components=3, n_iter=5)
        rank_L0 = sum(D_L0 > epsilon)
        if rank_L0 > 0:
            B0 = B0[:, :rank_L0]
            A0 = np.diag(D_L0[:rank_L0]) @ A0[:rank_L0, :]

            # U_A, D_0, _ = sp.svd(A0 @ A0.T, full_matrices=True)
            U_A, D_0, _ = randomized_svd(A0 @ A0.T, n_components=3, n_iter=5)
            D_0_inv = np.diag(D_0 ** -1)
            D_0_inv_sqrt = np.diag(D_0 ** -0.5)
            U_0 = B0 @ U_A

            nabla_f = grad[i] @ U_0 @ D_0_inv_sqrt
            FI = nabla_f @ nabla_f.T
            res.append(FI)
        res = np.array(res)

    return res

# replay memory and action selector
memory = m.ReplayMemory(M_SIZE, [5,h,w], n_actions, device)
sa = m.ActionSelector(EPS_START, EPS_END, policy_net, EPS_DECAY, n_actions, device)

steps_done = 0

def optimize_model(train):
    if not train:
        return
    state_batch, action_batch, reward_batch, n_state_batch, done_batch = memory.sample(BATCH_SIZE)

    q = policy_net(state_batch).gather(1, action_batch)
    nq = target_net(n_state_batch).max(1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = (nq * GAMMA)*(1.-done_batch[:,0]) + reward_batch[:,0]

    # Compute Huber loss
    loss = F.smooth_l1_loss(q, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def evaluate(step, policy_net, device, env, n_actions, eps=0.05, num_episode=5):
    env = wrap_deepmind(env)
    sa = m.ActionSelector(eps, eps, policy_net, EPS_DECAY, n_actions, device)
    e_rewards = []
    q = deque(maxlen=5)
    for i in range(num_episode):
        env.reset()
        e_reward = 0
        for _ in range(10): # no-op
            n_frame, _, done, _ = env.step(0)
            n_frame = m.fp(n_frame)
            q.append(n_frame)

        while not done:
            state = torch.cat(list(q))[1:].unsqueeze(0)
            action, eps = sa.select_action(state, train)
            n_frame, reward, done, info = env.step(action)
            n_frame = m.fp(n_frame)
            q.append(n_frame)
            
            e_reward += reward
        e_rewards.append(e_reward)

    f = open("file.txt",'a') 
    f.write("%f, %d, %d\n" % (float(sum(e_rewards))/float(num_episode), step, num_episode))
    f.close()

q = deque(maxlen=5)
done = True
eps = 0
episode_len = 0

progressive = tqdm(range(NUM_STEPS), total=NUM_STEPS, ncols=50, leave=False, unit='b')
for step in progressive:
    if done: # life reset !!!
        env.reset()
        sum_reward = 0
        episode_len = 0
        img, _, _, _ = env.step(1) # BREAKOUT specific !!!
        for i in range(10): # no-op
            n_frame, _, _, _ = env.step(0)
            n_frame = m.fp(n_frame)
            q.append(n_frame)
        
    train = len(memory) > 50000
    # Select and perform an action
    state = torch.cat(list(q))[1:].unsqueeze(0)
    action, eps = sa.select_action(state, train)
    n_frame, reward, done, info = env.step(action)
    n_frame = m.fp(n_frame)

    # 5 frame as memory
    q.append(n_frame)
    model_net = sa.get_model()
    try:
        fi_score = cal_fi_score(state, model_net)
    except:
        fi_score = 1
    memory.push(torch.cat(list(q)).unsqueeze(0), action, reward, done) # here the n_frame means next frame from the previous time step
    if fi_score < 0.5:
        memory.push(torch.cat(list(q)).unsqueeze(0), action, reward,
                    done)
    episode_len += 1


    # Perform one step of the optimization (on the target network)
    if step % POLICY_UPDATE == 0:
        optimize_model(train)
    
    # Update the target network, copying all weights and biases in DQN
    if step % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
    
    if step % EVALUATE_FREQ == 0:
        evaluate(step, policy_net, device, env_raw, n_actions, eps=0.05, num_episode=15)


