from __future__ import print_function
import warnings
import torch, os, gym, time, glob, argparse, sys
import numpy as np
from scipy.signal import lfilter
# from scipy.misc import imresize # preserves single-pixel info _unlike_ img = img[::2,::2]
import cv2
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.autograd import Variable
from torch.distributions import Categorical
import scipy.linalg as sp
from sklearn.utils.extmath import randomized_svd

from test import NNPolicy, SharedAdam, printlog, cost_func
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

os.environ['OMP_NUM_THREADS'] = '1'
warnings.simplefilter("ignore", UserWarning)

discount = lambda x, gamma: lfilter([1], [1, -gamma], x[::-1])[::-1]  # discounted rewards one liner
prepro = lambda img: cv2.resize(img[35:195].mean(2), (80, 80)).astype(np.float32).reshape(1, 80, 80) / 255.
def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env', default='Breakout-v4', type=str, help='gym environment')
    parser.add_argument('--processes', default=20, type=int, help='number of processes to train with')
    parser.add_argument('--render', default=False, type=bool, help='renders the atari environment')
    parser.add_argument('--test', default=True, type=bool, help='sets lr=0, chooses most likely actions')
    parser.add_argument('--rnn_steps', default=20, type=int, help='steps to train LSTM over')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--seed', default=1, type=int, help='seed random # generators (for reproducibility)')
    parser.add_argument('--gamma', default=0.99, type=float, help='rewards discount factor')
    parser.add_argument('--tau', default=1.0, type=float, help='generalized advantage estimation discount')
    parser.add_argument('--horizon', default=0.99, type=float, help='horizon for running averages')
    parser.add_argument('--hidden', default=256, type=int, help='hidden size of GRU')
    return parser.parse_args()

def train(shared_model, shared_optimizer, rank, args, info):
    env = gym.make(args.env)  # make a local (unshared) environment
    env.seed(args.seed + rank);
    torch.manual_seed(args.seed + rank)  # seed everything
    model = NNPolicy(channels=1, memsize=args.hidden, num_actions=args.num_actions)  # a local/unshared model
    state = torch.tensor(prepro(env.reset()))  # get first state

    start_time = last_disp_time = time.time()
    episode_length, epr, eploss, done = 0, 0, 0, True  # bookkeeping

    while info['frames'][0] <= 8e7 or args.test:  # openai baselines uses 40M frames...we'll use 80M
        model.load_state_dict(shared_model.state_dict())  # sync with shared model

        hx = torch.zeros(1, 256) if done else hx.detach()  # rnn activation vector
        values, logps, actions, rewards = [], [], [], []  # save values for computing gradientss

        for step in range(args.rnn_steps):
            episode_length += 1
            value, logit, hx = model((state.view(1, 1, 80, 80), hx))
            logp = F.log_softmax(logit, dim=-1)

            action = torch.exp(logp).multinomial(num_samples=1).data[0]  # logp.max(1)[1].data if args.test else
            state, reward, done, _ = env.step(action.numpy()[0])
            if args.render: env.render()

            state = torch.tensor(prepro(state));
            epr += reward
            reward = np.clip(reward, -1, 1)  # reward
            done = done or episode_length >= 1e4  # don't playing one ep for too long

            info['frames'].add_(1);
            num_frames = int(info['frames'].item())
            if num_frames % 2e6 == 0:  # save every 2M frames
                printlog(args, '\n\t{:.0f}M frames: saved model\n'.format(num_frames / 1e6))
                torch.save(shared_model.state_dict(), args.save_dir + 'model.{:.0f}.tar'.format(num_frames / 1e6))

            if done:  # update shared data
                info['episodes'] += 1
                interp = 1 if info['episodes'][0] == 1 else 1 - args.horizon
                info['run_epr'].mul_(1 - interp).add_(interp * epr)
                info['run_loss'].mul_(1 - interp).add_(interp * eploss)

            if rank == 0 and time.time() - last_disp_time > 60:  # print info ~ every minute
                elapsed = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time))
                printlog(args, 'time {}, episodes {:.0f}, frames {:.1f}M, mean episode rewards {:.2f}, run loss {:.2f}'
                         .format(elapsed, info['episodes'].item(), num_frames / 1e6,
                                 info['run_epr'].item(), info['run_loss'].item()))
                last_disp_time = time.time()

            if done:  # maybe print info.
                episode_length, epr, eploss = 0, 0, 0
                state = torch.tensor(prepro(env.reset()))

            values.append(value);
            logps.append(logp);
            actions.append(action);
            rewards.append(reward)

        next_value = torch.zeros(1, 1) if done else model((state.unsqueeze(0), hx))[0]
        values.append(next_value.detach())

        loss = cost_func(args, torch.cat(values), torch.cat(logps), torch.cat(actions), np.asarray(rewards))
        eploss += loss.item()
        shared_optimizer.zero_grad();
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 40)
        torch.save(model.state_dict(), "model_paras.pt".format())
        for param, shared_param in zip(model.parameters(), shared_model.parameters()):
            if shared_param.grad is None: shared_param._grad = param.grad  # sync gradients with shared model
        shared_optimizer.step()


def cal_fi_score(state, shared_model, hx, epsilon = 1e-3):
    a2 = time.time()
    state = Variable(state, requires_grad=True)
    value, prob, hx = shared_model((state.view(1, 1, 80, 80), hx))
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

    logits = prob.detach().cpu().numpy()
    grad = grad.reshape(1, -1)
    grad2 = grad2.reshape(1, -1)
    grad3 = grad3.reshape(1, -1)
    grad4 = grad4.reshape(1, -1)
    grad5 = grad5.reshape(1, -1)
    grad_mat = np.array([grad2, grad3, grad4, grad5])
    b = time.time()
    res = []
    # print(grad_mat)
    for i in range(len(grad)):
        # 对应的概率
        c = time.time()
        L0 = grad_mat[:, i, :].T @ np.diag(((logits[i, :]) ** 0.5 + 1e-10) ** -1)
        # L0 = grad_mat[:, i, :].T * (logits[i] ** 0.5)  old 计算方式
        B0, D_L0, A0 = sp.svd(L0, full_matrices=False)
        # B0, D_L0, A0 = randomized_svd(L0, n_components=1, n_iter=2)
        # rank_L0 = sum(D_L0 > epsilon)
        rank_L0 = 1
        # if rank_L0 > 0:
        B0 = B0[:, :rank_L0]
        A0 = np.diag(D_L0[:rank_L0]) @ A0[:rank_L0, :]

        d = time.time()

        U_A, D_0, _ = sp.svd(A0 @ A0.T, full_matrices=True)
        # U_A, D_0, _ = randomized_svd(A0 @ A0.T, n_components=1, n_iter=5)
        D_0_inv = np.diag(D_0 ** -1)
        D_0_inv_sqrt = np.diag(D_0 ** -0.5)
        U_0 = B0 @ U_A

        nabla_f = grad[i] @ U_0 @ D_0_inv_sqrt
        FI = nabla_f @ nabla_f.T
        res.append(FI)
        res = np.array(res)
        e = time.time()
        print(b-a2, c-b, d-c, e-d)
    return res

if __name__ == "__main__":
    if sys.version_info[0] > 2:
        mp.set_start_method('spawn')  # this must not be in global scope
    elif sys.platform == 'linux' or sys.platform == 'linux2':
        raise "Must be using Python 3 with linux!"  # or else you get a deadlock in conv2d

    args = get_args()
    args.save_dir = '{}/'.format(args.env.lower())  # 设置保存路径
    if args.render:  args.processes = 1; args.test = True  # 是否展示
    if args.test:  args.lr = 0  # 是否测试
    args.num_actions = gym.make(args.env).action_space.n  # 获取环境输出
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None  # 测试是否有文件夹存在

    torch.manual_seed(args.seed)  # 随机一个随机数
    shared_model = NNPolicy(channels=1, memsize=args.hidden, num_actions=args.num_actions).share_memory() # 模型
    shared_optimizer = SharedAdam(shared_model.parameters(), lr=args.lr)

    info = {k: torch.DoubleTensor([0]).share_memory_() for k in ['run_epr', 'run_loss', 'episodes', 'frames']}
    info['frames'] += shared_model.try_load(args.save_dir) * 1e6
    if int(info['frames'].item()) == 0: printlog(args, '', end='', mode='w')  # clear log file

    # train(shared_model, shared_optimizer, 0, args, info)
    # cal_fi_score(args, shared_model)

    env = gym.make(args.env)  # make a local (unshared) environment
    episode_length = 0
    model = NNPolicy(channels=1, memsize=args.hidden, num_actions=args.num_actions)  # a local/unshared model
    model.load_state_dict(shared_model.state_dict())
    state = torch.tensor(prepro(env.reset()))  # get first state
    hx = torch.zeros(1, 256)
    a = time.time()
    for step in range(20):
        episode_length += 1
        value, logit, hx = model((state.view(1, 1, 80, 80), hx))
        logp = F.log_softmax(logit, dim=-1)
        action = torch.exp(logp).multinomial(num_samples=1).data[0]  # logp.max(1)[1].data if args.test else
        state, reward, done, _ = env.step(action.numpy()[0])
        env.render()
        state = torch.tensor(prepro(state))
        print(cal_fi_score(state, model, hx))
    # print(time.time() - a)
