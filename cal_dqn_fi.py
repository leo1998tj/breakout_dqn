from dataclasses import dataclass
import os
# import ray
import matplotlib.pyplot as plt
import numpy as np
import pickle
import zlib
import tensorflow as tf


@dataclass
class Experience:
    state: np.ndarray

    action: float

    reward: float

    next_state: np.ndarray

    done: bool


class ReplayBuffer:

    def __init__(self, max_len, compress=True):

        self.max_len = max_len

        self.buffer = []

        self.compress = compress

        self.count = 0

    def __len__(self):
        return len(self.buffer)

    def push(self, exp):

        if self.compress:
            exp = zlib.compress(pickle.dumps(exp))

        if self.count == self.max_len:
            self.count = 0

        try:
            self.buffer[self.count] = exp
        except IndexError:
            self.buffer.append(exp)

        self.count += 1

    def get_minibatch(self, batch_size):

        N = len(self.buffer)

        indices = np.random.choice(
            np.arange(N), replace=False, size=batch_size)

        if self.compress:
            selected_experiences = [
                pickle.loads(zlib.decompress(self.buffer[idx])) for idx in indices]
        else:
            selected_experiences = [self.buffer[idx] for idx in indices]

        states = np.vstack(
            [exp.state for exp in selected_experiences]).astype(np.float32)

        actions = np.vstack(
            [exp.action for exp in selected_experiences]).astype(np.float32)

        rewards = np.array(
            [exp.reward for exp in selected_experiences]).reshape(-1, 1)

        next_states = np.vstack(
            [exp.next_state for exp in selected_experiences]
        ).astype(np.float32)

        dones = np.array(
            [exp.done for exp in selected_experiences]).reshape(-1, 1)

        return (states, actions, rewards, next_states, dones)


import numpy as np

import tensorflow.keras.layers as kl


class CategoricalQNet(tf.keras.Model):

    def __init__(self, actions_space, n_atoms, Z):

        super(CategoricalQNet, self).__init__()

        self.action_space = actions_space

        self.n_atoms = n_atoms

        self.Z = Z

        self.conv1 = kl.Conv2D(32, 8, strides=4, activation="relu",
                               kernel_initializer="he_normal")
        self.conv2 = kl.Conv2D(64, 4, strides=2, activation="relu",
                               kernel_initializer="he_normal")
        self.conv3 = kl.Conv2D(64, 3, strides=1, activation="relu",
                               kernel_initializer="he_normal")

        self.flatten1 = kl.Flatten()
        self.dense1 = kl.Dense(512, activation="relu",
                               kernel_initializer="he_normal")
        self.logits = kl.Dense(self.action_space * self.n_atoms,
                               kernel_initializer="he_normal")

    tf.config.experimental_run_functions_eagerly(True)

    @tf.function
    def call(self, x):

        batch_size = x.shape[0]

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.flatten1(x)
        x = self.dense1(x)

        logits = self.logits(x)
        logits = tf.reshape(logits, (batch_size, self.action_space, self.n_atoms))
        probs = tf.nn.softmax(logits, axis=2)

        return probs

    def sample_action(self, x, epsilon=None):

        if (epsilon is None) or (np.random.random() > epsilon):
            selected_actions, _ = self.sample_actions(x)
            selected_action = selected_actions[0][0].numpy()
        else:
            selected_action = np.random.choice(self.action_space)

        return selected_action

    def sample_actions(self, x):
        probs = self(x)
        q_means = tf.reduce_sum(probs * self.Z, axis=2, keepdims=True)
        selected_actions = tf.argmax(q_means, axis=1)
        return selected_actions, probs


def frame_preprocess(frame):
    def _frame_preprocess(frame):
        image = tf.cast(tf.convert_to_tensor(frame), tf.float32)
        image_gray = tf.image.rgb_to_grayscale(image)
        image_crop = tf.image.crop_to_bounding_box(image_gray, 34, 0, 160, 160)
        image_resize = tf.image.resize(image_crop, [84, 84])
        image_scaled = tf.divide(image_resize, 255)
        return image_scaled

    frame = _frame_preprocess(frame).numpy()[:, :, 0]

    return frame


from pathlib import Path
import shutil

import gym
import numpy as np
import tensorflow as tf
import collections


# from model import CategoricalQNet
# from buffer import Experience, ReplayBuffer
# from util import frame_preprocess


class CategoricalDQNAgent:

    def __init__(self, env_name="BreakoutDeterministic-v4",
                 n_atoms=51, Vmin=-10, Vmax=10, gamma=0.995,
                 n_frames=4, batch_size=32, lr=0.00025,
                 init_epsilon=0.95,
                 update_period=8,
                 target_update_period=10000):

        self.env_name = env_name

        self.n_atoms = n_atoms

        self.Vmin, self.Vmax = Vmin, Vmax

        self.delta_z = (self.Vmax - self.Vmin) / (self.n_atoms - 1)

        self.Z = np.linspace(self.Vmin, self.Vmax, self.n_atoms)

        self.gamma = gamma

        self.n_frames = n_frames

        self.batch_size = batch_size

        self.init_epsilon = init_epsilon

        self.epsilon_scheduler = (lambda steps:
                                  max(0.98 * (500000 - steps) / 500000, 0.1) if steps < 500000
                                  else max(0.05 + 0.05 * (1000000 - steps) / 500000, 0.05)
                                  )

        self.update_period = update_period

        self.target_update_period = target_update_period

        env = gym.make(self.env_name)

        self.action_space = env.action_space.n

        self.qnet = CategoricalQNet(
            self.action_space, self.n_atoms, self.Z)

        self.target_qnet = CategoricalQNet(
            self.action_space, self.n_atoms, self.Z)

        self.optimizer = tf.keras.optimizers.Adam(lr=lr, epsilon=0.01 / batch_size)

    def learn(self, n_episodes, buffer_size=800000, logdir="log"):

        logdir = Path(os.path.abspath('')).parent / logdir
        if logdir.exists():
            shutil.rmtree(logdir)
        self.summary_writer = tf.summary.create_file_writer(str(logdir))

        self.replay_buffer = ReplayBuffer(max_len=buffer_size)

        steps = 0
        for episode in range(1, n_episodes + 1):
            env = gym.make(self.env_name)

            frames = collections.deque(maxlen=4)
            frame = frame_preprocess(env.reset())
            for _ in range(self.n_frames):
                frames.append(frame)

            #: ネットワーク重みの初期化
            state = np.stack(frames, axis=2)[np.newaxis, ...]
            self.qnet(state)
            self.target_qnet(state)
            self.target_qnet.set_weights(self.qnet.get_weights())

            episode_rewards = 0
            episode_steps = 0

            done = False
            lives = 5
            while not done:

                steps += 1
                episode_steps += 1

                epsilon = self.epsilon_scheduler(steps)

                state = np.stack(frames, axis=2)[np.newaxis, ...]
                action = self.qnet.sample_action(state, epsilon=epsilon)
                next_frame, reward, done, info = env.step(action)
                episode_rewards += reward
                frames.append(frame_preprocess(next_frame))
                next_state = np.stack(frames, axis=2)[np.newaxis, ...]

                if done:
                    exp = Experience(state, action, reward, next_state, done)
                    self.replay_buffer.push(exp)
                    break
                else:
                    if info["ale.lives"] != lives:
                        lives = info["ale.lives"]
                        exp = Experience(state, action, reward, next_state, True)
                    else:
                        exp = Experience(state, action, reward, next_state, done)

                    self.replay_buffer.push(exp)

                if (len(self.replay_buffer) > 20000) and (steps % self.update_period == 0):
                    loss = self.update_network()

                    with self.summary_writer.as_default():
                        tf.summary.scalar("loss", loss, step=steps)
                        tf.summary.scalar("epsilon", epsilon, step=steps)
                        tf.summary.scalar("buffer_size", len(self.replay_buffer), step=steps)
                        tf.summary.scalar("train_score", episode_rewards, step=steps)
                        tf.summary.scalar("train_steps", episode_steps, step=steps)

                #: Hard target update
                if steps % self.target_update_period == 0:
                    self.target_qnet.set_weights(self.qnet.get_weights())

            print(f"Episode: {episode}, score: {episode_rewards}, steps: {episode_steps}")

            if episode % 20 == 0:
                test_scores, test_steps = self.test_play(n_testplay=1)
                with self.summary_writer.as_default():
                    tf.summary.scalar("test_score", test_scores[0], step=steps)
                    tf.summary.scalar("test_step", test_steps[0], step=steps)

            if episode % 1000 == 0:
                print("Model Saved")
                self.qnet.save_weights("checkpoints/qnet")

    def update_network(self):

        (states, actions, rewards,
         next_states, dones) = self.replay_buffer.get_minibatch(self.batch_size)

        next_actions, next_probs = self.target_qnet.sample_actions(next_states)

        onehot_mask = self.create_mask(next_actions)
        next_dists = tf.reduce_sum(next_probs * onehot_mask, axis=1).numpy()

        target_dists = self.shift_and_projection(rewards, dones, next_dists)

        onehot_mask = self.create_mask(actions)
        with tf.GradientTape() as tape:
            probs = self.qnet(states)

            dists = tf.reduce_sum(probs * onehot_mask, axis=1)

            dists = tf.clip_by_value(dists, 1e-6, 1.0)

            loss = tf.reduce_sum(
                -1 * target_dists * tf.math.log(dists), axis=1, keepdims=True)
            loss = tf.reduce_mean(loss)

        grads = tape.gradient(loss, self.qnet.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.qnet.trainable_variables))

        return loss

    def shift_and_projection(self, rewards, dones, next_dists):

        target_dists = np.zeros((self.batch_size, self.n_atoms))

        for j in range(self.n_atoms):
            tZ_j = np.minimum(self.Vmax, np.maximum(self.Vmin, rewards + self.gamma * self.Z[j]))
            bj = (tZ_j - self.Vmin) / self.delta_z

            lower_bj = np.floor(bj).astype(np.int8)
            upper_bj = np.ceil(bj).astype(np.int8)

            eq_mask = lower_bj == upper_bj
            neq_mask = lower_bj != upper_bj

            lower_probs = 1 - (bj - lower_bj)
            upper_probs = 1 - (upper_bj - bj)

            next_dist = next_dists[:, [j]]
            indices = np.arange(self.batch_size).reshape(-1, 1)

            target_dists[indices[neq_mask], lower_bj[neq_mask]] += (lower_probs * next_dist)[neq_mask]
            target_dists[indices[neq_mask], upper_bj[neq_mask]] += (upper_probs * next_dist)[neq_mask]

            target_dists[indices[eq_mask], lower_bj[eq_mask]] += (0.5 * next_dist)[eq_mask]
            target_dists[indices[eq_mask], upper_bj[eq_mask]] += (0.5 * next_dist)[eq_mask]

        for batch_idx in range(self.batch_size):

            if not dones[batch_idx]:
                continue
            else:
                target_dists[batch_idx, :] = 0
                tZ = np.minimum(self.Vmax, np.maximum(self.Vmin, rewards[batch_idx]))
                bj = (tZ - self.Vmin) / self.delta_z

                lower_bj = np.floor(bj).astype(np.int32)
                upper_bj = np.ceil(bj).astype(np.int32)

                if lower_bj == upper_bj:
                    target_dists[batch_idx, lower_bj] += 1.0
                else:
                    target_dists[batch_idx, lower_bj] += 1 - (bj - lower_bj)
                    target_dists[batch_idx, upper_bj] += 1 - (upper_bj - bj)

        return target_dists

    def create_mask(self, actions):

        mask = np.ones((self.batch_size, self.action_space, self.n_atoms))
        actions_onehot = tf.one_hot(
            tf.cast(actions, tf.int32), self.action_space, axis=1)

        for idx in range(self.batch_size):
            mask[idx, ...] = mask[idx, ...] * actions_onehot[idx, ...]

        return mask

    def test_play(self, n_testplay=1, monitor_dir=None,
                  checkpoint_path=None):

        if checkpoint_path:
            env = gym.make(self.env_name)
            frames = collections.deque(maxlen=4)
            frame = frame_preprocess(env.reset())
            for _ in range(self.n_frames):
                frames.append(frame)
            state = np.stack(frames, axis=2)[np.newaxis, ...]
            self.qnet(state)
            self.qnet.load_weights(checkpoint_path)

        if monitor_dir:
            monitor_dir = Path(monitor_dir)
            if monitor_dir.exists():
                shutil.rmtree(monitor_dir)
            monitor_dir.mkdir()
            env = gym.wrappers.Monitor(
                gym.make(self.env_name), monitor_dir, force=True,
                video_callable=(lambda ep: True))
        else:
            env = gym.make(self.env_name)

        scores = []
        steps = []
        for _ in range(n_testplay):

            frames = collections.deque(maxlen=4)

            frame = frame_preprocess(env.reset())
            for _ in range(self.n_frames):
                frames.append(frame)

            done = False
            episode_steps = 0
            episode_rewards = 0

            while not done:
                state = np.stack(frames, axis=2)[np.newaxis, ...]
                action = self.qnet.sample_action(state, epsilon=0.1)
                next_frame, reward, done, info = env.step(action)
                frames.append(frame_preprocess(next_frame))

                episode_rewards += reward
                episode_steps += 1
                if episode_steps > 500 and episode_rewards < 3:
                    break

            scores.append(episode_rewards)
            steps.append(episode_steps)

        return scores, steps


def main():
    agent = CategoricalDQNAgent()
    agent.learn(n_episodes=6001)


if __name__ == '__main__':
    main()

Vmin = -10
Vmax = 10
n_atoms = 51
Z = np.linspace(Vmin, Vmax, n_atoms)
breakout = CategoricalQNet(18, 51, Z)
#
# breakout.load_weights(‘ / home / jupyt / suiyang / Asteroids / 1111 / checkpoints / qnet
# ')

import gym
import collections

env_name = "BreakoutDeterministic-v4"
env = gym.make(env_name)

import gym
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


def sample_action(state, policy):
    probs = policy(state)
    q_means = tf.reduce_sum(probs * Z, axis=2, keepdims=True)
    selected_actions = tf.argmax(q_means, axis=1)
    selected_action = selected_actions[0][0].numpy()

    # if (epsilon is None) or (np.random.random() > epsilon):
    #    selected_action = selected_actions[0][0].numpy()
    # else:
    #    selected_action = np.random.choice(4)

    return selected_action


def generate_trajectory(policy, num_iter):
    eps = 0.95

    states = []
    actions = []
    rewards = []
    next_states = []
    score = 0

    frames = collections.deque(maxlen=4)
    frame = frame_preprocess(env.reset())

    for _ in range(4):
        frames.append(frame)
    state = np.stack(frames, axis=2)[np.newaxis, ...]

    for i in range(num_iter):

        ac = sample_action(state, policy)
        next_frame, reward, done, info = env.step(ac)

        frames.append(frame_preprocess(next_frame))
        next_state = np.stack(frames, axis=2)[np.newaxis, ...]

        # next_state, reward, done, _ = env.step(ac)
        states.append(state)
        next_states.append(next_state)
        # 独热编码
        actions.append([int(k == ac) for k in range(18)])
        rewards.append(reward)
        state = next_state
        score += reward
        # eps=max(0.98 * (1000 - i) / 1000, 0.1)

        if done:
            break

    return states, actions, rewards, next_states, score


def discount_reward(rewards, gamma, causal=True):
    n = len(rewards)
    dis_rewards = rewards.copy()
    if causal == False:
        for i in range(1, n):
            dis_rewards[i] += gamma * dis_rewards[i - 1]
        dis_rewards = len(dis_rewards) * [dis_rewards[-1]]
        dis_rewards = np.array(dis_rewards)
        return dis_rewards
    elif causal == True:
        for i in range(n - 1, 0, -1):
            dis_rewards[i - 1] += dis_rewards[i] * gamma
        dis_rewards = np.array(dis_rewards)
        # 归一化奖励
        return (dis_rewards - np.mean(dis_rewards)) / (np.std(dis_rewards))


import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

# from common import generate_trajectory, choose_action


round = 1
num_iter = 3000
# model_path = r'E:/RL不确定性推理/CartPole/model/AC_TD.pth'
# save_path = r'E:/RL不确定性推理/CartPole/dataset/AC_TD.npy'
# state=test_states[1388,...]
# +np.random.normal(0, 0.01, (1,84,84,4))

for i in range(round):
    states, actions, rewards, next_states, score = generate_trajectory(breakout, num_iter)
    states = np.array(states)
    next_states = np.array(next_states)
    actions = np.array(actions)
    reward = np.array(rewards)
    score = np.array(score)
    if i == 0:
        test_states1 = states
    else:
        test_states1 = np.vstack([test_states1, states])
    if i == 0:
        test_next_states1 = next_states
    else:
        test_next_states1 = np.vstack([test_next_states1, next_states])
    if i == 0:
        test_actions1 = actions
    else:
        test_actions1 = np.vstack([test_actions1, actions])

    if i == 0:
        test_reward1 = reward
    else:
        test_reward1 = np.vstack([test_reward1, reward])

    print(score)
    print(test_reward1.shape)

##np.save(save_path, result)


# FI0
from scipy import linalg
import time
import scipy.linalg as sp

res = []
epsilon = 1e-10
for t in range():  # length of trajectory

    time_start = time.time()
    w = tf.Variable(test_states1[t, ...], trainable=True)
    with tf.GradientTape() as tape:
        ac = test_actions1[t, :].argmax()
        prob = (breakout(w))[0, ac, :]
        q_means = tf.reduce_sum(prob * Z)
    grad_prob = tape.gradient(q_means, w)

    grad = {}
    for i in range(51):
        with tf.GradientTape() as tape:
            ac = test_actions1[t, :].argmax()
            loss = tf.math.log(breakout(w))[0, ac, i]
        grad[i] = tape.gradient(loss, w)

    p0 = np.zeros((51, 1))
    ac0 = test_actions1[t, :].argmax()
    for i in range(51):
        p0[i, 0] = breakout(w)[0, ac0, i].numpy()

    # w1 = tf.Variable(test_next_states1[t,...], trainable=True)
    # ac1=test_actions1[t+1,:].argmax()
    # p1=np.zeros((51,1))
    # for i in range(50):
    #    p1[i,0]=breakout(w1)[0,ac1,i].numpy()

    # L0=np.zeros((1,84*84))
    for i in range(51):
        d = (grad[i][0, ...].numpy().mean(2)).reshape((84 * 84, 1))
        if i == 0:
            L0 = d
        else:
            L0 = np.hstack((L0, d))

    F = np.zeros((1, 84 * 84))
    F = (grad_prob[0, ...].numpy().mean(2)).reshape((1, 84 * 84))

    B0, D_L0, A0 = sp.svd(L0, full_matrices=False)
    rank_L0 = sum(D_L0 > epsilon)
    if rank_L0 > 0:
        B0 = B0[:, :rank_L0]
        A0 = np.diag(D_L0[:rank_L0]) @ A0[:rank_L0, :]
        U_A, D_0, _ = sp.svd(A0 @ A0.T, full_matrices=True)
        D_0_inv = np.diag(D_0 ** -1)
        D_0_inv_sqrt = np.diag(D_0 ** -0.5)
        U_0 = B0 @ U_A

    nabla_f = F @ U_0 @ D_0_inv_sqrt
    fi = nabla_f @ nabla_f.T

    # fi=(-F/(1e+3))@linalg.inv(G+1e-18*np.eye(84*84))@(-F.T/(1e+3))
    res.append(fi)
    time_end = time.time()
    time0 = time_end - time_start
    print(t, time0, fi)