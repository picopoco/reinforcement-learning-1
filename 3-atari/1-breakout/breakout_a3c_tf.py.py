import gym
import time
import random
import threading
import numpy as np
import tensorflow as tf
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import Model
from keras.layers import Dense, Flatten, Input
from keras.layers.convolutional import Conv2D
from keras import backend as K

# global variables for A3C
global episode
episode = 0
EPISODES = 8000000
# In case of BreakoutDeterministic-v3, always skip 4 frames
# Deterministic-v4 version use 4 actions
env_name = "BreakoutDeterministic-v4"

# This is A3C(Asynchronous Advantage Actor Critic) agent(global) for the Cartpole
# In this example, we use A3C algorithm
class A3CAgent:
    def __init__(self, action_size):
        # environment settings
        self.state_size = (84, 84, 4)
        self.action_size = action_size

        self.discount_factor = 0.99
        self.no_op_steps = 30

        # optimizer parameters
        self.lr = 5e-4
        self.threads = 8

        # create model for actor and critic network
        self.model = self.build_model()

        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr, decay=.99)

        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())

        self.summary_placeholders, self.update_ops, self.summary_op =\
            self.setup_summary()
        self.summary_writer = tf.summary.FileWriter('summary/breakout_a3c',
                                                    self.sess.graph)

    def train(self):
        # self.load_model("./save_model/breakout_a3c")
        agents = [Agent(self.action_size, self.state_size, self.model, self.sess, self.optimizer,
                        self.discount_factor, [self.summary_op, self.summary_placeholders,
                        self.update_ops, self.summary_writer]) for _ in range(self.threads)]

        for agent in agents:
            time.sleep(1)
            agent.start()

        while True:
            time.sleep(60*5)
            self.save_model("./save_model/breakout_a3c")

    # approximate policy and value using Neural Network
    # actor -> state is input and probability of each action is output of network
    # critic -> state is input and value of state is output of network
    # actor and critic network share first hidden layer
    def build_model(self):
        input = Input(shape=self.state_size)
        conv = Conv2D(16, (8, 8), strides=(4, 4), activation='relu')(input)
        conv = Conv2D(32, (4, 4), strides=(2, 2), activation='relu')(conv)
        conv = Flatten()(conv)
        fc = Dense(256, activation='relu')(conv)
        policy = Dense(self.action_size, activation='softmax')(fc)
        value = Dense(1, activation='linear')(fc)

        model = Model(inputs=input, outputs=[policy, value])

        model.predict(np.random.rand(1, 84, 84, 4))

        model.set_weights(model.get_weights())

        model.summary()

        return model


    def load_model(self, name):
        self.moedl.load_weights(name + ".h5")

    def save_model(self, name):
        self.model.save_weights(name + ".h5")

    # make summary operators for tensorboard
    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q = tf.Variable(0.)
        episode_duration = tf.Variable(0.)

        tf.summary.scalar('Total Reward/Episode', episode_total_reward)
        tf.summary.scalar('Average Max Prob/Episode', episode_avg_max_q)
        tf.summary.scalar('Duration/Episode', episode_duration)

        summary_vars = [episode_total_reward, episode_avg_max_q, episode_duration]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op

# make agents(local) and start training
class Agent(threading.Thread):
    def __init__(self, action_size, state_size, model, sess, optimizer, discount_factor, summary_ops):
        threading.Thread.__init__(self)

        self.action_size = action_size
        self.state_size = state_size
        self.model = model
        self.sess = sess

        self.discount_factor = discount_factor
        self.summary_op, self.summary_placeholders, self.update_ops, self.summary_writer = summary_ops

        self.states, self.actions, self.rewards = [],[],[]
        self.local_model = self.build_local()
        self.optimizer = optimizer
        self.sess.run(tf.global_variables_initializer())
        self.feed_acts, self.feed_rwds, self.feed_advs, self.update_model = \
            self.make_grads_func()

        K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())

        self.avg_p_max = 0
        self.avg_loss = 0

        # t_max -> max batch size for training
        self.t_max = 20
        self.t = 0

    # Thread interactive with environment
    def run(self):
        # self.load_model('./save_model/breakout_a3c')
        global episode

        env = gym.make(env_name)

        step = 0

        while episode < EPISODES:
            done = False
            dead = False
            # 1 episode = 5 lives
            score, start_life = 0, 5
            observe = env.reset()
            next_observe = observe

            # this is one of DeepMind's idea.
            # just do nothing at the start of episode to avoid sub-optimal
            for _ in range(random.randint(1, 30)):
                observe = next_observe
                next_observe, _, _, _ = env.step(1)

            # At start of episode, there is no preceding frame. So just copy initial states to make history
            state = pre_processing(next_observe, observe)
            history = np.stack((state, state, state, state), axis=2)
            history = np.reshape([history], (1, 84, 84, 4))

            while not done:
                step += 1
                self.t += 1
                observe = next_observe
                # get action for the current history and go one step in environment
                action, policy = self.get_action(history)
                # change action to real_action
                if action == 0: real_action = 1
                elif action == 1: real_action = 2
                else: real_action = 3
               
                if dead:
                    action, real_action = 0, 1
                    dead = False

                next_observe, reward, done, info = env.step(real_action)
                # pre-process the observation --> history
                next_state = pre_processing(next_observe, observe)
                next_state = np.reshape([next_state], (1, 84, 84, 1))
                next_history = np.append(next_state, history[:, :, :, :3], axis=3)

                self.avg_p_max += np.amax(self.model.predict(np.float32(history / 255.))[0])

                # if the ball is fall, then the agent is dead --> episode is not over
                if start_life > info['ale.lives']:
                    dead = True
                    start_life = info['ale.lives']

                score += reward
                reward = np.clip(reward, -1., 1.)

                # save the sample <s, a, r, s'> to the replay memory
                self.memory(history, action, reward)

                # if agent is dead, then reset the history
                if dead:
                    history = np.stack((next_state, next_state, next_state, next_state), axis=2)
                    history = np.reshape([history], (1, 84, 84, 4))
                else:
                    history = next_history

                if self.t >= self.t_max or done:
                    self.train_t(done)
                    self.update_local()
                    self.t = 0

                # if done, plot the score over episodes
                if done:
                    episode += 1
                    print("episode:", episode, "  score:", score, "  step:", step)

                    stats = [score, self.avg_p_max / float(step),
                             step]
                    for i in range(len(stats)):
                        self.sess.run(self.update_ops[i], feed_dict={
                            self.summary_placeholders[i]: float(stats[i])
                        })
                    summary_str = self.sess.run(self.summary_op)
                    self.summary_writer.add_summary(summary_str, episode + 1)
                    self.avg_p_max = 0
                    self.avg_loss = 0
                    step = 0

    # In Policy Gradient, Q function is not available.
    # Instead agent uses sample returns for evaluating policy
    def discount_rewards(self, rewards, done):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        if not done:
            running_add = self.local_model.predict(np.float32(self.states[-1] / 255.))[1][0]
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards


    # update policy network and value network every episode
    def train_t(self, done):
        discounted_rewards = self.discount_rewards(self.rewards, done)

        states = np.zeros((len(self.states), 84, 84, 4))
        for i in range(len(self.states)):
            states[i] = self.states[i]

        states = np.float32(states / 255.)

        values = self.local_model.predict(states)[1]
        values = np.reshape(values, len(values))

        advantages = discounted_rewards - values

        self.sess.run(self.update_model, feed_dict={self.feed_acts:
                                                        self.actions,
                                                    self.feed_rwds:
                                                        discounted_rewards,
                                                    self.feed_advs: advantages,
                                                    self.local_model.input:
                                                        states})

        self.states, self.actions, self.rewards = [], [], []

    def update_local(self):
        self.local_model.set_weights(self.model.get_weights())

    def build_local(self):
        input = Input(shape=self.state_size)
        conv = Conv2D(16, (8, 8), strides=(4, 4), activation='relu')(input)
        conv = Conv2D(32, (4, 4), strides=(2, 2), activation='relu')(conv)
        conv = Flatten()(conv)
        fc = Dense(256, activation='relu')(conv)
        policy = Dense(self.action_size, activation='softmax')(fc)
        value = Dense(1, activation='linear')(fc)

        model = Model(inputs=input, outputs=[policy, value])

        model._make_predict_function()
        model.set_weights(self.model.get_weights())

        model.summary()

        return model

    def get_action(self, history):
        history = np.float32(history / 255.)
        policy = self.local_model.predict(history)[0][0]

        policy = policy - np.finfo(np.float32).epsneg

        histogram = np.random.multinomial(1, policy)
        action_index = int(np.nonzero(histogram)[0])
        return action_index, policy

    # make loss function for Policy Gradient
    # [log(action probability) * advantages] will be input for the back prop
    # we add entropy of action probability to loss
    def make_grads_func(self):
        action = K.placeholder(shape=[None, self.action_size])
        discounted_reward = K.placeholder(shape=(None,))
        advantages = K.placeholder(shape=(None, ))

        policy, value = self.local_model.output

        good_prob = K.sum(action * policy, axis=1)
        cross_entropy = K.log(good_prob + 1e-10) * advantages
        cross_entropy = -K.sum(cross_entropy)

        entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)
        entropy = K.sum(entropy)

        policy_loss = cross_entropy + 0.01*entropy

        value_loss = K.mean(K.square(discounted_reward - value))

        loss = policy_loss + 0.5*value_loss
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss,
                                        self.local_model.trainable_weights),
                                       1e+2)
        train = self.optimizer.apply_gradients(zip(grads,
                                                   self.model.trainable_weights))

        return action, discounted_reward, advantages, train

    # save <s, a ,r> of each step
    # this is used for calculating discounted rewards
    def memory(self, history, action, reward):
        self.states.append(history)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)
        self.rewards.append(reward)


# 210*160*3(color) --> 84*84(mono)
# float --> integer (to reduce the size of replay memory)
def pre_processing(next_observe, observe):
    processed_observe = np.maximum(next_observe, observe)
    processed_observe = np.uint8(resize(rgb2gray(processed_observe), (84, 84), mode='constant') * 255)
    return processed_observe


if __name__ == "__main__":
    global_agent = A3CAgent(action_size=3)
    global_agent.train()
