import tensorflow as tf
import gym
import random
import numpy as np
from a2c_agent import A2C
import game_utils
from game_utils import normalize_states, map_pixels

BATCH_SIZE = 16

pong = game_utils.pong

agent = A2C(pong.game)

class EnvModel:
    def __init__(self, game, sess=None):
        self.n_pixels = len(game.pixels)
        self.game = game
        # self.n_rewards = n_rewards
        with tf.variable_scope('env'):
            # [batchsize, 186, 160, len(rgb) + num_actions]
            self.inputs = tf.placeholder(tf.float32, [None, 186, 160, 3 + game.num_actions])
            self.targets = tf.placeholder(tf.int32, [None])
            self.target_rewards = tf.placeholder(tf.float32, [None])

            self.image, self.reward = self.predict()
            self.image_loss = self.image_loss_function()
            self.reward_loss = self.reward_loss_function()
            self.loss = self.loss_function()
            self.optimizer = self.optimize()

            if sess:
                self.session = sess
            else:
                self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())

        trainable_vars = tf.trainable_variables()

        # For saving/loading models
        self.saver = tf.train.Saver(trainable_vars)

    def basic_block(self, inputs, n1, n2, n3):
        # pool-and-inject
        inputs2 = tf.layers.max_pooling2d(inputs, inputs.shape[1:3], 1)
        inputs2 = tf.tile(inputs2, tf.constant([1, int(inputs.shape[1]), int(inputs.shape[2]), 1]))
        inputs2 = tf.concat([inputs, inputs2], 3)

        # conv1
        padded = tf.pad(inputs2, [[0, 0], [54, 54], [41, 41], [0, 0]])
        conv1 = tf.layers.conv2d(padded, n1, 1, 2, "valid", activation=tf.nn.relu)
        padded = tf.pad(conv1, [[0, 0], [24, 24], [24, 24], [0, 0]])
        conv1 = tf.layers.conv2d(padded, n1, 10, 1, "valid", activation=tf.nn.relu)

        # conv2
        conv2 = tf.layers.conv2d(inputs2, n2, 1, 1, activation=tf.nn.relu)
        padded = tf.pad(conv2, [[0, 0], [1, 1], [1, 1], [0, 0]])
        conv2 = tf.layers.conv2d(padded, n2, 3, 1, "valid", activation=tf.nn.relu)

        # conv3
        catted = tf.concat([conv1, conv2], 3)
        conv3 = tf.layers.conv2d(catted, n3, 1, 1, "valid", activation=tf.nn.relu)

        # concat with input
        out = tf.concat([conv3, inputs], 3)

        return out

    # Load the last saved checkpoint during training or used by test
    def load_last_checkpoint(self):
        self.saver.restore(self.session, tf.train.latest_checkpoint('models/env/'))

    def save_checkpoint(self):
        self.saver.save(self.session, 'models/env/env_saved_model')

    def predict(self):
        # initial convolutions
        conv = tf.layers.conv2d(self.inputs, 64, 1, 1, activation=tf.nn.relu)

        # invocations of basic blocks
        basic_block1 = self.basic_block(conv, 16, 32, 64)
        basic_block2 = self.basic_block(basic_block1, 16, 32, 64)

        # last conv
        image_conv = tf.layers.conv2d(basic_block2, 256, 1, 1)
        image = tf.reshape(image_conv, [-1, 256])

        # fully connected
        image = tf.layers.dense(image, self.n_pixels)
        # print(self.image.shape)
        # reward
        reward = tf.layers.conv2d(basic_block2, 192, 1, 1)
        reward = tf.layers.conv2d(reward, 64, 1, 1)
        reward = tf.reshape(reward, [-1, 186 * 160 * 64])
        reward = tf.layers.dense(reward, 1, activation=tf.nn.softmax)
        reward = tf.squeeze(reward)

        return image, reward

    def forward(self, states, actions):
        # self.load_last_checkpoint()
        return self.session.run([self.image, self.reward], feed_dict={self.inputs: self.convert_input(states, actions)})

    def image_loss_function(self):
        onehot = tf.one_hot(self.targets, self.n_pixels)
        losses = tf.nn.softmax_cross_entropy_with_logits(labels = onehot, logits = self.image)
        return tf.reduce_sum(losses)

    def reward_loss_function(self):
        losses = tf.losses.mean_squared_error(self.reward, self.target_rewards)
        return tf.reduce_sum(losses)

    def loss_function(self):
        reward_coef = 0.1
        return self.image_loss + reward_coef * self.reward_loss

    def optimize(self):
        train = tf.train.AdamOptimizer(0.001).minimize(self.loss)
        return train

    # train for 1 epoch
    def train_episode(self, feed_dict):
        return self.session.run([self.loss, self.optimizer], feed_dict=feed_dict)

    def convert_input(self, states, actions):
        batch_size = len(states)
        # delete top rows, make top/bottom borders black

        # convert actions to onehot representation
        onehot_actions = np.zeros((batch_size, 186, 160, self.game.num_actions))
        onehot_actions[range(batch_size), actions] = 1

        # concatenate states and actions to feed to optimizer
        inputs = np.concatenate([np.array(states)[:,1], onehot_actions], 3)
        return inputs

# return a batch of random actions
# replace this with a real policy
def next_actions(states):
    actions = []
    for s in states:
        actions.append(agent.next_action(s))
    return actions

# play BATCH_SIZE games using the policy given by next_actions
def next_batch(n_updates):
    envs = [gym.make(pong.name) for i in range(BATCH_SIZE)]
    states = normalize_states([env.reset() for env in envs])
    states = [[s,s] for s in states]
    #states, _, _, _ = zip(*[env.step(0) for env in envs])

    for i in range(n_updates):
        actions = next_actions(states)
        results = normalize_states([env.step(actions[i]) for i, env in enumerate(envs)])
        next_states, rewards, is_done, _ = zip(*results)

        next_states = [ list(a) for a in zip([a[1] for a in states], next_states) ]

        yield i, states, actions, rewards, next_states, is_done

        states = next_states



if __name__ == '__main__':
    # training
    # set up placeholders
    env_model = EnvModel(pong)
    env_model.load_last_checkpoint()

    for it, states, actions, rewards, next_states, is_done in next_batch(10):
        normalize_states(states)
        inputs = env_model.convert_input(states, actions)
        next_states = normalize_states(next_states)
        # convert target states to indexes, using map_pixels function
        # there are only 5 different kinds of pixels
        targets = map_pixels(next_states)
        env_model.forward(states, actions)
        loss, _ = env_model.train_episode(feed_dict={
            env_model.inputs: inputs,
            env_model.targets: targets,
            env_model.target_rewards: rewards
        })
        print(it, loss)
        env_model.save_checkpoint()
