import tensorflow as tf
import numpy as np
from environment import EnvModel
from a2c_agent import A2C
import game_utils

# a standard nn of conv layers and fully connected one
class I2A():
    def __init__(self, in_shape, num_actions, num_rewards, hidden_size, imagination, full_rollout=True):

        self.state = tf.placeholder(tf.float32)
        self.fc_in = tf.placeholder(tf.float32)
        self.x = tf.placeholder(tf.float32)

        self.in_shape = in_shape
        self.num_actions = num_actions
        self.num_rewards = num_rewards
        self.full_rollout = full_rollout

        self.features = self.set_features()
        self.critic = tf.layers.dense(self.x, 1)
        self.actor = tf.layers.dense(self.x, num_actions)

        self.imagination = imagination
        self.encoder = RolloutEncoder(in_shape, num_rewards, hidden_size)
        self.fc = self.set_fc()

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    # fully connected layer
    def set_fc(self):
        output_size = 256
        if self.full_rollout:
            return tf.layers.dense(self.fc_in, output_size, activation=tf.nn.relu)
        else:
            return tf.layers.dense(self.fc_in, output_size, activation=tf.nn.relu)

    def set_features(self):
        h0 = tf.layers.conv2d(self.state, filters=16, kernel_size=3, strides=1, activation=tf.nn.relu)
        out = tf.layers.conv2d(h0, filters=16, kernel_size=3, strides=2, activation=tf.nn.relu)
        return out

    def forward(self):
        batch_size = int(self.state.shape[0])
        imagined_state, imagined_reward = self.imagination(self.state)
        feed_dict = {self.encoder.state: imagined_state, self.encoder.reward: imagined_reward}
        hidden = self.session.run(self.encoder.encode(), feed_dict=feed_dict)

        state = self.features()

        # combine state
        x = tf.concat([state, hidden], 1)
        x = self.session.run(self.fc, feed_dict={self.fc_in:x})

        logit = self.session.run(self.actor, feed_dict={self.x:x})
        value = self.session.run(self.critic, feed_dict={self.x:x})

        return logit, value

# rollout encoder composed from GRU
class RolloutEncoder():
    def __init__(self, in_shape, num_rewards, hidden_size):
        # state shape: [num_steps * batch_size, width, height, depth]
        self.state = tf.placeholder(tf.float32)
        self.reward = tf.placeholder(tf.float32)

        self.hidden_size = hidden_size
        self.in_shape = in_shape
        self.num_rewards = num_rewards

        self.features = self.set_features()
        self.hidden_state = self.encode()

    def set_features(self):
        h0 = tf.layers.conv2d(self.state, filters=16, kernel_size=3, strides=1, activation=tf.nn.relu)
        out = tf.layers.conv2d(h0, filters=16, kernel_size=3, strides=2, activation=tf.nn.relu)
        return out

    def encode(self):
        rnn = tf.contrib.rnn.GRUCell(self.hidden_size)
        # check features shape
        rnn_input = tf.concat([self.features, self.reward], 2)
        _, next_state = tf.nn.dynamic_rnn(rnn, rnn_input)
        return next_state

# imagination core (IC) predicts the next time step conditioned on an action sampled from the rollout policy π
class ImaginationCore():
    def __init__(self, num_rollouts, num_states, num_actions, num_rewards, env_model, a2c, full_rollout=True):
        self.num_rollouts = num_rollouts
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_rewards = num_rewards
        self.full_rollout = full_rollout

        self.env_model = env_model
        self.a2c = a2c


    def imagine(self, state):
        batch_size = state.shape[0]

        rollout_states = []
        rollout_rewards = []

        if self.full_rollout:
            action = np.array([[[i] for i in range(self.num_actions)] for j in range(batch_size)])
            action = action.reshape((-1,))
            rollout_batch_size = batch_size * self.num_actions
        else:
            action = self.a2c.next_action(state)
            rollout_batch_size = batch_size

        print(action.shape)
        for step in range(self.num_rollouts):
            imagined_state, imagined_reward = self.env_model.forward(state, action)

            # convertion

            onehot_reward = tf.zeros(rollout_batch_size, self.num_rewards)
            onehot_reward[range(rollout_batch_size), imagined_reward] = 1

            rollout_states.append(imagined_state)
            rollout_rewards.append(imagined_state)

            state = imagined_state
            action = self.a2c.next_action(state)

        return rollout_states, rollout_rewards


if __name__ == '__main__':
    game = game_utils.pong
    a2c = A2C(game.name)
    env_model = EnvModel(len(game.pixels))
    env_model.load_last_checkpoint()

    # session = tf.Session()
    # session.run(tf.global_variables_initializer())
    # core = ImaginationCore(1, 1, 1, 1, env_model, a2c, False)
    # session.run(core.imagine(a2c.game.reset()))
