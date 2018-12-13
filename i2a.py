import tensorflow as tf
import numpy as np
import game_utils
from environment import EnvModel
from a2c_agent import A2C
from game_utils import normalize_states
import a2c_agent

MODEL_PATH_A2C = 'models/a2c/a2c_saved_model'
MODEL_PATH_ENV = 'models/env/env_saved_model'


# a standard nn of conv layers and fully connected one
class I2A():
    def __init__(self, game, in_shape, num_actions, num_rewards, num_rollouts, hidden_size, encoder, imagination, full_rollout=True):
        self.state = tf.placeholder(tf.float32, shape=[None, None, in_shape[0], in_shape[1], in_shape[2]])
        self.fc_in = tf.placeholder(tf.float32, shape=[num_rollouts, 238081])
        self.x = tf.placeholder(tf.float32, shape=[None])

        self.in_shape = in_shape
        self.num_actions = num_actions
        self.num_rewards = num_rewards
        self.full_rollout = full_rollout
        self.num_rollouts = num_rollouts

        self.features = self.set_features()
        # self.critic = tf.layers.dense(self.x, 1)
        # self.actor = tf.layers.dense(self.x, num_actions)

        # self.imagination = ImaginationCore(num_rollouts, game.num_actions, 6, env, a2c)
        # self.encoder = RolloutEncoder(num_rollouts, in_shape, num_rewards, hidden_size)
        self.imagination = imagination
        self.encoder = encoder
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
        state = tf.reshape(self.state, [-1, self.in_shape[0], self.in_shape[1], self.in_shape[2]])
        h0 = tf.layers.conv2d(state, filters=16, kernel_size=3, strides=1, activation=tf.nn.relu)
        out = tf.layers.conv2d(h0, filters=16, kernel_size=3, strides=2, activation=tf.nn.relu)
        return out

    def forward(self, states):
        batch_size = len(states)
        imagined_state, imagined_reward = self.imagination.imagine(states)
        feed_dict = {self.encoder.state: imagined_state, self.encoder.reward: imagined_reward}
        hidden = self.session.run(self.encoder.encode(), feed_dict=feed_dict)

        f = self.features()
        f = tf.reshape(f, [self.num_rollouts, -1])
        # combine state
        x = tf.concat([f, hidden], 1)
        x = self.session.run(self.fc, feed_dict={self.fc_in:x})

        print(x.shape)
        # logit = self.session.run(self.actor, feed_dict={self.x:x})
        # value = self.session.run(self.critic, feed_dict={self.x:x})

        # return logit, value

# rollout encoder composed from GRU
class RolloutEncoder():
    def __init__(self, roll_steps, in_shape, num_rewards, hidden_size):
        # state shape: [batch_size, num_steps, width, height, depth]
        self.state = tf.placeholder(tf.float32, shape=[None, None, in_shape[0], in_shape[1], in_shape[2]])
        self.reward = tf.placeholder(tf.float32, shape=[None, None, num_rewards])

        self.roll_steps = roll_steps
        self.hidden_size = hidden_size
        self.in_shape = in_shape
        self.num_rewards = num_rewards

        self.features = self.set_features()
        self.hidden_state = self.encode()

    def set_features(self):
        state = tf.reshape(self.state, [-1, self.in_shape[0], self.in_shape[1], self.in_shape[2]])
        h0 = tf.layers.conv2d(state, filters=16, kernel_size=3, strides=1, padding='SAME', activation=tf.nn.relu)
        out = tf.layers.conv2d(h0, filters=16, kernel_size=3, strides=2, padding='SAME', activation=tf.nn.relu)
        return out

    def encode(self):
        rnn = tf.contrib.rnn.GRUCell(self.hidden_size)
        # check features shape
        features = tf.reshape(self.features, [self.roll_steps, -1, int(self.in_shape[0]*self.in_shape[1]/4)*16])
        rnn_input = tf.concat([features, self.reward], 2)
        _, next_state = tf.nn.dynamic_rnn(rnn, rnn_input, dtype=tf.float32)
        return next_state

# imagination core (IC) predicts the next time step conditioned on an action sampled from the rollout policy π
class ImaginationCore():
    def __init__(self, num_rollouts, num_actions, num_rewards, env, a2c, full_rollout=False):
        self.num_rollouts = num_rollouts
        # self.num_states = num_states
        self.num_actions = num_actions
        self.num_rewards = num_rewards
        self.full_rollout = full_rollout
        self.env = env
        self.a2c = a2c


    def imagine(self, states):
        batch_size = states.shape[0]
        # fix state normalization
        states = np.array(normalize_states([[s, s] for s in states]))
        rollout_states = []
        rollout_rewards = []


        if self.full_rollout:
            actions = np.array([[[i] for i in range(self.num_actions)] for j in
                                range(batch_size)])

            actions = actions.reshape([-1,])

            rollout_batch_size = batch_size * self.num_actions
        else:
            with a2c.graph.as_default():
                # pass in current frame and next frame
                #actions = [self.env.model.next_actions(states)]
                actions = []
            for s in states:
                actions.append(self.a2c.model.next_action(s))
            rollout_batch_size = batch_size

        for step in range(self.num_rollouts):
            with env.graph.as_default():
                imagined_states, imagined_rewards = env.model.forward(states, actions)

            s, r = self.convert_states_rewards(imagined_states, imagined_rewards, rollout_batch_size)
            rollout_states.append(s)
            rollout_rewards.append(r)

            # reshape imagined state to a proper shape that can be passed into the environment
            imagined_states = np.reshape(imagined_states, (rollout_batch_size, 186, 160, 5))
            imagined_states = game_utils.onehot_to_pixels(imagined_states)

            # concatenate the current state and next state for next action
            next_states = [list(s) for s in zip([s[1] for s in states], imagined_states)]
            states = next_states
            with a2c.graph.as_default():
                actions = []
                for s in states:
                    actions.append(self.a2c.model.next_action(s))

        return np.array(rollout_states), np.array(rollout_rewards)

    def convert_states_rewards(self, imagined_states, imagined_rewards, rollout_batch_size):
        imagined_states = np.reshape(imagined_states, (rollout_batch_size, 186, 160, 5))
        imagined_states = game_utils.logits_to_pixels(imagined_states)
        # make rewards one hot
        if isinstance(imagined_rewards, int):
            imagined_rewards = np.array([imagined_rewards])
        onehot_rewards = np.zeros((rollout_batch_size, self.num_rewards))
        onehot_rewards[range(rollout_batch_size), imagined_rewards.astype(int)] = 1

        return imagined_states, onehot_rewards

class ImportA2CGraph():
    """  Importing and running isolated TF graph """
    def __init__(self, game):
        # Create local graph and use it in the session
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            # import saved model
            self.model = A2C(game, self.sess)
            self.model.load_last_checkpoint()


class ImportEnvGraph():
    """  Importing and running isolated TF graph """
    def __init__(self, game, loc):
        # Create local graph and use it in the session
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            # import saved model
            self.model = EnvModel(game, self.sess)
            self.model.load_last_checkpoint()

if __name__ == '__main__':
    pong = game_utils.pong

    a2c = ImportA2CGraph(pong.game)
    env = ImportEnvGraph(pong, MODEL_PATH_ENV)

    session = tf.Session()

    core = ImaginationCore(3, pong.num_actions, 6, env, a2c, False)
    # imagined_states, imagined_rewards = core.imagine(np.array([a2c.model.game.reset(), a2c.model.game.reset()]))

    encoder = RolloutEncoder(3, [186, 160, 3], 6, 1)
    i2a = I2A(pong, in_shape=[186, 160, 3], num_actions=pong.num_actions, num_rewards=6, num_rollouts=2, hidden_size=1,
              encoder=encoder, imagination=core)

    i2a.forward(np.array([a2c.model.game.reset(), a2c.model.game.reset()]))
    # feed_dict = {encoder.state: imagined_states,
    #              encoder.reward: imagined_rewards}
    # session.run(tf.global_variables_initializer())
    # h, f = session.run([encoder.hidden_state, encoder.features], feed_dict=feed_dict)
    # print(h.shape, f.shape)
    # x = np.concatenate((h, np.reshape(f, [3, -1])), axis=1)
    # print(x.shape)