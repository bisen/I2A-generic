import tensorflow as tf
import numpy as np
import game_utils
from environment import EnvModel
from a2c_agent import A2C
from game_utils import normalize_states
import a2c_agent

MODEL_PATH_A2C = 'models/a2c/a2c_saved_model'
MODEL_PATH_ENV = 'models/env/env_saved_model'

BATCH_SIZE = 3

# a standard nn of conv layers and fully connected one
class I2A():
    def __init__(self, in_shape, num_actions, num_rewards, num_rollouts, hidden_size, encoder, imagination, full_rollout=True):
        self.fc_in = tf.placeholder(tf.float32, shape=[num_rollouts, BATCH_SIZE*119040+1])
        self.x = tf.placeholder(tf.float32, shape=[num_rollouts, 256])
        self.reward = tf.placeholder(tf.float32, shape=[None, None, num_rewards])
        self.actions = tf.placeholder(tf.int32, shape=[num_rollouts])

        self.in_shape = in_shape
        self.num_actions = num_actions
        self.num_rewards = num_rewards
        self.full_rollout = full_rollout
        self.num_rollouts = num_rollouts
        self.hidden_size = hidden_size

        self.imagination = imagination
        self.encoder = encoder

        # fully connected layer
        self.fc = self.set_fc()
        self.critic = tf.layers.dense(self.fc, 1)
        self.actor = tf.nn.softmax(tf.layers.dense(self.fc, num_actions))

        self.loss_val = self.loss()

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        # For saving/loading models
        self.saver = tf.train.Saver(tf.global_variables())

    # Load the last saved checkpoint during training or used by test
    def load_last_checkpoint(self):
        self.saver.restore(self.session, tf.train.latest_checkpoint('models/i2a/'))

    def save_checkpoint(self):
        self.saver.save(self.session, 'models/i2a/i2a_saved_model')

    # fully connected layer
    def set_fc(self):
        output_size = 256
        if self.full_rollout:
            return tf.layers.dense(self.fc_in, output_size, activation=tf.nn.relu)
        else:
            return tf.layers.dense(self.fc_in, output_size, activation=tf.nn.relu)


    # output the suggested action and value in a batch
    def forward(self, states):
        batch_size = len(states)
        imagined_states, imagined_rewards, imagined_actions = self.imagination.imagine(states)
        hidden = self.encoder.get_hidden(imagined_states, imagined_rewards)
        f = self.encoder.get_f(imagined_states, imagined_rewards)
        f = np.reshape(f, [hidden.shape[0], -1])
        # combine state
        x = np.concatenate([f, hidden], 1)

        logit = self.session.run(self.actor, feed_dict={self.fc_in:x})
        value = self.session.run(self.critic, feed_dict={self.fc_in:x})

        return logit, value

    def loss(self):
        indicies = tf.range(0, tf.shape(self.actor)[0]) * 2 + self.actions
        actProbs = tf.gather(tf.reshape(self.actor, [-1]), indicies)
        advantage = self.reward[:,:1,1] - self.critic
        cLoss = tf.reduce_mean(tf.square(advantage))
        return cLoss

    def optimize(self, dict):
        self.session.run(tf.train.AdamOptimizer(.001).minimize(self.loss_val), feed_dict=dict)

# rollout encoder composed from GRU
class RolloutEncoder():
    def __init__(self, roll_steps, in_shape, num_rewards, hidden_size):
        # state shape: [batch_size, num_steps, width, height, depth]
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            self.state = tf.placeholder(tf.float32, shape=[None, None, in_shape[0], in_shape[1], in_shape[2]])
            self.reward = tf.placeholder(tf.float32, shape=[None, None, num_rewards])

            self.roll_steps = roll_steps
            self.hidden_size = hidden_size
            self.in_shape = in_shape
            self.num_rewards = num_rewards

            self.features = self.set_features()
            self.hidden_state = self.encode()

            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())

    # train states
    def set_features(self):
        state = tf.reshape(self.state, [-1, self.in_shape[0], self.in_shape[1], self.in_shape[2]])
        h0 = tf.layers.conv2d(state, filters=16, kernel_size=3, strides=1, padding='SAME', activation=tf.nn.relu)
        out = tf.layers.conv2d(h0, filters=16, kernel_size=3, strides=2, padding='SAME', activation=tf.nn.relu)
        return out

    # encode the rollout imagined states and actions
    def encode(self):
        rnn = tf.contrib.rnn.GRUCell(self.hidden_size)
        features = tf.reshape(self.features, [self.roll_steps, -1, int(self.in_shape[0]*self.in_shape[1]/4)*16])
        rnn_input = tf.concat([features, self.reward], 2)
        _, next_state = tf.nn.dynamic_rnn(rnn, rnn_input, dtype=tf.float32)
        return next_state

    def get_hidden(self, state, reward):
        return self.session.run(self.hidden_state, feed_dict={self.state: state, self.reward: reward})

    def get_f(self, state, reward):
        return self.session.run(self.features, feed_dict={self.state: state, self.reward: reward})

#imagination core (IC) predicts the next time step conditioned on an action sampled from the rollout policy π
class ImaginationCore():
    def __init__(self, num_rollouts, num_actions, num_rewards, env, a2c, full_rollout=False):
        self.num_rollouts = num_rollouts
        self.num_actions = num_actions
        self.num_rewards = num_rewards
        self.full_rollout = full_rollout
        self.env = env
        self.a2c = a2c

    # returns imagined actions, states, and rewards based on the number of rollouts
    def imagine(self, states):
        batch_size = states.shape[0]
        # fix state normalization
        states = np.array(normalize_states([[s, s] for s in states]))
        rollout_states = []
        rollout_rewards = []
        rollout_actions = []

        # starts with given states and roll out based on all possible actions
        if self.full_rollout:
            actions = np.array([[[i] for i in range(self.num_actions)] for j in
                                range(batch_size)])

            actions = actions.reshape([-1,])

            rollout_batch_size = batch_size * self.num_actions

        # roll out with only the action suggested by the a2c agent
        else:
            with a2c.graph.as_default():
                # pass in current frame and next frame
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
            rollout_actions.append(actions)

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

        return np.array(rollout_states), np.array(rollout_rewards), np.array(rollout_actions)

    # convert the shape of the states and rewards
    def convert_states_rewards(self, imagined_states, imagined_rewards, rollout_batch_size):
        imagined_states = np.reshape(imagined_states, (rollout_batch_size, 186, 160, 5))
        imagined_states = game_utils.logits_to_pixels(imagined_states)
        # make rewards one hot
        if isinstance(imagined_rewards, int):
            imagined_rewards = np.array([imagined_rewards])
        onehot_rewards = np.zeros((rollout_batch_size, self.num_rewards))
        onehot_rewards[range(rollout_batch_size), imagined_rewards.astype(int)] = 1

        return imagined_states, onehot_rewards

# import models with proper graph
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

    num_rollout = 3
    num_rewards = 6
    image_dim = [186, 160, 3]
    state = []
    for i in range(BATCH_SIZE):
        state.append(a2c.model.game.reset())
    state = np.array(state)

    # initialize core
    core = ImaginationCore(num_rollout, pong.num_actions, num_rewards, env, a2c, False)
    encoder = RolloutEncoder(num_rollout, image_dim, num_rewards, 1)

    # load and run I2A
    i2a = I2A(in_shape=image_dim, num_actions=pong.num_actions, num_rewards=num_rewards, num_rollouts=num_rollout, hidden_size=1,
              encoder=encoder, imagination=core)
    i2a.load_last_checkpoint()
    logit, value = i2a.forward(state)
    print('logit', logit)
    print('value', value)
