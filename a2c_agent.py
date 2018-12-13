import tensorflow as tf
import numpy as np
import gym
from game_utils import normalize_states

## STABLE LOG
def log(x):
    return tf.log(tf.maximum(x, 1e-5))

## HYPERPARAMETERS
GAMMA = 0.99
LEARNING_RATE = 0.00005
ENTROPY_FACTOR = 0.02
EPOCHS = 1000
TIMESTEPS = 2500
DECAY = 0.99
EPSILON = 1e-5
REWARD_FACTOR = 1.0
SAVE_EVERY = 100
VALUE_FACTOR = 1.0
RENDER = False
MINIBATCH_SIZE = 8
PER_TIMESTEP_REWARD = 0
STACK_SIZE = 2

class A2C:
    def __init__(self, game, sess=None):
        self.game = game
        self.num_actions = 3
        self.state_size = [2, 186, 160, 3]
        with tf.variable_scope('a2c'):
            self.state_input = tf.placeholder(tf.float32, [None] + self.state_size)

            # Define any additional placeholders needed for training your agent here:

            self.rewards = tf.placeholder( shape = [ None ], dtype = tf.float32 )
            self.actions = tf.placeholder( shape = [ None ], dtype = tf.int32 )

            self.common = self.common()
            self.state_value = self.critic()
            self.actor_probs = self.actor()
            self.loss_val = self.loss()
            self.train_op = self.optimizer()

            if sess:
                self.session = sess

            else:
                self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())


        trainable_vars = tf.trainable_variables()

        # For saving/loading models
        self.saver = tf.train.Saver(trainable_vars)

    # Load the last saved checkpoint during training or used by test
    def load_last_checkpoint(self):
        self.saver.restore(self.session, tf.train.latest_checkpoint('models/a2c/'))

    def save_checkpoint(self):
        self.saver.save(self.session, 'models/a2c/a2c_saved_model')

    def play(self):
        self.load_last_checkpoint()
        while True:
            st = self.game.reset()
            sss = [st] * STACK_SIZE

            for t in range(TIMESTEPS):
                self.game.render()
                actDist = self.session.run( self.actor_probs, feed_dict={ self.state_input: np.array( normalize_states([ sss[-STACK_SIZE:] ]) ) } )
                action_idx= np.random.choice( self.num_actions, 1, p=actDist[0] )[0]
                if action_idx == 0:
                    action = 0
                elif action_idx == 1:
                    action = 2
                elif action_idx == 2:
                    action = 5

                st1, _, done, _ = self.game.step(action)
                sss.append(st1)


    def next_action(self, state):
        # self.load_last_checkpoint()
        actDist = self.session.run( self.actor_probs, feed_dict={ self.state_input: np.array( [ state ] ) } )
        action_idx= np.random.choice( self.num_actions, 1, p=actDist[0] )[0]
        if action_idx == 0:
            action = 0
        if action_idx == 1:
            action = 2
        elif action_idx == 2:
            action = 5
        return action

    def optimizer(self):
        """
        :return: Optimizer for your loss function
        """
        return tf.train.AdamOptimizer(LEARNING_RATE).minimize( self.loss_val )

    def common(self):
        states = tf.reshape(tf.image.resize_images(tf.reshape(self.state_input, [-1,186,160,3]), [93,80]), [-1,2,93,80,3])/255.0

        h0 = tf.nn.elu(tf.layers.batch_normalization(tf.layers.conv3d(states, 8, [1,4,4], (1,2,2), padding='valid', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())))
        h1 = tf.nn.elu(tf.layers.batch_normalization(tf.layers.conv3d(h0, 16, [1,4,4], (1,2,2), padding='valid', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())))
        h2 = tf.nn.elu(tf.layers.batch_normalization(tf.layers.conv3d(h1, 16, [1,4,4], (1,2,2), padding='valid', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())))
        #import pdb; pdb.set_trace()
        h3 = tf.nn.elu(tf.layers.batch_normalization(tf.layers.dense(tf.layers.flatten(h2), 512, kernel_initializer=tf.contrib.layers.xavier_initializer())))
        return h3

    def critic(self):
        """
        Calculates the estimated value for every state in self.state_input. The critic should not depend on
        any other tensors besides self.state_input.
        :return: A tensor of shape [num_states] representing the estimated value of each state in the trajectory.
        """
        #import pdb; pdb.set_trace()
        output = tf.nn.elu(tf.layers.batch_normalization(tf.layers.dense(self.common, 128, kernel_initializer=tf.contrib.layers.xavier_initializer())))
        output = tf.nn.elu(tf.layers.batch_normalization(tf.layers.dense(output, 64, kernel_initializer=tf.contrib.layers.xavier_initializer())))
        output = tf.layers.dense(output, 1, kernel_initializer=tf.contrib.layers.xavier_initializer())
        return output

    def actor(self):
        """
        Calculates the action probabilities for every state in self.state_input. The actor should not depend on
        any other tensors besides self.state_input.
        :return: A tensor of shape [num_states, num_actions] representing the probability distribution
            over actions that is generated by your actor.
        """
        output = tf.nn.elu(tf.layers.batch_normalization(tf.layers.dense(self.common, 128, kernel_initializer=tf.contrib.layers.xavier_initializer())))
        output = tf.nn.elu(tf.layers.batch_normalization(tf.layers.dense(output, 64, kernel_initializer=tf.contrib.layers.xavier_initializer())))
        output = tf.layers.dense(output, self.num_actions, kernel_initializer=tf.contrib.layers.xavier_initializer())
        return tf.nn.softmax(output)


    def loss(self):
        """
        :return: A scalar tensor representing the combined actor and critic loss.
        """
        #import pdb; pdb.set_trace()
        indicies = tf.range(0, tf.shape(self.actor_probs)[0]) * self.num_actions + self.actions
        actProbs = tf.gather(tf.reshape(self.actor_probs, [-1]), indicies)
        # A(s, a) = Q(s, a) - V(s)
        advantage = self.rewards - self.state_value
        # loss for critic
        cLoss = tf.reduce_mean(tf.square(advantage))
        # loss for actor
        aLoss = -tf.reduce_mean(log(actProbs) * advantage)
        entropy = -tf.reduce_mean(actProbs * log(actProbs))
        return VALUE_FACTOR*cLoss + aLoss + ENTROPY_FACTOR*entropy

    def train_episode(self):
        """
        train_episode will be called 1000 times by the autograder to train your agent. In this method,
        run your agent for a single episode, then use that data to train your agent. Feel free to
        add any return values to this method.
        """

        c_actions = []
        c_rewards = []
        c_states = []


        for m in range(MINIBATCH_SIZE):
            rrr = []
            st = self.game.reset()
            sss = [st] * STACK_SIZE

            for t in range(TIMESTEPS):
                if RENDER:
                    self.game.render()
                actDist = self.session.run( self.actor_probs, feed_dict={ self.state_input: np.array( normalize_states([ sss[-STACK_SIZE:] ]) ) } )
                action_idx= np.random.choice( self.num_actions, 1, p=actDist[0] )[0]
                if action_idx == 0:
                    action = 0
                elif action_idx == 1:
                    action = 2
                elif action_idx == 2:
                    action = 5

                st1, reward, done, _ = self.game.step(action)

                c_states.append(sss[-STACK_SIZE:])
                rrr.append(reward * REWARD_FACTOR + PER_TIMESTEP_REWARD)
                c_actions.append(action_idx)

                sss.append(st1)

                if done or t == TIMESTEPS - 1:
                    print(np.sum(rrr))
                    d_r = [0]
                    #import pdb; pdb.set_trace()
                    reversed_rewards = rrr[:]
                    reversed_rewards.reverse()
                    for r in reversed_rewards:
                       d_r.append( d_r[ -1 ] * GAMMA + r )
                    disRs = d_r[1:]
                    disRs.reverse()

                    c_rewards = c_rewards + disRs

                    break

        _ , loss = self.session.run( [self.train_op, self.loss_val], feed_dict= { self.state_input: np.array( normalize_states(c_states)), self.rewards: np.array( c_rewards ), self.actions: np.array( c_actions ) } )
        print( "rewards: ", np.mean(c_rewards) )
        print( "loss: ", loss )
        return


if __name__ == '__main__':
    model = A2C(gym.make('Pong-v0'))
    model.save_checkpoint()
    # for i in range(EPOCHS):
    #     model.train_episode()
    #
    #     if i%SAVE_EVERY == 0:
    #         print("MODEL saved at iteration: ", i)
    #         model.save_checkpoint()
