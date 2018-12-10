import tensorflow as tf
import gym
import random
import numpy as np

BATCH_SIZE = 16

pixels = (
    (144, 72, 17),      # background
    (0, 0, 0),          # boundary
    (213, 130, 74),     # opponent
    (236, 236, 236),    # ball
    (92, 186, 92),      # player
)

num_actions = 6

pixel_to_type = {pixel: i for i, pixel in enumerate(pixels)}
def map_pixels(states):
    types = []
    for pixel in np.array(states).reshape(-1, 3):
        types.append(pixel_to_type[tuple(pixel)])
    return types

class BasicBlock:
    def __init__(self, inputs, n1, n2, n3):

        # pool-and-inject
        inputs2 = tf.layers.max_pooling2d(inputs, inputs.shape[1:3], 1)
        inputs2 = tf.tile(inputs2, tf.constant([1, int(inputs.shape[1]), int(inputs.shape[2]), 1]))
        inputs2 = tf.concat([inputs, inputs2], 3)
        
        # conv1
        padded = tf.pad(inputs2, [[0, 0], [54, 54], [41, 41], [0, 0]])
        conv1 = tf.layers.conv2d(padded, n1, 1, 2, "valid", activation = tf.nn.relu)
        padded = tf.pad(conv1, [[0, 0], [24, 24], [24, 24], [0, 0]])
        conv1 = tf.layers.conv2d(padded, n1, 10, 1, "valid", activation = tf.nn.relu)

        # conv2
        conv2 = tf.layers.conv2d(inputs2, n2, 1, 1, activation = tf.nn.relu)
        padded = tf.pad(conv2, [[0, 0], [1, 1], [1, 1], [0, 0]])
        conv2 = tf.layers.conv2d(padded, n2, 3, 1, "valid", activation = tf.nn.relu)

        # conv3
        catted = tf.concat([conv1, conv2], 3)
        conv3 = tf.layers.conv2d(catted, n3, 1, 1, "valid", activation = tf.nn.relu)

        # concat with input
        self.out = tf.concat([conv3, inputs], 3)

    def forward(self):
        return self.out

class EnvModel:
    def __init__(self, inputs, n_pixels, n_rewards):

        # initial convolutions
        conv = tf.layers.conv2d(inputs, 64, 1, 1, activation = tf.nn.relu)

        # invocations of basic blocks
        basic_block1 = BasicBlock(conv, 16, 32, 64).forward()
        basic_block2 = BasicBlock(basic_block1, 16, 32, 64).forward()

        # last conv
        image_conv = tf.layers.conv2d(basic_block2, 256, 1, 1)
        image = tf.reshape(image_conv, [-1, 256])

        # fully connected
        self.image = tf.layers.dense(image, n_pixels)
        print(self.image.shape)

        # reward
        reward = tf.layers.conv2d(basic_block2, 192, 1, 1)
        reward = tf.layers.conv2d(reward, 64, 1, 1)
        reward = tf.reshape(reward, [-1, 186 * 160 * 64])
        self.reward = tf.layers.dense(reward, n_rewards, activation = tf.nn.softmax)

    def forward(self):
        return self.image, self.reward

    def optimize(self, targets):
        onehot = tf.one_hot(targets, len(pixels))
        loss = tf.nn.softmax_cross_entropy_with_logits(labels = onehot, logits = self.image)
        train = tf.train.AdamOptimizer(0.001).minimize(loss)
        return loss, train

# return a batch of random actions
# replace this with a real policy
def next_actions(states):
    actions = []
    for i in range(BATCH_SIZE):
        actions.append(random.randrange(num_actions))
    return actions

# strip first 24 rows, make borders black
def normalize_states(states):
    normalized = []
    for s in states:
        s[24:34] = [0, 0, 0]
        s[-16:] = [0, 0, 0]
        s = s[24:]
        normalized.append(s)
    return normalized


# play BATCH_SIZE games using the policy given by next_actions
def next_batch(n_updates):
    envs = [gym.make("Pong-v0") for i in range(BATCH_SIZE)]
    states = [env.reset() for env in envs]
    states, _, _, _ = zip(*[env.step(0) for env in envs])

    for i in range(n_updates):
        actions = next_actions(states)
        results = [env.step(actions[i]) for i, env in enumerate(envs)]
        next_states, rewards, is_done, _ = zip(*results)

        yield i, states, actions, next_states, is_done

        states = next_states

# training
# set up placeholders
inputs_placeholder = tf.placeholder(tf.float32, [None, 186, 160, 3 + num_actions])
env_model = EnvModel(inputs_placeholder, len(pixels), 3)
targets_placeholder = tf.placeholder(tf.int32, [None])
loss, train = env_model.optimize(targets_placeholder)

# set up session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for it, states, actions, next_states, is_done in next_batch(100):

    # delete top rows, make top/bottom borders black
    states = normalize_states(states)
    next_states = normalize_states(next_states) 

    # convert actions to onehot representation
    onehot_actions = np.zeros((BATCH_SIZE, 186, 160, num_actions))
    onehot_actions[range(BATCH_SIZE), actions] = 1

    # concatenate states and actions to feed to optimizer
    inputs = np.concatenate([np.array(states), onehot_actions], 3)

    # convert target states to indexes, using map_pixels function
    # there are only 5 different kinds of pixels
    targets = map_pixels(next_states)

    sess.run([loss, train], feed_dict={inputs_placeholder: inputs, targets_placeholder: targets})
    print(it)
