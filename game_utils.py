import gym
import numpy as np

# a simple wrapper of gym atari games
class AtariGames():
    def __init__(self, name, pixels, num_action):
        self.game = gym.make(name)

        self.name = name
        self.pixels = pixels
        self.num_actions = num_action

pong_pixels = (
    (144, 72, 17),      # background
    (0, 0, 0),          # boundary
    (213, 130, 74),     # opponent
    (236, 236, 236),    # ball
    (92, 186, 92),      # player
)

pong = AtariGames('Pong-v0', pong_pixels, 3)

pixel_to_type = {pixel: i for i, pixel in enumerate(pong.pixels)}
type_to_pixel = {y:x for x, y in pixel_to_type.items()}

def map_pixels(states):
    types = []
    for pixel in np.array(states)[:,1].reshape(-1, 3):
        types.append(pixel_to_type[tuple(pixel)])
    return types



# strip first 24 rows, make borders black
def normalize_states(states):
    normalized = []
    for ss in states:
        frames = []
        for s in ss:
            s[24:34] = [0, 0, 0]
            s[-16:] = [0, 0, 0]
            s = s[24:]
            frames.append(s)
        normalized.append(frames)
    return np.array(normalized)

def onehot_to_pixels(states):
    pixel_images = np.zeros(states.shape[:-1] + (3,))
    for i, onehot_image in enumerate(states):
        # import pdb; pdb.set_trace()
        for x, _ in enumerate(onehot_image):
            for y, _ in enumerate(onehot_image[x]):
                index = np.argmax(onehot_image[x][y])
                pixel_images[i][x][y] = type_to_pixel[index]
    return pixel_images

def softmax(x, axis):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=axis)