import gym

# a simple wrapper of gym atari games
class AtariGames():
    def __init__(self, name, pixels):
        self.game = gym.make(name)

        self.name = name
        self.pixels = pixels
        self.num_actions = 3

pong_pixels = (
    (144, 72, 17),      # background
    (0, 0, 0),          # boundary
    (213, 130, 74),     # opponent
    (236, 236, 236),    # ball
    (92, 186, 92),      # player
)
pong = AtariGames('Pong-v0', pong_pixels)

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
    return normalized


