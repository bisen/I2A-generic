import gym

# a simple wrapper of gym atari games
class AtariGames():
    def __init__(self, name, pixels):
        self.game = gym.make(name)

        self.name = name
        self.pixels = pixels
        self.num_actions = self.game.action_space.n


pong_pixels = (
    (144, 72, 17),      # background
    (0, 0, 0),          # boundary
    (213, 130, 74),     # opponent
    (236, 236, 236),    # ball
    (92, 186, 92),      # player
)
pong = AtariGames('Pong-v0', pong_pixels)

