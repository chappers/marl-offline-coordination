from supersuit import resize, frame_skip, frame_stack, sticky_actions, color_reduction_v0
from pettingzoo.atari import space_invaders_v1

env = space_invaders_v1.env()

# repeat_action_probability is set to 0.25 to introduce non-determinism to the system
env = sticky_actions(env, repeat_action_probability=0.25)

# downscale observation for faster processing
env = resize(env, (84, 84))

# allow agent to see everything on the screen despite Atari's flickering screen problem
env = frame_stack(env, 4)

# skip frames for faster processing and less control
# to be compatable with gym, use frame_skip(env, (2,5))
env = frame_skip(env, 4)