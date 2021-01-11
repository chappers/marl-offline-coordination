# marl-offline-coordination

> Theme: how can we generalise (easily?) when scaling up multi-agents and have it perform in a stable fashion? The proposal is: train on lower number of agents, then scale up the number of agents - however many approaches will fail for a variety of reasons in this setting (hypothesis); what modifications do we think we can consider in order to enable this within this particular problem space? The proposed approach is to create multi-agents which can adapt to differing number of agents, and can scale up (in a graph structure) so that we can better combine this knowledge. 

This is going to use the butterfly environments and offline RL setup in order to tease out efficient MARL algorithms. 
The idea is how can we fine-tune policies from one set of multi-agent environments to another, when considering cooperative nature of the environments. Is there a way for this transfer of knowledge to happen? What if it is completely degenerative? How can our policies control for the fact that the information is plain "bad"? 

We'll look at how to:

*  Use SAC as a benchmark
*  use IQL with DQN as another benchmark
*  use QMIX as a comparison
*  use BRAC/BEAR as a comparison
*  use our approach which is behavioural policies + coordination information

We'll aim to use the following environments, as we can easily configure the number of agents:

*  [Pistolball](https://www.pettingzoo.ml/butterfly/pistonball)
*  [KAZ](https://www.pettingzoo.ml/butterfly/knights_archers_zombies)
*  [Prison (for debugging, and simple baseline)](https://www.pettingzoo.ml/butterfly/prison)
*  [Multi-walker](https://www.pettingzoo.ml/sisl/multiwalker)
*  [Waterworld](https://www.pettingzoo.ml/sisl/waterworld)
*  [Pursuit](https://www.pettingzoo.ml/sisl/pursuit)
*  [Space Invaders](https://www.pettingzoo.ml/atari/space_invaders) (use the default [OpenAi Gym](https://gym.openai.com/envs/SpaceInvaders-v0/) to train for single player)


We will use marlkit to reimplement a number of the algorithms for simplicity sake. **All environments are discrete**

```
conda install pytorch==1.4.0 torchvision==0.5.0 cpuonly -c pytorch
pip install supersuit "pettingzoo[all]"
```

We will probably preprocess everything to be `[84, 84]` over grayscale

Sample from Atari:

```
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
```

See here for usage as well: https://github.com/pettingzoopaper/pettingzoopaper

## Tests

Run tests as a module!

```
python -m test.<path_to_modules>
```

Or via `pytest`

```
pytest -m test.marl
```
