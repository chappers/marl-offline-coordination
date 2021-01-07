import numpy as np


def multitask_rollout(
    env,
    agent,
    max_path_length=np.inf,
    render=False,
    render_kwargs=None,
    observation_key=None,
    desired_goal_key=None,
    get_action_kwargs=None,
    return_dict_obs=False,
):
    if render_kwargs is None:
        render_kwargs = {}
    if get_action_kwargs is None:
        get_action_kwargs = {}
    dict_obs = []
    dict_next_obs = []
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    next_observations = []
    path_length = 0
    agent.reset()
    o = env.reset()
    if render:
        env.render(**render_kwargs)
    goal = o[desired_goal_key]
    while path_length < max_path_length:
        dict_obs.append(o)
        if observation_key:
            o = o[observation_key]
        new_obs = np.hstack((o, goal))
        a, agent_info = agent.get_action(new_obs, **get_action_kwargs)
        next_o, r, d, env_info = env.step(a)
        if render:
            env.render(**render_kwargs)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        next_observations.append(next_o)
        dict_next_obs.append(next_o)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    next_observations = np.array(next_observations)
    if return_dict_obs:
        observations = dict_obs
        next_observations = dict_next_obs
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
        goals=np.repeat(goal[None], path_length, 0),
        full_observations=dict_obs,
    )


def rollout(
    env,
    agent,
    max_path_length=np.inf,
    render=False,
    render_kwargs=None,
):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos
    """
    if render_kwargs is None:
        render_kwargs = {}
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    agent.reset()
    next_o = None
    path_length = 0
    if render:
        env.render(**render_kwargs)
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if render:
            env.render(**render_kwargs)

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack((observations[1:, :], np.expand_dims(next_o, 0)))
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )


def marl_rollout(
    env,
    agent,
    max_path_length=np.inf,
    render=False,
    render_kwargs=None,
):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos

    ---

    For multi-agent rollout, the global state HAS TO be generated here, particularly
    when considering the MAVEN setup. make sure you use the wrapper to support these

    *  observations
    *  states
    *  states_0

    petting zoo uses attr `agents` to indicate active agents. This needs to be used to filter
    for storing the paths and rollouts somehow
    """
    ENV_OBS = "obs"
    ENV_STATE = "state"
    ENV_STATE_0 = "state_0"
    action_space = env.multi_agent_action_space.n

    if render_kwargs is None:
        render_kwargs = {}
    observations = []
    states = []
    states_0 = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []

    # because we use parallel enviornments, we should ideally sort the outputs first, and then
    # change to a list, for consistency
    o = env.reset()
    agent.reset()
    next_o = None
    path_length = 0
    if render:
        env.render(**render_kwargs)
    while path_length < max_path_length:
        a, agent_info = agent.get_action(
            o
        )  # need to make sure this is implemented correctly too
        next_o, r, d, env_info = env.step(a)
        o_ = []
        s_ = []
        s0_ = []
        r_ = []
        t_ = []
        a_ = []
        ai_ = []
        ei_ = []
        # it should filter based on the agents which are active
        for idx, ag_name in enumerate(env._wrapped_env.possible_agents):
            if ag_name in env._wrapped_env.agents:
                o_.append(o[idx][ENV_OBS])
                s_.append(o[idx][ENV_STATE])
                s0_.append(o[idx][ENV_STATE_0])
                r_.append(r[idx])
                t_.append(d[idx])
                a__ = [0 for _ in range(action_space)]
                a__[a[idx]] = 1
                a_.append(a__)
                if len(agent_info) == 0:
                    ai_.append({})
                else:
                    ai_.append(agent_info[idx])
                if len(env_info) == 0:
                    ei_.append({})
                else:
                    ei_.append(env_info[idx])
        observations.append(o_)
        states.append([s_[0]])
        states_0.append([s0_[0]])
        rewards.append([r_])
        terminals.append([t_])
        actions.append(a_)
        agent_infos.append([ai_])
        env_infos.append([ei_])
        path_length += 1
        if all(d):
            break
        o = next_o
        if render:
            env.render(**render_kwargs)

    o_ = []
    s_ = []
    s0_ = []
    for idx, ag_name in enumerate(env._wrapped_env.possible_agents):
        if ag_name in env._wrapped_env.agents:
            o_.append(o[idx][ENV_OBS])
            s_.append(o[idx][ENV_STATE])
            s0_.append(o[idx][ENV_STATE_0])
    actions = np.array(actions)
    observations = np.array(observations)
    states = np.array(states)
    states_0 = np.array(states_0)

    next_observations = np.array([o_])

    #' I presume these might be needed if we ever implement QTRAN?
    # next_states = np.array([s_[0]])
    # next_states_0 = np.array([s0_[0]])

    next_observations = np.concatenate(
        (
            observations[1:, :, :],
            next_observations,
        ),
        axis=0,
    )
    # next_states = np.concatenate((states[1:, :, :], next_states), axis=0)
    # next_states_0 = np.concatenate((states_0[1:, :, :], next_states_0), axis=0)

    return dict(
        observations=observations,
        states=states,
        states_0=states_0,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        # next_states=next_states,
        # next_states_0=next_states_0,
        terminals=terminals,
        agent_infos=agent_infos,
        env_infos=env_infos,
    )


def function_rollout(
    env,
    agent_fn,
    max_path_length=np.inf,
    render=False,
    render_kwargs=None,
):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos
    """
    if render_kwargs is None:
        render_kwargs = {}
    observations = []
    actions = []
    rewards = []
    terminals = []
    env_infos = []
    o = env.reset()
    next_o = None
    path_length = 0
    if render:
        env.render(**render_kwargs)
    while path_length < max_path_length:
        a = agent_fn(o)
        try:
            if not (np.isfinite(a).all()):
                print("non finite action found, sampling at random...")
                a_rand = env.action_space.sample()  # hack for sac breaking
                a[~np.isfinite(a)] = a_rand[~np.isfinite(a)]
        except Exception as e:
            print("Exception found - sampling random action", e)
            a = env.action_space.sample()

        next_o, r, d, env_info = env.step(a)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if render:
            env.render(**render_kwargs)

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack((observations[1:, :], np.expand_dims(next_o, 0)))
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        env_infos=env_infos,
    )
