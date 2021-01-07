Algorithm Notes
===============

Parameter sharing w/ individual networks are implemented as per the QMIX approach in IQL and QMIX, via the usage of `obs_agent_id` field.

Using this naively will also allow for items as per here: https://arxiv.org/pdf/2006.07169.pdf and here: https://arxiv.org/abs/2006.07869

IQL: supported with just usage of DQN 
IAC: supported through usage of SAC or more accurately A2C (we can do this via the discretisation proposal as in the other paper?) They used A2C, we can use AWAC variation as the building blocks for this approach.

None of the actual multi-agent algorithms are complete.

Central-V: actors are trained on observation, critic uses only the mixer (to do): https://github.com/AnujMahajanOxf/MAVEN/tree/master/maven_code/src/modules/critics
COMA: actors are trained on observation, look here: https://github.com/oxwhirl/pymarl/blob/master/src/learners/coma_learner.py and here: https://github.com/AnujMahajanOxf/MAVEN/tree/master/maven_code/src/modules/critics and here: https://github.com/AnujMahajanOxf/MAVEN/blob/master/maven_code/src/learners/coma_learner.py
MADDPG - maybe we'll do a DDPG variant based on TD3 instead. https://github.com/openai/maddpg, it is suggested jsut to use a gumbel-softmax as the output for discrete action space, and the critic is **unshared** and is just the concat of the observations and actions. 

---

Note all of these environments can be used for fine tuning, as the petting zoo environments have a parameter which is the "max agents" parameter which allows it to appropriately initialize the size for all the networks before evaluation - we just won't be updating the full network to convergence as a lot of the inputs would be blank! 

Our paper looks at how we can overcome this using appropriate and flexible structures (pooling of active agents) (namely graphs)? and regularising policy/updates. 

---

Using GRU in the naive case is like 3 times slower - can we just run one experiment with GRUs and then exclude it?

---

The focus of the paper should be on how a generalised approach to building embeddings for the critics benefits a wide range of algorithms including:

*  QMIX, where the hypernetwork is also something which uses graph pooling
*  Central-V, with SAC as the base
*  COMA, with graph pooling over the top of COMA
*  MADDPG, with graph pooling over the joint observation and action space

Assist with the transfer and fine tuning of the algorithm without focussing too hard of the algorithm specifics. We then talk about the fine-tuning aspect which deals more specifically with:

1.  the deficiencies with DQN and when you train off single ones and expand to multiple
2.  the advantages of SAC which allows for this, with reference to other papers (BEAR, AWAC)
3.  Talk about ways of regularising the policy to reflect this, and conclude with the final architecture choice (or the lack thereof, as it works with one shot)

**Ablations to justify choice**

-  Performance of 0 shot networks when using graph networks as the feature generation for the critic/value networks (in MADDPG, it is the joint action obs, similar to COMA, QMIX, Central-V and others); we can see which ones do better
-  Once we have the 0 shot networks, how we can then fine tune safely TRPO style or otherwise?

Or alternatively, train on a smaller toy problem, and evaluate on a much harder environment with more agents. To determine ability to generalise! We can then compare the choices in networks with the graph networks, this shows the degradation in performance issues(?). To summarise:
*  Training environment can have different number of agents
*  Evaluation environment only has max number of agents

We expect the convergence to be much faster where a dense graph embedding (pooling) is used irrespective of the algorithm versus with the "sparse" set. 

**How should we self regularise?**

When dealing with regularization in the context of parameter sharing in the networks; there is an expectation that, if done correctly, there shouldn't be too much "degenerative" behaviour. We can self-regularise using MRL with other network policies rather than our own(?), especially since the tail end of the observations are extremely sparse, but probably fairly informative. 

The notional idea is that if we regularise the behavior with other (agents which have been present), the "lower numbered" agents would be present "more often", and will help in the regularisation process, compared with the agents which "aren't present often", we can use munchausen variation for ease of implementation


