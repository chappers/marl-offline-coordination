Algorithm Notes
===============

Parameter sharing w/ individual networks are implemented as per the QMIX approach in IQL and QMIX, via the usage of `obs_agent_id` field.

Using this naively will also allow for items as per here: https://arxiv.org/pdf/2006.07169.pdf and here: https://arxiv.org/abs/2006.07869

IQL: supported with just usage of DQN 
IAC: supported through usage of SAC or more accurately TD3

None of the actual multi-agent algorithms are complete.

Central-V: actors are trained on observation, critic uses only the mixer (to do): https://github.com/AnujMahajanOxf/MAVEN/tree/master/maven_code/src/modules/critics
COMA: actors are trained on observation, look here: https://github.com/oxwhirl/pymarl/blob/master/src/learners/coma_learner.py and here: https://github.com/AnujMahajanOxf/MAVEN/tree/master/maven_code/src/modules/critics and here: https://github.com/AnujMahajanOxf/MAVEN/blob/master/maven_code/src/learners/coma_learner.py
MADDPG - maybe we'll do a DDPG variant based on TD3 instead. https://github.com/openai/maddpg, it is suggested jsut to use a gumbel-softmax as the output for discrete action space, and the critic is **unshared** and is just the concat of the observations and actions. 

---

Note all of these environments can be used for fine tuning, as the petting zoo environments have a parameter which is the "max agents" parameter which allows it to appropriately initialize the size for all the networks before evaluation - we just won't be updating the full network to convergence as a lot of the inputs would be blank! 

Our paper looks at how we can overcome this using appropriate and flexible structures (pooling of active agents) (namely graphs)? and regularising policy/updates. 


