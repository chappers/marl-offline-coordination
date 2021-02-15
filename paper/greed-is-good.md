# Greed is Good - Learning When to Ignore Cooperation in Multi-Agent Reinforcement Learning

> In QCGraph, we looked at how coordination via graph networks between agents can be useful in contructing $Q$ mixing function. In this paper, we'll look instead at "anti-coordination" around how we can estimate a conversative $Q$ function (through the behaviour policy) to learn when we should or shouldn't mix individual $Q$ values. 

Recent multi-agent reinforcement learning typically focusses around the credit assignment problem (QMIX, LICA), and how to leverage information from other agents to learn efficiently (SEAC). However, inspite of this, there are many instances where independent approaches still out-perform approaches which consider cooperation and communication. In this work, we'll leverage and aim to exploit these scenarios through estimation the projected lower bound of the $Q$ function in the independent and cooperative scenarios. 

Papers to read:

*  Conservative Q Learning (Neurips 2020)
*  Shared Experience Actor Critic (Neurips 2020) - MARL
*  Learning Implicit Credit Assignment (Neurips 2020) - MARL

Sim. papers as before:

*  BRAC/BEAR
*  QMIX
*  MADDPG
*  QTRAN

We claim we can make a small adjustment to improve estimation of Q values in both the scnearios where there are out of sample actions and in scenarios where the best policy is infact the greedy policy.

# Method

In order to estimate a conservative lower bound, this has been explored in quantile regression q learning and conservative q learning. To control for the update in policy, we can directly penalise through munchaussen or offline approaches (such as BEAR or BRAC approaches). 

To this end the highlight is that we want to pick the Q function which does something like this:



$$Q \leftarrow \underset{Q}{\argmin} \text{ } \alpha \left( \mathbb{E}_{s \sim \mathcal{D}, a \sim \mu(a \vert s)} [Q(s, a)] - \mathbb{E}_{s \sim \mathcal{D}, a \sim \beta(a \vert s)} [Q(s, a)] \right) + \text{TD Error}$$

Whereby the LHS is the maximum based on the current policy (if we set $\mu = \pi$) and the RHS is the bound based on the behavior policy $\beta$, (which can be learned through estimating the cloned behavior via an auto-encoder).

We then either implicitly or explicit regularise using munchaussen or KL divergence (see CQL) - but this time, we want to regularise if using munchhaussen based on the alternative Q (in the Q learning scenarios), rather than itself, so that the reward signal changes(?) - double check that this is mathematically sound.

We then examine the difference as suggested, and use the reciprocal as the weight for either:
*  Lagrangian approach (i.e. we have a learnable parameter which we perform gradient descent on)
*  Direct approach (at each learning cycle, we take the min based on "how close" it is to the conservative estimate based on the behavioural policy - mimicing taking $\min(Q_1, Q_2)$ as in SAC)

For the choice of the Q network. This is where the "greedy" part of the algorithm comes in, as this determines how much we rely on independent or cooperative networks.

**Rough Idea**

We want an approach which can apply over several different approaches to verify that it works. We want something a bit different than the CQL approach, as it will just be applying CQL - not particularly interesting. 

We could think of something a bit different approach specifically around a conversative estimate in the multi-agent setting. E.g. `max(agent) - min(agent)` But this doesn't really do anything if we have parameter sharing (or does it?). 

We could argue getting conservative estimates for mixers is hard to do in a naive setting, because you have to be conservative on all fronts - maybe a comparison is if we only do a lower bound on the mixer, rather than on the agent level $Q$ function. The other item to check is that we can actually guarentee some kind of lower bound on the mixer. For example, in the QMIX setting, we probably want something different (maybe) since there is a monotonic constraint.

## Proofs and Justification

We can show (of course) if we take the min that it will be the conservative estimate - but are there other guarentees we can provide? Can we demonstrate that - if the concept truly is independent, that there is a sane way to demonstrate that the concept will be learned?

Verify that it is compatible with IGM (it should be trivial to demonstrate this) - but we also have to talk about why QMIX sometimes fails in the greedy scenario when it really shouldn't. 

We can adapt this to LICA and other models of that class, and adapt MADDPG trivially. 

# Ablations

We have several choices or items which influence our model choice:

*  How we perform the greedy choice, using lagrange or direct
*  How we regularise via munchaussen or direct
*  What we regularise against? Do we regularise against the policy of all other agents? there is some shared experience ideas here.
*  What we reglarise against - is it the distribution of the QMIX or Q independent?

