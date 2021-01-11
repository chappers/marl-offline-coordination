This directory contains the experiments to run. These are ran via:

```
python -m experiement.<path to experiment>
```

We will generally stay consistent w.r.t. hyperparameters, and will rely on previous literature to set them appropriately.

We have noticed that previous literature:

*  Some papers leverage GRU heavily
*  Some papers only use MLPs, with stacked frames as preprocessing instead

We have tried to show the variations between the two approaches here, as it is difficult to replicate all results in all papers in a consistent way (i.e. some algorithms perform strongly on GRU, as the agents already can solve them independently, whereas in the MLP setting, extra information helps the agents immensely)

The default configuration is based on the defaults provided by:

*  pymarl
*  rllib
*  rlkit

depending on the algorithm. We have made our own judgement call on how to normalise the hyperparameters where they somewhat collide.

In general the hyperparameters used are (as per pymarl - GRU):

*  agents: 64
*  mixing embed: 32

