Please read ALGO_NOTES.md for rationale. 

Flat environments via stacked frames:

[x] IQL
[x] VDN
[x] QMIX

[x] IAC
[x] SEAC - implemented with lambda = 1 as per paper
[x] Central-V
[x] COMA

[x] IPG
[x] MADDPG

Recurrent environments via GRU:

[x] IQL
[x] VDN
[x] QMIX

[x] IAC - GRU ALL - very slow
[x] IAC - GRU Actor, MLP Critic, better comparison with IQL - vectorised a bit
[x] SEAC - implemented with lambda = 1 as per paper
[x] Central-V - this is IAC, but instead of parameter sharing the critic, we will use the state repr.
[x] COMA - only the actor has GRU layer

[x] IPG
[x] MADDPG

Then the non-independent variations require dense graph pooling variations? But it doesn't make sense from a mixing perspective. So either, we have a heuristic way of pooling (e.g. global mean pool) or zero padded. We can play around with a non-monotonic dense graph pool. 

----

Next aim to refactor everything so that it supports "ragged" environments...