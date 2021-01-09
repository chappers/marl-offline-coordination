Please read ALGO_NOTES.md for rationale. 

Flat environments via stacked frames:

[x] IQL
[x] VDN
[x] QMIX

[x] IAC
[x] SEAC - implemented with lambda = 1 as per paper
[ ] Central-V
[x] COMA

[ ] IPG
[ ] MADDPG

Recurrent environments via GRU:

[x] IQL
[x] VDN
[x] QMIX

[x] IAC - GRU ALL - very slow
[x] IAC - GRU Actor, MLP Critic, better comparison with IQL - vectorised a bit
[x] SEAC - implemented with lambda = 1 as per paper
[ ] Central-V - this is IAC, but instead of parameter sharing the critic, we will use the state repr.
[x] COMA - only the actor has GRU layer

[ ] IPG
[ ] MADDPG

Then the non-independent variations require dense graph pooling variations