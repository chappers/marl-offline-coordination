
```sh
brew install swig
pip install -r requirements.txt
```

We'll choose the easier environments (pursuit and waterworld)

- pursuit
- pursuit small
- pursuit medium
- waterworld
- waterworld small
- waterworld medium
- reference
- spread
- tag

---

Residual Network behave like boosting - uses the resnet style module to demonstrate superior performance over IQL that doesn't use it + IQL with sharing (shrinkage of VDN is basically what the resnet does - technically we're stacking additively, but thats the skip connection and secret sauce)

Ablation:

* IQL
* IQL with resnet
* IQL with resnet w/ shrinkage
* VDN with resnet w/ shrinkage (or QMix with embedding size of 1)

---

DBR just implement it naively and demonstrate how it performs compared with BEAR

SEAC
IAC
BEAR (MARL)
DBR (MARL)

in the research setting, both BEAR and DBR use shared experience to trigger to make use of offline data efficiently