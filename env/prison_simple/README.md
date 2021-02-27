Simplified version of the Prison map in butterfly environments as part of petting zoo. 


The observation is of size 35 + 2 , where they move `abs(x), x ~ N(5, 2)` pixels either left or right (or do nothing), The observation is a randomly generated "blob" centered in their position in order to provide some noisy ness, the extra 2 parameters are whether the left/right wall has been activated by the agent (we'll make them positive number `abs(x), x ~ N(5, 2)`), the blob will be a sample of 20 points, distributed `N(a, 2)`, where `a` is where the agent is located, and we'll attribute the locations by rounding down.

Agent receives reward of 1 if both left and right are activated, otherwise 0, and the counters reset.
