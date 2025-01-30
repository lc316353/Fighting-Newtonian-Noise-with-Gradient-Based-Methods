# Code used to calculate the results of https://arxiv.org/abs/2411.03251

## Fighting Gravity Gradient Noise with Gradient-Based Optimization Methods

#### Abstract:

Gravity gradient noise in gravitational wave detectors originates from density fluctuations in the adjacency of the interferometer mirrors. At the Einstein Telescope, this noise source is expected to be dominant for low frequencies. Its impact is proposed to be reduced with the help of an array of seismometers that will be placed around the interferometer endpoints. 
We reformulate and implement the problem of finding the optimal seismometer positions in a differentiable way. We then explore the use of first-order gradient-based optimization for the design of the seismometer array for 1 Hz and 10 Hz and compare its performance and computational cost to two metaheuristic algorithms. For 1 Hz, we introduce a constraint term to prevent unphysical optimization results in the gradient-based method. 
In general, we find that it is an efficient strategy to initialize the gradient-based optimizer with a fast metaheuristic algorithm. For a small number of seismometers, this strategy results in approximately the same noise reduction as with the metaheuristics. For larger numbers of seismometers, gradient-based optimization outperforms the two metaheuristics by a factor of 2.25 for the faster of the two and a factor of 1.4 for the other one, which is significantly outperformed by gradient-based optimization in terms of computational efficiency.


##### Comments:

We are aware that this code is probably not very user-friendly, but we think this is better than not being transparent. Maybe we find the time to make an improved version of the code and place it here.


Last Update: 30.01.2025
