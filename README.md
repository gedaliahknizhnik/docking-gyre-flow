# Docking in Gyre-Like Flows

This repository is a sub-project of the [Modboats](https://www.modlabupenn.org/modboats/) project. Our goal is to develop controllers that utilize the energy in flow-like environments to help Modboats dock with each other.

## Background

Modboats have been shown to be able to ride gyre-like flows with limited information and can demonstrate at least 70% energy savings when doing so[^1]. We now aim to show that we can use this approach to find and dock to other Modboat modules.

The control approach involves thrusting either towards or away from the gyre center to adjust the boat's radial position, and thus its angular velocity. The controller brings the radius of the active boat to the radius of the target as the phase difference comes to zero, resulting in a dock.

[^1]: G. Knizhnik, P. Li, X. Yu and M. A. Hsieh, "Flow-Based Control of Marine Robots in Gyre-Like Environments," 2022 International Conference on Robotics and Automation (ICRA), 2022, pp. 3047-3053, [doi: 10.1109/ICRA46639.2022.9812331](https://ieeexplore.ieee.org/document/9812331).

## Controllers

The simulation environment supports two controllers:

1. Flow-Based Rendezvous and Docking (FBRD) - as described above, this controller enables rendezvous by thrusting radially to faster water to catch up and to slower water to slow down, until the Modboat is within one radius of the target. (The other stages of FBRD - following and docking - as described in the paper, are not implemented here).
2. Naive approach - simply thrust towards the target's current position until it is reached.

## Simulation

The goal of this repository in particular is to simulate the controllers we are developing in gyre models and verify:
1. That they converge, especially in cases where experimental evaluation runs into boundary conditions due to limited space.
2. That they are robust to noise in the flow model.
3. That the cost of FBRD scales more optimally than the cost of the naive approach. Simulations show that the naive approach has cost that grows linearly with the gyre dimension, while FBRD has constant cost. 

Usage occurs within the module `simulation.py`. 


### Parameters:

You must choose a controller to use:

```python
# EITHER
_CONTROLLER_TO_USE = FBRDController
# OR
_CONTROLLER_TO_USE = NaiveController
```

Choose a level of noise to use. Noise is injected as zero-mean Gaussian noise, where the noise level is the standard deviation of the Gaussian distribution, and its units are [m/s].

```python
_NOISE_LEVEL_M_PER_S = 0.000
```

Choose the number of iterations:

```
_NUM_ITERS = 50
```

Lastly, make sure to set a naming string for the simulation, which will identify the files it creates.

```python
_OUTPUT_NAME = "with_noise"
```

### Setup 

With the parameters set, the remaining task is to choose which of the following simulations to run:

```python

def main():

    # To run a single simulation
    run_simulation_single(output_name=_OUTPUT_NAME)
    
    # To evaluate repeatability, convergence, etc.
    run_simulation_many_rankine_vortex(output_name=_OUTPUT_NAME)
    
    # To evaluate scaling
    run_simulation_many_rankine_vortex_changing_radius(output_name=_OUTPUT_NAME)
```

The actual simulation is run by the function `run_simulation()`, which can be modified as well.

### Output:

Each of the simulations above outputs 3 files, stamped with the `_OUTPUT_NAME` string and the `_NUM_ITERS` quantity as well.

1. A `data` file, which contains a list of  `SimulationOutput` objects that can be used to fully reconstruct all the performed simulations.
    * This file tends to be **very large**, especially as `_NUM_ITERS` grows.
2. An `output` file that contains summary data for the simulations run, and can be used for cost/success statistics.
3. A `params` file that contains `SimulationParams` objects that save info about the flowfield.
 
### Advanced

Advaned flow parameters, such as the particular flow model and its parameters, can be modified within each of the above functions. Available flow models can be found in the `flowfield` module.