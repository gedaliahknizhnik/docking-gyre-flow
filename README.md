# Docking in Gyre-Like Flows

This repository is a sub-project of the [Modboats](https://www.modlabupenn.org/modboats/) project. Our goal is to develop controllers that utilize the energy in flow-like environments to help Modboats dock with each other.

## Background

Modboats have been shown to be able to ride gyre-like flows with limited information and can demonstrate at least 70% energy savings when doing so[^1]. We now aim to show that we can use this approach to find and dock to other Modboat modules.

The control approach involves thrusting either towards or away from the gyre center to adjust the boat's radial position, and thus its angular velocity. The controller brings the radius of the active boat to the radius of the target as the phase difference comes to zero, resulting in a dock.

[^1]: G. Knizhnik, P. Li, X. Yu and M. A. Hsieh, "Flow-Based Control of Marine Robots in Gyre-Like Environments," 2022 International Conference on Robotics and Automation (ICRA), 2022, pp. 3047-3053, [doi: 10.1109/ICRA46639.2022.9812331](https://ieeexplore.ieee.org/document/9812331).

## Simulation

The goal of this repository in particular is to simulate the controllers we are developing in gyre models and verify that they converge. Usage information and results will be added here as the repository progresses, but the main file to run is `simulation.py`.