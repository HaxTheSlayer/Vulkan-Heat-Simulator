# Vulkan 2D Heat Equation Simulator

A real-time, GPU-accelerated 2D heat equation simulator built with Vulkan and C++. This project models the isotropic diffusion of heat across an 800x800 grid using a Forward Time, Centered Space (FTCS) finite difference scheme.

## Overview
This project was developed to bridge the gap between continuous mathematics (Partial Differential Equations) and raw hardware performance. By offloading the numerical analysis to a Vulkan Compute Shader, the simulation runs at a smooth 60 FPS while calculating thousands of cell interactions per frame.

### Key Features
* **Real-Time Interactivity:** Users can use their mouse to "paint" extreme heat onto the grid in real-time, watching it dynamically diffuse according to the laws of thermodynamics.
* **GPU Compute Shader:** The math is entirely offloaded to the GPU using a custom GLSL compute shader.
* **Ping-Pong Storage Buffers:** Seamlessly swaps the Nth and (N+1)th state arrays in memory without expensive CPU-GPU data transfers.
* **Full-Screen Triangle Rendering:** Utilizes a vertex shader trick to render the grid with a single triangle, passing the computation data to a fragment shader to generate a dynamic heat color gradient.

## The Mathematics
The simulation solves the 2D Heat Equation:
∂u/∂t = α (∂²u/∂x² + ∂²u/∂y²)

This is discretized using the **Forward Euler Method** (FTCS) in the compute shader to calculate the temperature of each cell for the next time step. To ensure numerical stability, spatial and temporal step sizes were balanced with the thermal diffusivity constant (α).

## Acknowledgments
A massive thank you to [Sascha Willems](https://github.com/SaschaWillems/Vulkan) for his incredible open-source Vulkan C++ examples. His repositories were an invaluable resource for understanding Vulkan's synchronization, pipeline setup, and memory management architecture.
