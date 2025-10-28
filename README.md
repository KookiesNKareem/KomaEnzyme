# KomaEnzyme

⚠️ **Work in Progress** — This repository is under active development. APIs and implementations may change.

RF pulse optimization for MRI using automatic differentiation with Enzyme.jl and KomaMRICore.jl.

## Overview

This repository implements gradient-based optimization of RF pulses for selective excitation in MRI using Enzyme.jl for automatic differentiation. The code performs Bloch equation simulations on CPU and GPU backends via KernelAbstractions.jl, enabling efficient computation of gradients for large-scale pulse design problems.

## Features

- **Multiple AD backends**: Forward-mode, reverse-mode (Enzyme), and finite differences
- **CPU & GPU support**: KernelAbstractions.jl enables portable kernels across backends
- **Phase-matched loss**: Compensates for arbitrary global phase during optimization
- **1D & 2D pulse design**: Slice-selective and 2D spatially-selective RF pulses
- **Butterworth target profiles**: Smooth transition bands for realistic excitation patterns

## Scripts

- `1d_cpu_reverse.jl` — 1D slice-selective RF optimization (CPU, Enzyme reverse-mode)
- `1d_cpu_forward.jl` — 1D optimization using forward-mode AD
- `1d_cpu_finitediff.jl` — 1D optimization using finite differences
- `1d_gpu_forward.jl` — 1D optimization on GPU with forward-mode AD
- `1d_gpu_reverse.jl` — 1D optimization on GPU with Enzyme reverse-mode
- `2d_gpu_reverse.jl` — 2D spatially-selective pulse design on GPU

## Requirements

**Important**: This code requires specific versions to work correctly.

- **Julia 1.10** (tested with 1.10.x, compatibility with newer versions not guaranteed)
- **CUDA-capable GPU** (for GPU scripts)
- **Specific package versions** (see `Project.toml`):
  - Enzyme.jl (for automatic differentiation)
  - KomaMRICore v0.9.5
  - KernelAbstractions v0.9.38
  - CUDA v5.8.3

## Installation

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()  # Installs exact versions from Project.toml
```

**Note**: Due to rapid development in the Enzyme.jl and KomaMRICore ecosystems, using different package versions may lead to compatibility issues or incorrect gradient computations. The `Project.toml` pins tested versions.

## Usage

Run any script directly:

```bash
julia --project=. 1d_cpu_reverse.jl
```

For GPU scripts, ensure CUDA is properly configured:

```bash
julia --project=. 1d_gpu_reverse.jl
```

## Key Parameters

- `Nspins`: Number of spatial locations (100 for CPU demos, 100k+ for benchmarks)
- `Nrf`: Number of RF control points (typically ~256)
- `TBP`: Time-bandwidth product for slice profile sharpness
- `η_base`: Learning rate for gradient descent

## Methods

The optimization minimizes a phase-matched mean squared error between simulated and target magnetization profiles. The phase-matching compensates for arbitrary global phase φ₀ and residual k-space moment κ = γ∫G(t)dt, ensuring physically meaningful gradients during optimization.

## Output

Scripts generate plots showing:
1. Optimized RF waveform (real/imaginary components)
2. Achieved vs target slice profile magnitude

Results are saved to `profile_and_rf.png`.

## Citation

If you use this code, please cite:
- [KomaMRI](https://github.com/JuliaHealth/KomaMRI.jl)
- [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl)

## Author

Kareem Fareed, Carlos Castillo Passi

## License

See LICENSE file for details.
