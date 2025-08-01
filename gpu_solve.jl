using CUDA
using KernelAbstractions
using Adapt
using Enzyme
using LinearAlgebra
using Random
using .EnzymeRules
using KernelAbstractions: NDRange, StaticSize
include("miniKomaCore.jl")

const γ = 42.58e6 * 2π

function solve_steps!(M_xy, M_z, p_x, p_y, p_z, ΔBz, T1v, T2v, ρv,
                      Δt_steps, backend, threads, blocks,
                      s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1)
    excitation!(backend, threads)(M_xy, M_z, p_x, p_y, p_z, ΔBz, T1v, T2v, ρv,
               UInt32(length(M_xy)), s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1,
               UInt32(length(Δt_steps)), ndrange=blocks)
    CUDA.synchronize()
    return M_xy
end

function init_gpu_arrays(cpu_mxy::Vector{Complex{Float32}}, dt::Float32,
                          cpu_Δt::AbstractRange{Float32}, tmax::UInt32,
                          params, backend)
    Nsteps = ceil(Int, tmax / dt)
    threads = 256
    Nspins  = length(cpu_mxy)
    blocks  = (Nspins + threads - 1) ÷ threads
    # prepare arrays
    cpu_mz = zeros(Float32, Nspins)
    M_xy = adapt(backend, cpu_mxy)
    M_z  = adapt(backend, cpu_mz)
    Δt   = adapt(backend, collect(cpu_Δt))
    # random coords for demo
    p_x = p_y = p_z = adapt(backend, rand(Float32, Nspins))
    ΔBz = adapt(backend, zeros(Float32, Nspins))
    T1v = fill!(similar(ΔBz), params.T1)
    T2v = fill!(similar(ΔBz), params.T2)
    ρv  = fill!(similar(ΔBz), params.M0)
    # step sequences
    s_Gx = adapt(backend, rand(Float32, length(cpu_Δt)))
    s_Gy = adapt(backend, rand(Float32, length(cpu_Δt)))
    s_Gz = adapt(backend, rand(Float32, length(cpu_Δt)))
    s_B1 = adapt(backend, rand(Complex{Float32}, length(cpu_Δt)))
    s_f  = adapt(backend, rand(Float32, length(cpu_Δt)))
    return threads, blocks, Nsteps, Δt,
           M_xy, M_z, p_x, p_y, p_z, ΔBz, T1v, T2v, ρv,
           s_Gx, s_Gy, s_Gz, Δt, s_f, s_B1
end

# Autodiff setup
Random.seed!(123)
cpu_init = rand(Complex{Float32}, 10)
const backend = CUDA.CUDABackend()
const params = (T1=1.0f0, T2=0.5f0, M0=1.0f0)

dt, tmax = 0.001f0, UInt32(1)
cpu_Δt = 0.0f0:dt:tmax
threads, blocks, Nsteps, Δt,
M_xy, M_z, p_x, p_y, p_z,
ΔBz, T1v, T2v, ρv,
s_Gx, s_Gy, s_Gz, s_Δt, s_f, s_B1 = init_gpu_arrays(cpu_init, dt, cpu_Δt, tmax, params, backend)

target = adapt(backend, rand(Float32, 10))



# Objective
function f(M_xy0)
    M_xy = solve_steps!(M_xy0, M_z, p_x, p_y, p_z,
                           ΔBz, T1v, T2v, ρv,
                           Δt, backend, threads, blocks,
                           s_Gx, s_Gy, s_Gz, s_Δt, s_f, s_B1)
    return gpu_sum(M_xy, target)
end

# Gradient descent loop
grad = similar(M_xy)
for i in 1:100
    # 1) Forward pass
    acc = f(M_xy)        
    val = Array(acc)[1]       
    
    # 2) Reverse‐mode Jacobian (1×N) as a 1‐tuple
    (Jmat,) = jacobian(Reverse, f, M_xy)

    # 3) Copy the gradient into your buffer
    g = copy(vec(Jmat))  
    CUDA.copyto!(grad, g)

    println("Iter $i: val = $val, ∥grad∥=$(norm(grad))")
end