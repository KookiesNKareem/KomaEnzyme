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
    excitation!(backend, threads)(
      M_xy, M_z, p_x, p_y, p_z, ΔBz, T1v, T2v, ρv,
      UInt32(length(M_xy)), s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1,
      UInt32(length(Δt_steps)), ndrange=blocks
    )
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

    cpu_mz = zeros(Float32, Nspins)
    M_xy = adapt(backend, cpu_mxy)
    M_z  = adapt(backend, cpu_mz)
    Δt   = adapt(backend, collect(cpu_Δt))

    p_x = adapt(backend, rand(Float32, Nspins))
    p_y = adapt(backend, rand(Float32, Nspins))
    p_z = adapt(backend, rand(Float32, Nspins))
    ΔBz = adapt(backend, zeros(Float32, Nspins))
    T1v = fill!(similar(ΔBz), params.T1)
    T2v = fill!(similar(ΔBz), params.T2)
    ρv  = fill!(similar(ΔBz), params.M0)

    s_Gx = adapt(backend, rand(Float32, length(cpu_Δt)))
    s_Gy = adapt(backend, rand(Float32, length(cpu_Δt)))
    s_Gz = adapt(backend, rand(Float32, length(cpu_Δt)))
    s_B1 = adapt(backend, rand(Complex{Float32}, length(cpu_Δt)))
    s_f  = adapt(backend, rand(Float32, length(cpu_Δt)))

    return threads, blocks, Nsteps, Δt,
           M_xy, M_z, p_x, p_y, p_z, ΔBz, T1v, T2v, ρv,
           s_Gx, s_Gy, s_Gz, Δt, s_f, s_B1
end

Random.seed!(123)
cpu_init = rand(Complex{Float32}, 10)
const backend = CUDA.CUDABackend()
const params = (T1=1.0f0, T2=0.5f0, M0=1.0f0)

dt, tmax = 0.001f0, UInt32(1)
cpu_Δt = 0.0f0:dt:tmax

threads, blocks, Nsteps, Δt,
M_xy_gpu, M_z, p_x, p_y, p_z,
ΔBz, T1v, T2v, ρv,
s_Gx, s_Gy, s_Gz, s_Δt, s_f, s_B1 =
  init_gpu_arrays(cpu_init, dt, cpu_Δt, tmax, params, backend)

target = adapt(backend, rand(Float32, length(cpu_init)))

Mx0 = real(cpu_init)
My0 = imag(cpu_init)
X = adapt(backend, Float32[vcat(Mx0, My0)...])
N = length(Mx0)

function f(X)

    Mx = @view X[1:N]
    My = @view X[N+1:2N]
    M_xy0 = complex.(Mx, My)

    M_xy_final = solve_steps!(
      M_xy0, M_z, p_x, p_y, p_z,
      ΔBz, T1v, T2v, ρv,
      Δt, backend, threads, blocks,
      s_Gx, s_Gy, s_Gz, s_Δt, s_f, s_B1
    )

    return sum(abs2, M_xy_final .- target)
end

const lr = 1f-20

for iter in 1:100
    loss, ∇X = value_and_gradient(f, AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse)), X)


    CUDA.@sync CUDA.axpy!(-lr, ∇X, X)

    println(
      "Iter $iter: loss = $(Array(loss)), ∥grad∥ = $(norm(∇X))"
    )
end


Mxy_opt = complex.(X[1:N], X[N+1:2N])

Mxy_cpu = Array(Mxy_opt)
spin_idx = 1:length(Mxy_cpu)
magnitude = abs.(Mxy_cpu)

ENV["GKSwstype"] = "100"
using Plots
gr()

p = plot(
  spin_idx, magnitude;
  xlabel="Spin index",
  ylabel="|M_xy|",
  title="Magnitude of M_xy",
  legend=false
)
savefig(p, "mxy_plot.png")
println("✔ Plot saved to mxy_plot.png")