using CUDA
using KernelAbstractions
using KernelAbstractions.Extras: @unroll
using Adapt
using Enzyme
using LinearAlgebra
using Random
using DifferentiationInterface

#==== Physics & GPU kernel setup ====#
const PHYS = (
    γ = 42.58e6 * 2π,  # gyromagnetic ratio in Hz/T
    Tscale = 1.0       # unit scale for Δf/Δt if needed
)

@kernel unsafe_indices=true inbounds=true function excitation!(
    M_xy::AbstractVector{Complex{T}}, M_z::AbstractVector{T},
    p_x::AbstractVector{T}, p_y::AbstractVector{T}, p_z::AbstractVector{T},
    ΔBz::AbstractVector{T}, T1::AbstractVector{T}, T2::AbstractVector{T}, ρ::AbstractVector{T},
    Gx::AbstractVector{T}, Gy::AbstractVector{T}, Gz::AbstractVector{T},
    Δt::T, B::T, M0::T
) where {T}
    i = @index(Global, Linear)
    if i <= length(M_z)

        Mxy = M_xy[i]
        mz  = M_z[i]

        γ = PHYS.γ

        ω = γ * B

        cross_x =  ω * (im * Mxy)
        cross_z = 0
        relax_x = -real(Mxy)/T2[i]
        relax_y = -imag(Mxy)/T2[i]
        relax_z = (mz - M0)/T1[i]

        Mxy_new = Mxy + Complex(relax_x, relax_y) * Δt + cross_x * Δt
        mz_new   = mz   + relax_z * Δt
        M_xy[i] = Mxy_new
        M_z[i]  = mz_new
    end
end


const GPU = CUDABackend()
function step_gpu!(M_xy, M_z, p_x, p_y, p_z, ΔBz, T1, T2, ρ, dt, B, M0)
    n = length(M_z)
    threads = 256
    blocks  = cld(n, threads)
    ker = excitation!(GPU, threads)
    ker(M_xy, M_z, p_x, p_y, p_z, ΔBz, T1, T2, ρ,
        nothing, nothing, nothing,
        dt, B, M0;
        ndrange = (blocks,))
    KernelAbstractions.synchronize(GPU)
    return
end

target = rand(Float32, 3)
x0 = rand(Float32, 3)
params = (
    γ = PHYS.γ, 
    B = 1f-6, 
    T1 = 1.0f0, 
    T2 = 0.5f0, 
    M0 = 1.0f0)
dt = 0.001f0
tmax = 1.0f0

function solve_gpu(m0, dt, tmax, params)
    Nsteps = Int(round(tmax/dt))

    M_xy = fill(Complex{Float32}(m0[1], m0[2]), length(m0)) |> adapt(GPU)
    M_z  = fill(m0[3], length(m0)) |> adapt(GPU)

    n = length(m0)
    p_x = fill(m0[1], n) |> adapt(GPU)
    p_y = fill(m0[2], n) |> adapt(GPU)
    p_z = fill(m0[3], n) |> adapt(GPU)
    ΔBz= zeros(Float32,n) |> adapt(GPU)
    T1v = fill(params.T1,n) |> adapt(GPU)
    T2v = fill(params.T2,n) |> adapt(GPU)
    ρv  = fill(params.M0,n) |> adapt(GPU)
    
    for _ in 1:Nsteps
        step_gpu!(M_xy, M_z, p_x, p_y, p_z, ΔBz, T1v, T2v, ρv, dt, params.B, params.M0)
    end

    Mf = Array(M_z)[1]
    return Mf
end

f = x -> begin
    final = solve_gpu(x, dt, tmax, params)
    return sum(abs2, final .- target[3])
end


function gradient_descent!(x, α)
    val, grad = value_and_gradient(f, AutoEnzyme(;mode=Reverse), x)
    println("Current value: ", val)
    println("Gradient: ", grad)
    return x .- α .* grad, val
end


x = x0
α =1f-2
iters = 1000


for i in 1:iters
    val, grad = value_and_gradient(f, AutoEnzyme(; mode=Reverse), x)
    x .= x .- α .* grad
end

println("Optimized x: ", x)
