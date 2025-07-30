using CUDA, KernelAbstractions, Adapt
using Enzyme, DifferentiationInterface
using LinearAlgebra, Random
using .EnzymeRules

const PHYS = (
    γ = 42.58e6 * 2π,
    Tscale = 1.0
)

@kernel unsafe_indices=true inbounds=true function excitation!(
    M_xy::AbstractVector{Complex{T}}, M_z::AbstractVector{T},
    N_Spins::UInt32,
    N_Δt::UInt32
) where {T}
    γ = 42.58f6 * 2π

    @uniform N = @groupsize()[1]
    i_l = @index(Local,  Linear)
    i_g = @index(Group,  Linear)
    i   = (i_g - 1u32)*UInt32(N) + i_l
    # if i <= N_Spins
    #     # load per‐spin state
    #     # x, y, z      = get_spin_coordinates(p_x, p_y, p_z, i, 1)
    #     x = p_x[i]
    #     y = p_y[i]
    #     z= p_z[i]
    #     ΔBz          = p_ΔBz[i]
    #     Mxy_r, Mxy_i = reim(M_xy[i])
    #     Mz           = M_z[i]
    #     ρ            = p_ρ[i]
    #     T1           = p_T1[i]
    #     T2           = p_T2[i]
    #   for s_idx in UInt32(1):N_Δt
    #     # x, y, z = get_spin_coordinates(p_x, p_y, p_z, i, s_idx)
    #     # effective field
    #     Bz = (x * s_Gx[s_idx] + y * s_Gy[s_idx] + z * s_Gz[s_idx]) + ΔBz - s_Δf[s_idx] / γ
    #     B1_r, B1_i = reim(s_B1[s_idx])
    #     B = sqrt(B1_r^2 + B1_i^2 + Bz^2)
    #     Δt = s_Δt[s_idx]
    #     φ  = -π * γ * B * Δt
    #     sin_φ, cos_φ = sincos(φ)
    #     # rotation coefficients
    #     α_r =  cos_φ
    #     α_i = -(iszero(B) ? Bz/(B + eps(T)) : Bz/B) * sin_φ
    #     β_r =  (iszero(B) ?  B1_i/(B + eps(T)) : B1_i/B) * sin_φ
    #     β_i = -(iszero(B) ?  B1_r/(B + eps(T)) : B1_r/B) * sin_φ
    #     # apply rotation + relaxation
    #     Mxy_new_r = 2*(Mxy_i*(α_r*α_i - β_r*β_i) + Mz*(α_i*β_i + α_r*β_r)) + Mxy_r*(α_r^2 - α_i^2 - β_r^2 + β_i^2)
    #     Mxy_new_i = -2*(Mxy_r*(α_r*α_i + β_r*β_i) - Mz*(α_r*β_i - α_i*β_r)) + Mxy_i*(α_r^2 - α_i^2 + β_r^2 - β_i^2)
    #     Mz_new     =  Mz*(α_r^2 + α_i^2 - β_r^2 - β_i^2) - 2*(Mxy_r*(α_r*β_r - α_i*β_i) + Mxy_i*(α_r*β_i + α_i*β_r))
    #     ΔT1 = exp(-Δt/T1)
    #     ΔT2 = exp(-Δt/T2)
    #     Mxy_r = Mxy_new_r * ΔT2
    #     Mxy_i = Mxy_new_i * ΔT2
    #     Mz    = Mz_new    * ΔT1 + ρ*(1 - ΔT1)
    #   end
      M_xy[i] = Complex(Mxy_r, Mxy_i)
      M_z[i]  = Mz
    # end
end

function solve_steps!(M_xy, M_z, p_x, p_y, p_z, ΔBz, T1v, T2v, ρv, dt, B, M0, GPU, threads, blocks, Nsteps)
    ker = excitation!(GPU, threads)
    for _ in 1:Nsteps
        ker(M_xy, M_z, p_x, p_y, p_z,
            ΔBz, T1v, T2v, ρv,
            dt, B, M0; ndrange = blocks)
    end
    KernelAbstractions.synchronize(GPU)
    return M_z
end

function init_gpu_arrays(cpu_mxy, dt, tmax, params, GPU)
    Nsteps = ceil(Int, tmax / dt)

    threads = 256
    cpu_mz = zeros(Float32, length(cpu_mxy))
    blocks  = length(cpu_mz)

    M_xy = adapt(GPU, cpu_mxy)
    M_z  = adapt(GPU, cpu_mz)

    p_x = p_y = p_z = adapt(GPU, zeros(Float32, length(cpu_mz)))
    ΔBz = adapt(GPU, zeros(Float32, length(cpu_mz)))
    T1v = fill!(similar(ΔBz), params.T1)
    T2v = fill!(similar(ΔBz), params.T2)
    ρv  = fill!(similar(ΔBz), 1.0f0)

    return threads, blocks, Nsteps, M_xy, M_z, p_x, p_y, p_z, ΔBz, T1v, T2v, ρv
end

Random.seed!(123)
cpu_target_z = rand(Float32, 10)

const GPU_BACKEND = CUDA.CUDABackend()
const target_z = adapt(GPU_BACKEND, cpu_target_z)

const params = (γ = PHYS.γ, B = 1f-6, T1 = 1.0f0, T2 = 0.5f0, M0 = 1.0f0)
mxy_init = rand(Complex{Float32}, 10)
α  = 1f-2
iters = 100

dt, tmax = 0.001f0, 1.0f0
threads, blocks, Nsteps,
M_xy, M_z, p_x, p_y, p_z,
ΔBz, T1v, T2v, ρv = init_gpu_arrays(mxy_init, dt, tmax, params, GPU_BACKEND)
const other_args = (M_z, p_x, p_y, p_z, ΔBz, T1v, T2v, ρv, dt, threads, blocks, Nsteps)

function f(mxy0::AbstractVector{Complex{T}}) where {T}
    M_z, p_x, p_y, p_z, ΔBz, T1v, T2v, ρv, dt, threads, blocks, Nsteps = other_args
    final_Mz = solve_steps!(mxy0, M_z, p_x, p_y, p_z,
        ΔBz, T1v, T2v, ρv,
        dt, params.B, params.M0,
        GPU_BACKEND, threads, blocks, Nsteps )
    return Float32(sum(abs2, final_Mz .- target_z))
end
∂Mxy = similar(mxy_init)
for i in 1:iters
    autodiff(Reverse,
        f,
        Duplicated(mxy_init, ∂Mxy))
    Enzyme.autodiff_deferred(Reverse,
        Enzyme.Const(f), Const,
        Duplicated(mxy_init, ∂Mxy))
    # val, grad_mxy = value_and_gradient(f, AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse)), mxy_init)
    println("iter $i – ", ∂Mxy)
    break
    # mxy_init .-= α .* ∂Mxy
end

println("Optimized mxy_init: ", mxy_init)