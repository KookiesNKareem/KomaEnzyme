import CUDA
using KernelAbstractions
using Adapt
using DifferentiationInterface
import Enzyme
using LinearAlgebra: norm
import Random: seed!

CUDA.allowscalar(false)
Enzyme.Compiler.VERBOSE_ERRORS[] = true

const GROUP_SIZE = 256
const Nspins     = 10
const Nt         = 100
const dt         = 1f-5
const γ          = 2π * 42.58f6  # Float32-friendly

backend = CUDA.CUDABackend()
seed!(42)

# ---------------- Host-side allocations ----------------
# M_xy is now a single real vector of length 2N: [real(1:N); imag(1:N)]
M_xy = zeros(Float32, 2Nspins)
M_z  = zeros(Float32, Nspins)

T1, T2, M0 = 1.0f0, 0.5f0, 1.0f0

obj = (;
    p_x   = zeros(Float32, Nspins),
    p_y   = zeros(Float32, Nspins),
    p_z   = zeros(Float32, Nspins),
    p_ΔBz = zeros(Float32, Nspins),
    p_T1  = fill(T1, Nspins),
    p_T2  = fill(T2, Nspins),
    p_ρ   = fill(M0, Nspins)
)

seq = (;
    s_Gx  = zeros(Float32, Nt),
    s_Gy  = zeros(Float32, Nt),
    s_Gz  = zeros(Float32, Nt),
    s_Δt  = fill(dt, Nt),
    s_Δf  = zeros(Float32, Nt),
    s_B1  = 1f-6 .* ones(ComplexF32, Nt)
)

# target as a single real vector [real; imag] to match M_xy layout
target_c = rand(ComplexF32, Nspins)
target   = vcat(Float32.(real.(target_c)), Float32.(imag.(target_c)))

# ---------------- Move to device ----------------
M_xy   = adapt(backend, M_xy)
M_z    = adapt(backend, M_z)
obj    = adapt(backend, obj)
seq    = adapt(backend, seq)
target = adapt(backend, target)

const N_Spins32 = Int32(length(obj.p_x))
const N_Δt32    = Int32(length(seq.s_Δt))

# ---------------- Kernel (typed; uses single 2N vector for M_xy) ----------------
@kernel unsafe_indices=true inbounds=true function excitation_kernel!(
    M_xy::AbstractVector{T}, M_z::AbstractVector{T},
    p_x::AbstractVector{T}, p_y::AbstractVector{T}, p_z::AbstractVector{T},
    p_ΔBz::AbstractVector{T}, p_T1::AbstractVector{T}, p_T2::AbstractVector{T}, p_ρ::AbstractVector{T},
    N_Spins::Int32,
    s_Gx::AbstractVector{T}, s_Gy::AbstractVector{T}, s_Gz::AbstractVector{T},
    s_Δt::AbstractVector{T}, s_Δf::AbstractVector{T}, s_B1::AbstractVector{Complex{T}},
    N_Δt::Int32
) where {T}

    i = Int(@index(Global, Linear))
    if i <= Int(N_Spins)
        N   = Int(N_Spins)
        ir  = i                 # real slot
        ii  = i + N             # imag slot

        x   = p_x[i];  y = p_y[i];  z = p_z[i]
        ΔBz = p_ΔBz[i]
        ρ   = p_ρ[i]
        T1  = p_T1[i]
        T2  = p_T2[i]

        Mxy_r = M_xy[ir]
        Mxy_i = M_xy[ii]
        Mz    = M_z[i]

        s_idx = 1
        @inbounds while s_idx <= Int(N_Δt)
            Bz   = (x*s_Gx[s_idx] + y*s_Gy[s_idx] + z*s_Gz[s_idx]) + ΔBz - s_Δf[s_idx] / T(γ)
            B1_r = real(s_B1[s_idx])
            B1_i = imag(s_B1[s_idx])
            B    = sqrt(B1_r*B1_r + B1_i*B1_i + Bz*Bz)
            Δt   = s_Δt[s_idx]

            φ = T(-π) * T(γ) * B * Δt
            sin_φ, cos_φ = sincos(φ)

            denom = abs(B) < T(1e-20) ? eps(T) : B
            α_r =  cos_φ
            α_i = -(Bz / denom) * sin_φ
            β_r =  (B1_i / denom) * sin_φ
            β_i = -(B1_r / denom) * sin_φ

            Mxy_new_r = 2 * (Mxy_i * (α_r * α_i - β_r * β_i) + Mz * (α_i * β_i + α_r * β_r)) +
                        Mxy_r * (α_r*α_r - α_i*α_i - β_r*β_r + β_i*β_i)

            Mxy_new_i = -2 * (Mxy_r * (α_r * α_i + β_r * β_i) - Mz * (α_r * β_i - α_i * β_r)) +
                        Mxy_i * (α_r*α_r - α_i*α_i + β_r*β_r - β_i*β_i)

            Mz_new =    Mz * (α_r*α_r + α_i*α_i - β_r*β_r - β_i*β_i) -
                        2 * (Mxy_r * (α_r * β_r - α_i * β_i) + Mxy_i * (α_r * β_i + α_i * β_r))

            ΔT1   = exp(-Δt / T1)
            ΔT2   = exp(-Δt / T2)
            Mxy_r = Mxy_new_r * ΔT2
            Mxy_i = Mxy_new_i * ΔT2
            Mz    = Mz_new * ΔT1 + ρ * (T(1) - ΔT1)

            s_idx += 1
        end

        M_xy[ir] = Mxy_r
        M_xy[ii] = Mxy_i
        M_z[i]   = Mz
    end
end

function excitation_caller!(M_xy, M_z,
    p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
    s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1,
    N_Spins::Int32, N_Δt::Int32,
    backend)

    ndrange = Int(N_Spins)  # one thread per spin
    k = excitation_kernel!(backend, GROUP_SIZE)
    k(M_xy, M_z,
      p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ, N_Spins,
      s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1, N_Δt;
      ndrange=ndrange)
    KernelAbstractions.synchronize(backend)
    return nothing
end

# Loss keeps M_xy as 2N real vector and differentiates through the kernel
function loss_fn(M_xy, M_z, obj, seq, target, backend)
    excitation_caller!(M_xy, M_z,
        obj.p_x, obj.p_y, obj.p_z, obj.p_ΔBz, obj.p_T1, obj.p_T2, obj.p_ρ,
        seq.s_Gx, seq.s_Gy, seq.s_Gz, seq.s_Δt, seq.s_Δf, seq.s_B1,
        N_Spins32, N_Δt32, backend)
    KernelAbstractions.synchronize(backend)
    return sum((M_xy .- target) .* (M_xy .- target))
end

ad_backend = AutoEnzyme(; mode = Enzyme.Reverse)

println("Running Enzyme AD through the GPU kernel (M_xy as 2N real vector)")
loss_val, ∇M_xy = value_and_gradient(
    loss_fn, ad_backend,
    M_xy, Constant(M_z), Constant(obj), Constant(seq), Constant(target), Constant(backend)
)
println("loss = ", loss_val)
