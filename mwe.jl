############################
# GPU loss → RF gradient  ##
# (Enzyme seeded reverse) ##
############################

import CUDA
using KernelAbstractions
using Adapt
import Enzyme
import Random: seed!

CUDA.allowscalar(false)

# -------------------------------
# Problem size / constants
# -------------------------------
const GROUP_SIZE = 256
const Nspins     = 10
const Nt         = 100
const dt         = 1f-5
const γ          = Float32(2π) * 42.58f6  # keep as Float32

const N_Spins32 = Int32(Nspins)
const N_Δt32    = Int32(Nt)

# -------------------------------
# Backend
# -------------------------------
backend = CUDA.CUDABackend()
seed!(42)

# -------------------------------
# Allocate & initialize (host)
# -------------------------------
M_xy_h = zeros(Float32, 2Nspins)
M_z_h  = zeros(Float32, Nspins)

p_x_h   = zeros(Float32, Nspins)
p_y_h   = zeros(Float32, Nspins)
p_z_h   = zeros(Float32, Nspins)
p_ΔBz_h = zeros(Float32, Nspins)
p_T1_h  = fill(1.0f0, Nspins)
p_T2_h  = fill(0.5f0, Nspins)
p_ρ_h   = fill(1.0f0, Nspins)

s_Gx_h  = zeros(Float32, Nt)
s_Gy_h  = zeros(Float32, Nt)
s_Gz_h  = zeros(Float32, Nt)
s_Δt_h  = fill(dt, Nt)
s_Δf_h  = zeros(Float32, Nt)
s_B1r_h = fill(1f-6, Nt)
s_B1i_h = zeros(Float32, Nt)

target_c = rand(ComplexF32, Nspins)
target_h = vcat(Float32.(real.(target_c)), Float32.(imag.(target_c)))

# -------------------------------
# Move to device
# -------------------------------
M_xy   = adapt(backend, M_xy_h)
M_z    = adapt(backend, M_z_h)
p_x    = adapt(backend, p_x_h)
p_y    = adapt(backend, p_y_h)
p_z    = adapt(backend, p_z_h)
p_ΔBz  = adapt(backend, p_ΔBz_h)
p_T1   = adapt(backend, p_T1_h)
p_T2   = adapt(backend, p_T2_h)
p_ρ    = adapt(backend, p_ρ_h)
s_Gx   = adapt(backend, s_Gx_h)
s_Gy   = adapt(backend, s_Gy_h)
s_Gz   = adapt(backend, s_Gz_h)
s_Δt   = adapt(backend, s_Δt_h)
s_Δf   = adapt(backend, s_Δf_h)
s_B1r  = adapt(backend, s_B1r_h)
s_B1i  = adapt(backend, s_B1i_h)
target = adapt(backend, target_h)

# -------------------------------
# Physics kernel
# -------------------------------
@kernel unsafe_indices=true inbounds=true function excitation_kernel!(
    M_xy::AbstractVector{T}, M_z::AbstractVector{T},
    p_x::AbstractVector{T}, p_y::AbstractVector{T}, p_z::AbstractVector{T},
    p_ΔBz::AbstractVector{T}, p_T1::AbstractVector{T}, p_T2::AbstractVector{T}, p_ρ::AbstractVector{T},
    s_Gx::AbstractVector{T}, s_Gy::AbstractVector{T}, s_Gz::AbstractVector{T},
    s_Δt::AbstractVector{T}, s_Δf::AbstractVector{T}, s_B1r::AbstractVector{T}, s_B1i::AbstractVector{T},
    N_Spins::Int32, N_Δt::Int32
) where {T}

    i = Int(@index(Global, Linear))
    if i <= Int(N_Spins)
        N   = Int(N_Spins)
        ir  = i
        ii  = i + N

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
            gx  = s_Gx[s_idx]; gy = s_Gy[s_idx]; gz = s_Gz[s_idx]
            Δt  = s_Δt[s_idx]
            df  = s_Δf[s_idx]
            b1r = s_B1r[s_idx]; b1i = s_B1i[s_idx]

            Bz = (x*gx + y*gy + z*gz) + ΔBz - df / T(γ)
            B  = sqrt(b1r*b1r + b1i*b1i + Bz*Bz)

            ϕ  = T(-π) * T(γ) * B * Δt
            sϕ = sin(ϕ); cϕ = cos(ϕ)

            denom = B + T(1e-20)
            α_r =  cϕ
            α_i = -(Bz/denom) * sϕ
            β_r =  (b1i/denom) * sϕ
            β_i = -(b1r/denom) * sϕ

            Mxy_new_r = 2 * (Mxy_i * (α_r*α_i - β_r*β_i) + Mz * (α_i*β_i + α_r*β_r)) +
                        Mxy_r * (α_r*α_r - α_i*α_i - β_r*β_r + β_i*β_i)

            Mxy_new_i = -2 * (Mxy_r * (α_r*α_i + β_r*β_i) - Mz * (α_r*β_i - α_i*β_r)) +
                        Mxy_i * (α_r*α_r - α_i*α_i + β_r*β_r - β_i*β_i)

            Mz_new =    Mz * (α_r*α_r + α_i*α_i - β_r*β_r - β_i*β_i) -
                        2 * (Mxy_r * (α_r*β_r - α_i*β_i) + Mxy_i * (α_r*β_i + α_i*β_r))

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

# Kernel launcher (no synchronize here; handle syncs in callers)
function excitation_caller!(M_xy, M_z,
    p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
    s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1r, s_B1i,
    N_Spins::Int32, N_Δt::Int32, backend)

    k = excitation_kernel!(backend, GROUP_SIZE)
    k(M_xy, M_z,
      p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
      s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1r, s_B1i,
      N_Spins, N_Δt; ndrange=Int(N_Spins))
    return nothing
end

# -------------------------------
# GPU loss (outside AD)
# -------------------------------
function gpu_loss(M_xy::CUDA.CuArray{T}, target::CUDA.CuArray{T}) where {T<:AbstractFloat}
    # device reduction; returns a host scalar after tiny memcpy
    return CUDA.sum((M_xy .- target) .^ 2)
end

# -------------------------------
# Seed gradient of loss wrt M (GPU)
# -------------------------------
@kernel function loss_grad_M!(dM, M, TGT)
    i = @index(Global, Linear)
    if i <= length(M)
        @inbounds dM[i] = 2f0 * (M[i] - TGT[i])
    end
end

# -------------------------------
# Wrapper for physics only (used by Enzyme)
# -------------------------------
Base.@noinline function excite_only!(
    M_xy, M_z,
    p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
    s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1r, s_B1i,
    backend,
)
    excitation_caller!(M_xy, M_z,
        p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
        s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1r, s_B1i,
        N_Spins32, N_Δt32, backend)
    return nothing
end

# -------------------------------
# Main: forward → seed → reverse
# -------------------------------
function grad_rf!(
    ∇B1r::CUDA.CuArray{Float32}, ∇B1i::CUDA.CuArray{Float32},
    M_xy, M_z,
    p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
    s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1r, s_B1i,
    target,
    backend,
)
    # Forward (GPU)
    excite_only!(M_xy, M_z,
        p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
        s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1r, s_B1i,
        backend)
    KernelAbstractions.synchronize(backend)

    # Seed dL/dM (GPU)
    dM_xy = similar(M_xy)
    loss_grad_M!(backend, GROUP_SIZE)(dM_xy, M_xy, target; ndrange=length(M_xy))
    KernelAbstractions.synchronize(backend)

    # Zero seeds for other duplicated args
    dM_z = similar(M_z);  fill!(dM_z, 0f0)
    dΔt  = similar(s_Δt); fill!(dΔt,  0f0)
    dΔf  = similar(s_Δf); fill!(dΔf,  0f0)
    fill!(∇B1r, 0f0); fill!(∇B1i, 0f0)

    # Reverse through physics only, seeded with dM_xy
    Enzyme.autodiff(Enzyme.Reverse, excite_only!,
        Enzyme.Duplicated(M_xy, dM_xy),            # seed ∂L/∂M
        Enzyme.Duplicated(M_z,  dM_z),
        Enzyme.Const(p_x), Enzyme.Const(p_y), Enzyme.Const(p_z),
        Enzyme.Const(p_ΔBz), Enzyme.Const(p_T1), Enzyme.Const(p_T2), Enzyme.Const(p_ρ),
        Enzyme.Const(s_Gx), Enzyme.Const(s_Gy), Enzyme.Const(s_Gz),
        Enzyme.Duplicated(s_Δt, dΔt),
        Enzyme.Duplicated(s_Δf, dΔf),
        Enzyme.Duplicated(s_B1r, ∇B1r),            # outputs: ∇L/∂B1r
        Enzyme.Duplicated(s_B1i, ∇B1i),            # outputs: ∇L/∂B1i
        Enzyme.Const(backend),
    )
    KernelAbstractions.synchronize(backend)

    # Loss value (GPU reduce, post-AD)
    L = gpu_loss(M_xy, target)
    return L
end

# -------------------------------
# Run once to demonstrate
# -------------------------------
∇B1r = similar(s_B1r); ∇B1i = similar(s_B1i)
loss_val = grad_rf!(∇B1r, ∇B1i,
                    M_xy, M_z,
                    p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
                    s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1r, s_B1i,
                    target,
                    backend)

println("loss = ", loss_val)
# quick sanity peek (host copies only for logging)
println("‖∇B1r‖₂ = ", sqrt(sum(abs2, Array(∇B1r))))
println("‖∇B1i‖₂ = ", sqrt(sum(abs2, Array(∇B1i))))
