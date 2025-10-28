# 1D RF Slice Design with Enzyme Automatic Differentiation
#
# Features:
# - Real-only RF controls on design grid (Nrf)
# - Timeline from Koma discretization
# - Phase-matched loss function
# - GPU/CPU backend support via KernelAbstractions
using KomaMRICore, Suppressor
using Random: seed!
using KernelAbstractions
using KernelAbstractions: @kernel, @index, @atomic
using Adapt
using CUDA
import Enzyme
using Statistics: mean, std
using LinearAlgebra: norm
using Plots
using BenchmarkTools

# -------------- Constants / Params --------------
seed!(42)

const B1   = 4.9e-6
const Trf  = 3.2e-3
const TBP  = 8.0
const Δz   = 6e-3
const zmax = 8e-3
const fmax = TBP / Trf
const γf64 = 2π * 42.57747892e6       # rad/(s·T)
const γ    = Float32(γf64)

const Nspins = 100_000
const GROUP_SIZE = 256

z  = collect(range(-zmax, zmax, length=Nspins))
Gz = fmax / (Float64(γf64) * Δz)

# -------------- Koma objects --------------
sys = Scanner()
seq_full = PulseDesigner.RF_sinc(B1, Trf, sys; G=[Gz; 0; 0], TBP=TBP)
seq = seq_full
obj = Phantom(; x=z)

# Controls on original RF grid
const Nrf = length(seq.RF[1].A)

# -------------- Simulation config --------------
sim_params = KomaMRICore.default_sim_params()
sim_params["Δt_rf"]       = Trf / (2*(Nrf - 1))   # Koma internal ×2 upsample
sim_params["Δt"]          = Inf                   # RF-only window per your constraint
sim_params["return_type"] = "state"
sim_params["precision"]   = "f32"
sim_params["Nthreads"]    = 1
sim_params["sim_method"]  = KomaMRICore.Bloch()

# Baseline (for demod/compare)
mag_sinc = @suppress simulate(obj, seq, sys; sim_params)

# -------------- Target profile --------------
butterworth_degree = 5
target_profile = 0.5im ./ (1 .+ (z ./ (Δz / 2)).^(2*butterworth_degree))
const TARGET_R_h = Float32.(real.(target_profile))
const TARGET_I_h = Float32.(imag.(target_profile))
const TARGET_MAG = Float32.(abs.(target_profile))

# -------------- Backend --------------
CUDA.device!(1)
const backend = CUDA.CUDABackend()
CUDA.limit!(CUDA.CU_LIMIT_MALLOC_HEAP_SIZE, 20*1024^3)
# const backend = KernelAbstractions.CPU()

# Warmup: compile kernels & AD once
function _gpu_warmup!()
    x_w = zeros(Float32, Nrf)
    ∇x_w = similar(x_w)
    try
        loss_and_grad!(∇x_w, x_w)
    catch
        # ignore any first-call hiccups
    end
    CUDA.synchronize()
    return nothing
end

# =============================================================
# Discretize Koma timeline (RF-only window)
# =============================================================
function get_koma_timeline(seq, sim_params)
    seqd = KomaMRICore.discretize(seq; sampling_params=sim_params)
    Nt = length(seqd.Δt)
    B1i = ComplexF32.(seqd.B1[1:Nt])
    Gxi = Float32.(seqd.Gx[1:Nt])
    Gyi = Float32.(seqd.Gy[1:Nt])
    Gzi = Float32.(seqd.Gz[1:Nt])
    Δti = Float32.(seqd.Δt[1:Nt])
    Δfi = Float32.(seqd.Δf[1:Nt])       # angular freq (rad/s), per Koma
    ti  = Float32.(seqd.t[1:Nt])
    rf_active_idx = findall(x -> abs(x) > 1e-10, B1i)
    return (B1=B1i, Gx=Gxi, Gy=Gyi, Gz=Gzi, Δt=Δti, Δf=Δfi, t=ti,
            rf_active_idx=rf_active_idx, Nt=Int32(Nt), Nrf_original=Nrf)
end

const TL    = get_koma_timeline(seq, sim_params)
const rf_idx = TL.rf_active_idx
const Lrf_taps = length(rf_idx)

# Residual kx moment (for demod)
const KX_RESID = sum(TL.Gx .* TL.Δt)           # T·s/m

# =============================================================
# Device buffers
# =============================================================
const N         = length(z)
const N_Spins32 = Int32(N)
const N_Δt32    = TL.Nt

adapt_dev(x) = CUDA.adapt(CuArray, x)

# State
M_xy  = adapt_dev(zeros(Float32, 2N))   # [1:N]=Re, [N+1:2N]=Im
M_z   = adapt_dev(ones(Float32, N))

# Spins/params
p_x   = adapt_dev(Float32.(z))
p_y   = adapt_dev(zeros(Float32, N))
p_z   = adapt_dev(zeros(Float32, N))
p_ΔBz = adapt_dev(zeros(Float32, N))
p_T1  = adapt_dev(fill(Float32(1e9), N))
p_T2  = adapt_dev(fill(Float32(1e9), N))
p_ρ   = adapt_dev(ones(Float32, N))

# Timeline
s_Gx  = adapt_dev(TL.Gx)
s_Gy  = adapt_dev(TL.Gy)
s_Gz  = adapt_dev(TL.Gz)
s_Δt  = adapt_dev(TL.Δt)
s_Δf  = adapt_dev(TL.Δf)

# RF (we overwrite only rf_idx; Im stays 0)
s_B1r = adapt_dev(real.(TL.B1))
s_B1i = adapt_dev(imag.(TL.B1))

# Targets (device copies for seeding)
target_r_dev = adapt_dev(TARGET_R_h)
target_i_dev = adapt_dev(TARGET_I_h)

# Host RF scratch (full timeline)
B1r_host = copy(real.(TL.B1))
B1i_host = copy(imag.(TL.B1))

# Loss accumulation buffer on device
loss_buf = adapt_dev(zeros(Float32, 1))
# Adjoint buffer for loss (seed with 1)
dloss_buf = similar(loss_buf)
# Per-spin error buffer and a temp buffer for atomics-free reduction
loss_err = adapt_dev(zeros(Float32, N))
loss_tmp = adapt_dev(zeros(Float32, cld(N, 2)))

# =============================================================
# Kernels (Bloch)
# =============================================================
@kernel inbounds=true function excitation_kernel!(
    M_xy::AbstractVector{T}, M_z::AbstractVector{T},
    p_x::AbstractVector{T}, p_y::AbstractVector{T}, p_z::AbstractVector{T},
    p_ΔBz::AbstractVector{T}, p_T1::AbstractVector{T}, p_T2::AbstractVector{T}, p_ρ::AbstractVector{T},
    s_Gx::AbstractVector{T}, s_Gy::AbstractVector{T}, s_Gz::AbstractVector{T},
    s_Δt::AbstractVector{T}, s_Δf::AbstractVector{T}, s_B1r::AbstractVector{T}, s_B1i::AbstractVector{T},
    N_Spins::Int32, N_Δt::Int32
) where {T}
    eps_d = T(1f-20)
    i = Int(Tuple(@index(Global))[1])
    if i <= Int(N_Spins)
        N   = Int(N_Spins)
        ir  = i
        ii  = i + N

        x   = p_x[i]; y = p_y[i]; z = p_z[i]
        ΔBz = p_ΔBz[i]
        ρ   = p_ρ[i]; T1 = p_T1[i]; T2 = p_T2[i]

        Mx  = M_xy[ir];  My = M_xy[ii];  Mz = M_z[i]

        s_idx = 1
        @inbounds while s_idx <= Int(N_Δt)
            gx  = s_Gx[s_idx]; gy = s_Gy[s_idx]; gz = s_Gz[s_idx]
            Δt  = s_Δt[s_idx]; df = s_Δf[s_idx]
            b1r = s_B1r[s_idx]; b1i = s_B1i[s_idx]

            Bz = (x*gx + y*gy + z*gz) + ΔBz - df / T(γ)

            B  = sqrt(b1r*b1r + b1i*b1i + Bz*Bz)
            φ  = T(-π) * T(γ) * B * Δt
            sφ = sin(φ); cφ = cos(φ)
            denom = B + eps_d

            α_r =  cφ
            α_i = -(Bz/denom) * sφ
            β_r =  (b1i/denom) * sφ
            β_i = -(b1r/denom) * sφ

            Mx_new = 2 * (My * (α_r*α_i - β_r*β_i) + Mz * (α_i*β_i + α_r*β_r)) +
                     Mx * (α_r*α_r - α_i*α_i - β_r*β_r + β_i*β_i)

            My_new = -2 * (Mx * (α_r*α_i + β_r*β_i) - Mz * (α_r*β_i - α_i*β_r)) +
                     My * (α_r*α_r - α_i*α_i + β_r*β_r - β_i*β_i)

            Mz_new =    Mz * (α_r*α_r + α_i*α_i - β_r*β_r - β_i*β_i) -
                        2 * (Mx * (α_r*β_r - α_i*β_i) + My * (α_r*β_i + α_i*β_r))

            ΔT1 = exp(-Δt / T1)
            ΔT2 = exp(-Δt / T2)

            Mx = Mx_new * ΔT2
            My = My_new * ΔT2
            Mz = Mz_new * ΔT1 + ρ * (T(1f0) - ΔT1)

            s_idx += 1
        end

        M_xy[ir] = Mx
        M_xy[ii] = My
        M_z[i]   = Mz
    end
end

# Fused excitation + per-spin loss kernel (single launch, AD-friendly)
@kernel inbounds=true function excite_and_loss_per_spin!(
    err::AbstractVector{Float32},
    p_x::AbstractVector{Float32}, p_y::AbstractVector{Float32}, p_z::AbstractVector{Float32},
    p_ΔBz::AbstractVector{Float32}, p_T1::AbstractVector{Float32}, p_T2::AbstractVector{Float32}, p_ρ::AbstractVector{Float32},
    s_Gx::AbstractVector{Float32}, s_Gy::AbstractVector{Float32}, s_Gz::AbstractVector{Float32},
    s_Δt::AbstractVector{Float32}, s_Δf::AbstractVector{Float32},
    s_B1r::AbstractVector{Float32}, s_B1i::AbstractVector{Float32},
    cosφ::AbstractVector{Float32},  sinφ::AbstractVector{Float32},
    target_r::AbstractVector{Float32}, target_i::AbstractVector{Float32},
    N_Spins::Int32, N_Δt::Int32
)
    i = Int(Tuple(@index(Global))[1])
    if i <= Int(N_Spins)
        T = Float32
        eps_d = T(1f-20)
        N = Int(N_Spins)
        x   = p_x[i]; y = p_y[i]; z = p_z[i]
        ΔBz = p_ΔBz[i]
        ρ   = p_ρ[i]; T1 = p_T1[i]; T2 = p_T2[i]
        Mx  = T(0);  My = T(0);  Mz = T(1)
        s_idx = 1
        @inbounds while s_idx <= Int(N_Δt)
            gx  = s_Gx[s_idx]; gy = s_Gy[s_idx]; gz = s_Gz[s_idx]
            Δt  = s_Δt[s_idx]; df = s_Δf[s_idx]
            b1r = s_B1r[s_idx]; b1i = s_B1i[s_idx]
            Bz = (x*gx + y*gy + z*gz) + ΔBz - df / T(γ)
            B  = sqrt(b1r*b1r + b1i*b1i + Bz*Bz)
            φ  = T(-π) * T(γ) * B * Δt
            sφ = sin(φ); cφ = cos(φ)
            denom = B + eps_d
            α_r =  cφ
            α_i = -(Bz/denom) * sφ
            β_r =  (b1i/denom) * sφ
            β_i = -(b1r/denom) * sφ
            Mx_new = 2 * (My * (α_r*α_i - β_r*β_i) + Mz * (α_i*β_i + α_r*β_r)) +
                     Mx * (α_r*α_r - α_i*α_i - β_r*β_r + β_i*β_i)
            My_new = -2 * (Mx * (α_r*α_i + β_r*β_i) - Mz * (α_r*β_i - α_i*β_r)) +
                     My * (α_r*α_r - α_i*α_i + β_r*β_r - β_i*β_i)
            Mz_new =    Mz * (α_r*α_r + α_i*α_i - β_r*β_r - β_i*β_i) -
                        2 * (Mx * (α_r*β_r - α_i*β_i) + My * (α_r*β_i + α_i*β_r))
            ΔT1 = exp(-Δt / T1)
            ΔT2 = exp(-Δt / T2)
            Mx = Mx_new * ΔT2
            My = My_new * ΔT2
            Mz = Mz_new * ΔT1 + ρ * (T(1f0) - ΔT1)
            s_idx += 1
        end
        c = cosφ[i]; s = sinφ[i]
        Tr = target_r[i]; Ti = target_i[i]
        Mrot_r = Mx*c + My*s
        Mrot_i = My*c - Mx*s
        dr = Mrot_r - Tr
        di = Mrot_i - Ti
        err[i] = 0.5f0 * (dr*dr + di*di) / T(N)
    end
end

# Launcher
function launch_excitation!(
    M_xy, M_z,
    p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
    s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1r, s_B1i,
    N_Spins::Int32, N_Δt::Int32, backend)
    k = excitation_kernel!(backend)
    k(M_xy, M_z,
      p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
      s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1r, s_B1i,
      N_Spins, N_Δt; ndrange=Int(N_Spins), workgroupsize=GROUP_SIZE)
    return nothing
end

Base.@noinline function excite_only!(
    M_xy, M_z,
    p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
    s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1r, s_B1i,
    backend,
)
    launch_excitation!(
        M_xy, M_z,
        p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
        s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1r, s_B1i,
        N_Spins32, N_Δt32, backend)
    return nothing
end

Base.@noinline function end2end_loss_fused_per_spin!(
    err_buf::AbstractVector{Float32},
    p_x::AbstractVector{Float32}, p_y::AbstractVector{Float32}, p_z::AbstractVector{Float32},
    p_ΔBz::AbstractVector{Float32}, p_T1::AbstractVector{Float32}, p_T2::AbstractVector{Float32}, p_ρ::AbstractVector{Float32},
    s_Gx::AbstractVector{Float32}, s_Gy::AbstractVector{Float32}, s_Gz::AbstractVector{Float32},
    s_Δt::AbstractVector{Float32}, s_Δf::AbstractVector{Float32},
    s_B1r::AbstractVector{Float32}, s_B1i::AbstractVector{Float32},
    target_r_dev::AbstractVector{Float32}, target_i_dev::AbstractVector{Float32},
    cosφ_dev::AbstractVector{Float32},  sinφ_dev::AbstractVector{Float32},
    backend)
    excite_and_loss_per_spin!(backend, GROUP_SIZE)(
        err_buf,
        p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
        s_Gx, s_Gy, s_Gz, s_Δt, s_Δf,
        s_B1r, s_B1i,
        cosφ_dev, sinφ_dev,
        target_r_dev, target_i_dev,
        N_Spins32, N_Δt32;
        ndrange=Int(N))
    return nothing
end

# =============================================================
# Phase-matched loss pieces
# =============================================================
@kernel function phase_matched_seed_per_spin!(
    dM::AbstractVector{Float32}, M::AbstractVector{Float32},
    target_r::AbstractVector{Float32}, target_i::AbstractVector{Float32},
    cosφ::AbstractVector{Float32},  sinφ::AbstractVector{Float32}, λ::Float32
)
    i = Int(Tuple(@index(Global))[1])
    N = length(target_r)
    if i <= N
        ir = i; ii = i + N
        Mr = M[ir]; Mi = M[ii]
        Tr = target_r[i]; Ti = target_i[i]
        c = cosφ[i]; s = sinφ[i]
        Mrot_r = Mr*c + Mi*s
        Mrot_i = Mi*c - Mr*s
        scale = λ / Float32(N)
        g_rot_r = scale * (Mrot_r - Tr)
        g_rot_i = scale * (Mrot_i - Ti)
        dM[ir] =  g_rot_r * c - g_rot_i * s
        dM[ii] =  g_rot_r * s + g_rot_i * c
    end
end

# Device loss reduction kernel (phase-matched), returns scalar via atomic add
@kernel function phase_matched_loss_reduce!(
    acc::AbstractVector{Float32}, M::AbstractVector{Float32},
    target_r::AbstractVector{Float32}, target_i::AbstractVector{Float32},
    cosφ::AbstractVector{Float32},  sinφ::AbstractVector{Float32}, N_Spins::Int32
)
    i = Int(Tuple(@index(Global))[1])
    N = Int(N_Spins)
    if i <= N
        ir = i; ii = i + N
        Mr = M[ir]; Mi = M[ii]
        Tr = target_r[i]; Ti = target_i[i]
        c = cosφ[i]; s = sinφ[i]
        Mrot_r = Mr*c + Mi*s
        Mrot_i = Mi*c - Mr*s
        dr = Mrot_r - Tr
        di = Mrot_i - Ti
        e = 0.5f0 * (dr*dr + di*di) / Float32(N)
        @atomic acc[1] += e
    end
end

# Compute per-spin phase-matched squared error (no atomics)
@kernel function phase_matched_loss_per_spin!(
    err::AbstractVector{Float32}, M::AbstractVector{Float32},
    target_r::AbstractVector{Float32}, target_i::AbstractVector{Float32},
    cosφ::AbstractVector{Float32},  sinφ::AbstractVector{Float32}, N_Spins::Int32
)
    i = Int(Tuple(@index(Global))[1])
    N = Int(N_Spins)
    if i <= N
        ir = i; ii = i + N
        Mr = M[ir]; Mi = M[ii]
        Tr = target_r[i]; Ti = target_i[i]
        c = cosφ[i]; s = sinφ[i]
        Mrot_r = Mr*c + Mi*s
        Mrot_i = Mi*c - Mr*s
        dr = Mrot_r - Tr
        di = Mrot_i - Ti
        err[i] = 0.5f0 * (dr*dr + di*di) / Float32(N)
    end
end

# Pairwise reduction without atomics: out[j] = in[2j-1] + in[2j]
@kernel function reduce_pairwise!(
    out::AbstractVector{Float32}, in::AbstractVector{Float32}, in_len::Int32
)
    j = Int(Tuple(@index(Global))[1])
    len = Int(in_len)
    outlen = (len + 1) >>> 1
    if j <= outlen
        i1 = 2*j - 1
        i2 = 2*j
        s = in[i1]
        if i2 <= len
            s += in[i2]
        end
        out[j] = s
    end
end

# Copy first element of vec into acc[1] on device (no host scalar indexing)
@kernel function write_first!(acc::AbstractVector{Float32}, vec::AbstractVector{Float32})
    i = Int(Tuple(@index(Global))[1])
    if i == 1
        acc[1] = vec[1]
    end
end

# Global phase φ0 and spatial demod κ·x
function compute_phase_arrays(Mr::Vector{Float32}, Mi::Vector{Float32})
    N  = length(Mr)
    κ  = Float32(γ) * Float32(KX_RESID)     # [rad/m]
    xh = Float32.(z)
    acc_r = 0.0f0; acc_i = 0.0f0
    @inbounds for i in 1:N
        θ = κ * xh[i]; c = cos(θ); s = sin(θ)
        Mr_dem =  Mr[i]*c + Mi[i]*s
        Mi_dem =  Mi[i]*c - Mr[i]*s
        Tr = TARGET_R_h[i]; Ti = TARGET_I_h[i]
        acc_r +=  Tr*Mr_dem + Ti*Mi_dem
        acc_i +=  Tr*Mi_dem - Ti*Mr_dem
    end
    φ0 = atan(acc_i, acc_r)
    cosφ = similar(Mr);  sinφ = similar(Mr)
    c0 = cos(φ0); s0 = sin(φ0)
    @inbounds for i in 1:N
        θ = κ * xh[i]; cθ = cos(θ); sθ = sin(θ)
        cosφ[i] = c0*cθ - s0*sθ
        sinφ[i] = s0*cθ + c0*sθ
    end
    return cosφ, sinφ
end

# =============================================================
# Control <-> timeline mapping (linear)
# =============================================================
const n_ctrl = Nrf
const n_taps = Lrf_taps
const ctrl_pos = collect(range(0f0, 1f0, length=n_ctrl))
const tap_pos  = collect(range(0f0, 1f0, length=n_taps))

function map_x_to_timeline!(B1r::Vector{Float32}, B1i::Vector{Float32}, x::Vector{Float32})
    @assert length(x) == n_ctrl
    B1r .= real.(TL.B1);  fill!(B1i, 0f0)
    j = 1
    @inbounds for (k, idx) in enumerate(rf_idx)
        t = tap_pos[k]
        while j < n_ctrl && ctrl_pos[j+1] < t
            j += 1
        end
        if j == n_ctrl
            B1r[idx] = x[end]
        else
            α = (t - ctrl_pos[j]) / (ctrl_pos[j+1] - ctrl_pos[j] + 1f-20)
            B1r[idx] = (1f0-α)*x[j] + α*x[j+1]
        end
    end
    return nothing
end

# Exact adjoint of the linear interp used in map_x_to_timeline!,
# under the ∑ Δt · (·) inner product on the timeline.
function scatter_grad_to_x!(∇x::Vector{Float32}, dB1r_timeline::AbstractVector{Float32})
    @assert length(dB1r_timeline) == length(TL.Δt)
    resize!(∇x, n_ctrl);  fill!(∇x, 0f0)
    wsum = zeros(Float32, n_ctrl)

    j = 1
    @inbounds for (k, idx) in enumerate(rf_idx)
        t   = tap_pos[k]
        Δtw = TL.Δt[idx]                # <-- key: time-step weight
        while j < n_ctrl && ctrl_pos[j+1] < t
            j += 1
        end
        if j == n_ctrl
            ∇x[end] += Δtw * dB1r_timeline[idx]
            wsum[end] += Δtw
        else
            α  = (t - ctrl_pos[j]) / (ctrl_pos[j+1] - ctrl_pos[j] + 1f-20)
            w0 = (1f0 - α) * Δtw
            w1 = α * Δtw
            ∇x[j]   += w0 * dB1r_timeline[idx];  wsum[j]   += w0
            ∇x[j+1] += w1 * dB1r_timeline[idx];  wsum[j+1] += w1
        end
    end

    @inbounds for i in 1:n_ctrl
        if wsum[i] > 0f0
            ∇x[i] /= wsum[i]
        end
    end
    return ∇x
end

# =============================================================
# Forward, Loss, Grad
# =============================================================
function forward!(x::Vector{Float32})
    map_x_to_timeline!(B1r_host, B1i_host, x)
    copyto!(s_B1r, B1r_host); copyto!(s_B1i, B1i_host)
    fill!(M_xy, 0f0); fill!(M_z, 1f0)
    excite_only!(M_xy, M_z,
        p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
        s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1r, s_B1i,
        backend)
    KernelAbstractions.synchronize(backend)
    Mr_h = Array(view(M_xy, 1:N))
    Mi_h = Array(view(M_xy, N+1:2N))
    cosφ, sinφ = compute_phase_arrays(Mr_h, Mi_h)
    loss_sum = 0.0f0
    @inbounds for i in 1:N
        c = cosφ[i]; s = sinφ[i]
        Mrot_r = Mr_h[i]*c + Mi_h[i]*s
        Mrot_i = Mi_h[i]*c - Mr_h[i]*s
        dr = Mrot_r - TARGET_R_h[i]
        di = Mrot_i - TARGET_I_h[i]
        loss_sum += dr*dr + di*di
    end
    L = 0.5f0 * loss_sum / Float32(N)
    return L, (cosφ=cosφ, sinφ=sinφ)
end

# Device adjoint scratch
dM_xy = similar(M_xy)
dM_z  = similar(M_z)
∇B1r  = similar(s_B1r)
∇B1i  = similar(s_B1i)

# Device loss function used for Enzyme to backprop dL/dM automatically
Base.@noinline function loss_from_state(
    M_xy_dev::AbstractVector{Float32}, M_z_dev::AbstractVector{Float32},
    target_r_dev::AbstractVector{Float32}, target_i_dev::AbstractVector{Float32},
    cosφ_dev::AbstractVector{Float32},  sinφ::AbstractVector{Float32},
    backend)
    fill!(loss_buf, 0f0)
    phase_matched_loss_reduce!(backend, GROUP_SIZE)(
        loss_buf, M_xy_dev, target_r_dev, target_i_dev, cosφ_dev, sinφ_dev, N_Spins32; ndrange=Int(N))
    KernelAbstractions.synchronize(backend)
    return Array(loss_buf)[1]
end

Base.@noinline function device_loss_into_buffer!(
    acc::AbstractVector{Float32},
    M_xy_dev::AbstractVector{Float32},
    target_r_dev::AbstractVector{Float32}, target_i_dev::AbstractVector{Float32},
    cosφ_dev::AbstractVector{Float32},  sinφ::AbstractVector{Float32},
    err_buf::AbstractVector{Float32}, tmp_buf::AbstractVector{Float32},
    backend)
    fill!(acc, 0f0)
    # 1) Per-spin errors (no atomics)
    phase_matched_loss_per_spin!(backend, GROUP_SIZE)(
        err_buf, M_xy_dev, target_r_dev, target_i_dev, cosφ_dev, sinφ, N_Spins32; ndrange=Int(N))
    # 2) Multi-pass pairwise reduction to 1 value (no atomics)
    cur = err_buf; cur_len = N_Spins32
    tmp = tmp_buf
    while Int(cur_len) > 1
        out_len = Int32((Int(cur_len) + 1) >>> 1)
        reduce_pairwise!(backend, GROUP_SIZE)(tmp, cur, cur_len; ndrange=Int(out_len))
        KernelAbstractions.synchronize(backend)
        cur, tmp = tmp, cur
        cur_len = out_len
    end
    # 3) Write the single result to acc[1] on device
    write_first!(backend, GROUP_SIZE)(acc, cur; ndrange=1)
    KernelAbstractions.synchronize(backend)
    return nothing
end

# End-to-end loss that runs excitation then loss, for single-call AD from RF timeline
Base.@noinline function end2end_loss_into_buffer!(
    acc::AbstractVector{Float32},
    M_xy_dev::AbstractVector{Float32}, M_z_dev::AbstractVector{Float32},
    p_x::AbstractVector{Float32}, p_y::AbstractVector{Float32}, p_z::AbstractVector{Float32},
    p_ΔBz::AbstractVector{Float32}, p_T1::AbstractVector{Float32}, p_T2::AbstractVector{Float32}, p_ρ::AbstractVector{Float32},
    s_Gx::AbstractVector{Float32}, s_Gy::AbstractVector{Float32}, s_Gz::AbstractVector{Float32},
    s_Δt::AbstractVector{Float32}, s_Δf::AbstractVector{Float32},
    s_B1r::AbstractVector{Float32}, s_B1i::AbstractVector{Float32},
    target_r_dev::AbstractVector{Float32}, target_i_dev::AbstractVector{Float32},
    cosφ_dev::AbstractVector{Float32},  sinφ_dev::AbstractVector{Float32},
    err_buf::AbstractVector{Float32}, tmp_buf::AbstractVector{Float32},
    backend)
    fill!(M_xy_dev, 0f0); fill!(M_z_dev, 1f0)
    excite_only!(M_xy_dev, M_z_dev,
        p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
        s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1r, s_B1i,
        backend)
    device_loss_into_buffer!(acc, M_xy_dev, target_r_dev, target_i_dev, cosφ_dev, sinφ_dev, err_buf, tmp_buf, backend)
    return nothing
end

# End-to-end loss that writes per-spin errors (no device reduction)
Base.@noinline function end2end_loss_per_spin!(
    err_buf::AbstractVector{Float32},
    M_xy_dev::AbstractVector{Float32}, M_z_dev::AbstractVector{Float32},
    p_x::AbstractVector{Float32}, p_y::AbstractVector{Float32}, p_z::AbstractVector{Float32},
    p_ΔBz::AbstractVector{Float32}, p_T1::AbstractVector{Float32}, p_T2::AbstractVector{Float32}, p_ρ::AbstractVector{Float32},
    s_Gx::AbstractVector{Float32}, s_Gy::AbstractVector{Float32}, s_Gz::AbstractVector{Float32},
    s_Δt::AbstractVector{Float32}, s_Δf::AbstractVector{Float32},
    s_B1r::AbstractVector{Float32}, s_B1i::AbstractVector{Float32},
    target_r_dev::AbstractVector{Float32}, target_i_dev::AbstractVector{Float32},
    cosφ_dev::AbstractVector{Float32},  sinφ_dev::AbstractVector{Float32},
    backend)
    fill!(M_xy_dev, 0f0); fill!(M_z_dev, 1f0)
    excite_only!(M_xy_dev, M_z_dev,
        p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
        s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1r, s_B1i,
        backend)
    phase_matched_loss_per_spin!(backend, GROUP_SIZE)(
        err_buf, M_xy_dev, target_r_dev, target_i_dev, cosφ_dev, sinφ_dev, N_Spins32; ndrange=Int(N))
    KernelAbstractions.synchronize(backend)
    return nothing
end

function loss_and_grad!(∇x::Vector{Float32}, x::Vector{Float32})
    # Forward once to get current phase arrays (treated constant for AD)
    _, cφ = forward!(x)
    cosφ_dev = adapt_dev(cφ.cosφ)
    sinφ_dev = adapt_dev(cφ.sinφ)

    # AD from per-spin errors -> RF timeline using fused kernel
    fill!(loss_err, 0f0)
    dloss_err = similar(loss_err); fill!(dloss_err, 1.0f0)
    fill!(∇B1r, 0f0); fill!(∇B1i, 0f0)

    Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.ReverseWithPrimal),
        end2end_loss_fused_per_spin!,
        Enzyme.Duplicated(loss_err, dloss_err),
        Enzyme.Const(p_x), Enzyme.Const(p_y), Enzyme.Const(p_z),
        Enzyme.Const(p_ΔBz), Enzyme.Const(p_T1), Enzyme.Const(p_T2), Enzyme.Const(p_ρ),
        Enzyme.Const(s_Gx), Enzyme.Const(s_Gy), Enzyme.Const(s_Gz),
        Enzyme.Const(s_Δt), Enzyme.Const(s_Δf),
        Enzyme.Duplicated(s_B1r, ∇B1r), Enzyme.Const(s_B1i),
        Enzyme.Const(target_r_dev), Enzyme.Const(target_i_dev),
        Enzyme.Const(cosφ_dev), Enzyme.Const(sinφ_dev),
        Enzyme.Const(backend),
    )
    KernelAbstractions.synchronize(backend)
    L = sum(Array(loss_err))

    dB1r_host = Array(∇B1r)
    scatter_grad_to_x!(∇x, dB1r_host)
    return L
end

# =============================================================
# Sanity check vs Koma (forward-only; demod for compare)
# =============================================================
let
    x0_ctrl = Float32.(real.(seq.RF[1].A))
    map_x_to_timeline!(B1r_host, B1i_host, x0_ctrl)
    copyto!(s_B1r, B1r_host); copyto!(s_B1i, B1i_host)
    fill!(M_xy, 0f0); fill!(M_z, 1f0)
    launch_excitation!(
        M_xy, M_z,
        p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
        s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1r, s_B1i,
        N_Spins32, N_Δt32, backend)
    KernelAbstractions.synchronize(backend)
    Mr = Array(view(M_xy, 1:N)); Mi = Array(view(M_xy, N+1:2N))
    cosφ, sinφ = compute_phase_arrays(Mr, Mi)
    Mrot_mag = sqrt.((Mr .* cosφ .+ Mi .* sinφ).^2 .+ (Mi .* cosφ .- Mr .* sinφ).^2)
    diff_mag = Mrot_mag .- Float32.(abs.(mag_sinc.xy))
    @show sqrt(mean(diff_mag.^2))
end

function plot_rf_and_profile(x)
    map_x_to_timeline!(B1r_host, B1i_host, x)
    copyto!(s_B1r, B1r_host); fill!(s_B1i, 0f0)
    fill!(M_xy, 0f0); fill!(M_z, 1f0)
    launch_excitation!(
        M_xy, M_z,
        p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
        s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1r, s_B1i,
        N_Spins32, N_Δt32, backend)
    KernelAbstractions.synchronize(backend)

    Mr = Array(view(M_xy, 1:N)); Mi = Array(view(M_xy, N+1:2N))
    cosφ, sinφ = compute_phase_arrays(Mr, Mi)
    Mrot_r = Mr .* cosφ .+ Mi .* sinφ
    Mrot_i = Mi .* cosφ .- Mr .* sinφ
    mag = sqrt.(Mrot_r .* Mrot_r .+ Mrot_i .* Mrot_i)

    p_r = plot(title="RF real (controls)", ylabel="B1_r (T)"); plot!(p_r, 1:length(x), x; label="Re(B1) controls", lw=2)
    t_edges  = cumsum(vcat(0f0, TL.Δt[1:end-1])); t_center = t_edges .+ 0.5f0 .* TL.Δt
    p_i = plot(title="RF imag (timeline)", ylabel="B1_i (T)"); plot!(p_i, t_center[rf_idx], B1i_host[rf_idx]; label="Im(B1) taps", lw=2)
    p_prof = plot(title="|M_xy|(z)", xlabel="z (m)", ylabel="magnitude")
    plot!(p_prof, z, TARGET_MAG; label="Target |M|", lw=3, ls=:dash)
    plot!(p_prof, z, mag;        label="Achieved", lw=2)
    plot(p_r, p_i, p_prof; layout=(3,1), size=(900,900))
end


# =============================================================
# Malitsky-Mishchenko Adaptive Gradient Descent
# =============================================================

"""
Malitsky-Mishchenko adaptive gradient descent optimizer.

Parameters:
    λ_i = min{√(1 + θ_{i-1}) * λ_{i-1}, ||x^i - x^{i-1}|| / (2||∇f(x^i) - ∇f(x^{i-1})||)}
    x^{i+1} = x^i - λ_i * ∇f(x^i)
    θ_i = λ_i / λ_{i-1}

Reference: Malitsky & Mishchenko (2020), "Adaptive gradient descent without descent"
"""
function train_loop_mm!(x::Vector{Float32}, nsteps::Int, rf_clip::Float32; λ_init::Float32=1f-9)
    # Preallocate buffers
    ∇x_curr = zeros(Float32, length(x))
    ∇x_prev = zeros(Float32, length(x))
    x_prev = copy(x)

    λ_prev = λ_init
    θ_prev = 1.0f0

    @inbounds for i in 1:nsteps
        # Compute current gradient
        _ = loss_and_grad!(∇x_curr, x)
        CUDA.synchronize()

        if i == 1
            # First iteration: simple gradient step
            λ = λ_init
            @. x = clamp(x - λ * ∇x_curr, -rf_clip, rf_clip)
        else
            # Compute step size via Malitsky-Mishchenko formula
            Δx_norm = norm(x .- x_prev)
            Δg_norm = norm(∇x_curr .- ∇x_prev)

            # λ_i = min{√(1 + θ_{i-1}) * λ_{i-1}, ||x^i - x^{i-1}|| / (2||∇f(x^i) - ∇f(x^{i-1})||)}
            if Δg_norm > 1f-20
                λ_candidate = Δx_norm / (2.0f0 * Δg_norm)
                λ = min(sqrt(1.0f0 + θ_prev) * λ_prev, λ_candidate)
            else
                λ = sqrt(1.0f0 + θ_prev) * λ_prev
            end

            # Gradient descent step with updated step size
            @. x = clamp(x - λ * ∇x_curr, -rf_clip, rf_clip)
        end

        # Store for next iteration
        copyto!(x_prev, x)
        copyto!(∇x_prev, ∇x_curr)

        # Update θ_i = λ_i / λ_{i-1} for next iteration
        θ_prev = λ / λ_prev
        λ_prev = λ
    end

    return nothing
end

# ---------------- Warmup once ----------------
_gpu_warmup!()

# ---------------- Optimization parameters ----------------
nsteps  = 20
λ_init  = 1f-9     # Initial step size for Malitsky-Mishchenko
rf_clip = 2f-5     # RF amplitude clipping

# Optional benchmark (disabled by default)
const DO_BENCHMARK = false
if DO_BENCHMARK
    bt = @benchmark train_loop_mm!(xloc, nst, clip; λ_init=$λ_init) setup=(xloc = zeros(Float32, n_ctrl); nst = $nsteps; clip = $rf_clip) evals=1 samples=2 seconds=300000
    mean_s = mean(bt).time * 1e-9
    std_s  = std(bt).time  * 1e-9
    @info "Optimization loop time — mean/std" mean_s=round(mean_s, digits=4) std_s=round(std_s, digits=4) samples=length(bt.times)
end

# ---------------- Run Malitsky-Mishchenko optimization ----------------
@info "Running Malitsky-Mishchenko optimization for $nsteps steps..."
x = zeros(Float32, n_ctrl)
train_loop_mm!(x, nsteps, rf_clip; λ_init=λ_init)

# ---------------- Plot RF + demodulated profile ----------------
plt = plot_rf_and_profile(x)
savefig(plt, "profile_and_rf.png")
@info "Saved plot to profile_and_rf.png"