# RF slice design w/ Enzyme AD — CPU-ONLY (RF window aligned to Gx flat-top)
using KomaMRICore, Suppressor
using Random: seed!
using KernelAbstractions
using KernelAbstractions: @kernel, @index, CPU, @atomic
import Enzyme
using Statistics: mean, std
using LinearAlgebra: norm
using Plots
using BenchmarkTools

# ---------------- Params ----------------
seed!(42)

const B1   = 4.9e-6
const Trf  = 3.2e-3
const TBP  = 8.0
const Δz   = 6e-3
const zmax = 8e-3
const fmax = TBP / Trf
const γf64 = 2π * 42.57747892e6
const γ    = Float32(γf64)

const Nspins = 100

xline = collect(range(-zmax, zmax, length=Nspins))  # slice axis along x
Gx_val = fmax / (Float64(γf64) * Δz)

# ---------------- Koma objects ----------------
sys = Scanner()
seq_full = PulseDesigner.RF_sinc(B1, Trf, sys; G=[Gx_val; 0; 0], TBP=TBP)
seq = seq_full
obj = Phantom(; x=xline)

const Nrf = length(seq.RF[1].A)
@info "Nrf (number of RF controls)" Nrf
@info "Nspins (number of spins)" Nspins
@info "Thread count" Threads.nthreads()

# ---------------- Simulation config ----------------
sim_params = KomaMRICore.default_sim_params()
sim_params["Δt_rf"] = Trf / (2*(Nrf - 1))
sim_params["Δt"] = Inf                   # RF-only window
sim_params["return_type"] = "state"
sim_params["precision"] = "f32"
sim_params["Nthreads"] = 1
sim_params["sim_method"] = KomaMRICore.Bloch()

mag_sinc = @suppress simulate(obj, seq, sys; sim_params)

# ---------------- Target profile ----------------
butterworth_degree = 5
target_profile = 0.5im ./ (1 .+ (xline ./ (Δz / 2)).^(2*butterworth_degree))
const TARGET_R_h = Float32.(real.(target_profile))
const TARGET_I_h = Float32.(imag.(target_profile))
const TARGET_MAG = Float32.(abs.(target_profile))

# ---------------- Discretize timeline ----------------
function get_koma_timeline(seq, sim_params)
    seqd = KomaMRICore.discretize(seq; sampling_params=sim_params)
    Nt  = length(seqd.Δt)
    B1  = ComplexF32.(seqd.B1[1:Nt])
    Gx  = Float32.(seqd.Gx[1:Nt])
    Gy  = Float32.(seqd.Gy[1:Nt])
    Gz  = Float32.(seqd.Gz[1:Nt])
    Δt  = Float32.(seqd.Δt[1:Nt])
    Δf  = Float32.(seqd.Δf[1:Nt])
    t   = Float32.(seqd.t[1:Nt])
    rf_active_idx = findall(x -> abs(x) > 1e-10, B1)
    return (B1=B1, Gx=Gx, Gy=Gy, Gz=Gz, Δt=Δt, Δf=Δf, t=t,
            rf_active_idx=rf_active_idx, Nt=Int32(Nt), Nrf_original=Nrf)
end

const TL  = get_koma_timeline(seq, sim_params)
const N   = length(xline)
const N_Spins32 = Int32(N)
const N_Δt32    = TL.Nt

# Time axes
const t_edges  = cumsum(vcat(0f0, TL.Δt[1:end-1]))
const t_center = t_edges .+ 0.5f0 .* TL.Δt

# Residual kx moment for demod
const KX_RESID = sum(TL.Gx .* TL.Δt)

"""
Use original Koma RF active taps (no Gx window alignment). Linear control<->timeline map
with tap_pos and ctrl_pos in [0,1].
"""
const rf_idx   = TL.rf_active_idx
const Lrf_taps = length(rf_idx)
const n_ctrl   = Nrf
const n_taps   = Lrf_taps
const ctrl_pos = collect(Float32, range(0f0, 1f0, length=n_ctrl))
const tap_pos  = collect(Float32, range(0f0, 1f0, length=n_taps))

# ---------------- Buffers (CPU) ----------------
adapt_dev(x) = x

M_xy  = zeros(Float32, 2N)
M_z   = ones(Float32, N)

p_x   = Float32.(xline)
p_y   = zeros(Float32, N)
p_z   = zeros(Float32, N)
p_ΔBz = zeros(Float32, N)
p_T1  = fill(Float32(1e9), N)
p_T2  = fill(Float32(1e9), N)
p_ρ   = ones(Float32, N)

s_Gx  = copy(TL.Gx);  s_Gy = copy(TL.Gy);  s_Gz = copy(TL.Gz)
s_Δt  = copy(TL.Δt);  s_Δf = copy(TL.Δf)

s_B1r = zeros(Float32, length(TL.B1))
s_B1i = zeros(Float32, length(TL.B1))

B1r_host = similar(s_B1r);  B1i_host = similar(s_B1i)

target_r_dev = TARGET_R_h
target_i_dev = TARGET_I_h

# Loss buffers (CPU)
loss_buf = zeros(Float32, 1)
dloss_buf = similar(loss_buf)
loss_err = zeros(Float32, N)
loss_tmp = zeros(Float32, cld(N, 2))

const backend = CPU()

# ---------------- Kernels ----------------
@kernel inbounds=true function excitation_kernel!(
    M_xy::AbstractVector{T}, M_z::AbstractVector{T},
    p_x::AbstractVector{T}, p_y::AbstractVector{T}, p_z::AbstractVector{T},
    p_ΔBz::AbstractVector{T}, p_T1::AbstractVector{T}, p_T2::AbstractVector{T}, p_ρ::AbstractVector{T},
    s_Gx::AbstractVector{T}, s_Gy::AbstractVector{T}, s_Gz::AbstractVector{T},
    s_Δt::AbstractVector{T}, s_Δf::AbstractVector{T}, s_B1r::AbstractVector{T}, s_B1i::AbstractVector{T},
    N_Spins::Int32, N_Δt::Int32
) where {T}
    eps_d = T(1f-20)
    i = @index(Global, Linear)
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

function launch_excitation!(
    M_xy, M_z,
    p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
    s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1r, s_B1i,
    N_Spins::Int32, N_Δt::Int32, backend)

    k = excitation_kernel!(backend)
    k(M_xy, M_z,
      p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
      s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1r, s_B1i,
      N_Spins, N_Δt; ndrange=Int(N_Spins))
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
    i = @index(Global, Linear)
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

# Phase-matched loss per-spin (no atomics)
@kernel function phase_matched_loss_per_spin!(
    err::AbstractVector{Float32}, M::AbstractVector{Float32},
    target_r::AbstractVector{Float32}, target_i::AbstractVector{Float32},
    cosφ::AbstractVector{Float32},  sinφ::AbstractVector{Float32}, N_Spins::Int32
)
    i = @index(Global, Linear)
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
    j = @index(Global, Linear)
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

# Copy first element of vec into acc[1]
@kernel function write_first!(acc::AbstractVector{Float32}, vec::AbstractVector{Float32})
    i = @index(Global, Linear)
    if i == 1
        acc[1] = vec[1]
    end
end

# Launcher for fused per-spin loss
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
    excite_and_loss_per_spin!(backend)(
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

# Device loss reduction path (no atomics)
Base.@noinline function device_loss_into_buffer!(
    acc::AbstractVector{Float32},
    M_xy_dev::AbstractVector{Float32},
    target_r_dev::AbstractVector{Float32}, target_i_dev::AbstractVector{Float32},
    cosφ_dev::AbstractVector{Float32},  sinφ::AbstractVector{Float32},
    err_buf::AbstractVector{Float32}, tmp_buf::AbstractVector{Float32},
    backend)
    fill!(acc, 0f0)
    phase_matched_loss_per_spin!(backend)(
        err_buf, M_xy_dev, target_r_dev, target_i_dev, cosφ_dev, sinφ, N_Spins32; ndrange=Int(N))
    cur = err_buf; cur_len = N_Spins32
    tmp = tmp_buf
    while Int(cur_len) > 1
        out_len = Int32((Int(cur_len) + 1) >>> 1)
        reduce_pairwise!(backend)(tmp, cur, cur_len; ndrange=Int(out_len))
        KernelAbstractions.synchronize(backend)
        cur, tmp = tmp, cur
        cur_len = out_len
    end
    write_first!(backend)(acc, cur; ndrange=1)
    KernelAbstractions.synchronize(backend)
    return nothing
end

# End-to-end loss into buffer (simulate then reduce loss on device)
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

@kernel function phase_matched_seed_per_spin!(
    dM::AbstractVector{Float32}, M::AbstractVector{Float32},
    target_r::AbstractVector{Float32}, target_i::AbstractVector{Float32},
    cosφ::AbstractVector{Float32},  sinφ::AbstractVector{Float32}, λ::Float32
)
    i = @index(Global, Linear)
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

function compute_phase_arrays(Mr::Vector{Float32}, Mi::Vector{Float32})
    N  = length(Mr)
    κ  = Float32(γ) * Float32(KX_RESID)
    xh = Float32.(xline)
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

# -------- Control <-> timeline mapping over RF active taps --------
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

function scatter_grad_to_x!(∇x::Vector{Float32}, dB1r_timeline::AbstractVector{Float32})
    @assert length(dB1r_timeline) == length(TL.Δt)
    resize!(∇x, n_ctrl);  fill!(∇x, 0f0)
    wsum = zeros(Float32, n_ctrl)
    j = 1
    @inbounds for (k, idx) in enumerate(rf_idx)
        t   = tap_pos[k]
        Δtw = TL.Δt[idx]
        while j < n_ctrl && ctrl_pos[j+1] < t
            j += 1
        end
        if j == n_ctrl
            ∇x[end] += Δtw * dB1r_timeline[idx];  wsum[end] += Δtw
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

# ---------------- Forward/Loss/Grad ----------------
function forward!(x::Vector{Float32})
    map_x_to_timeline!(B1r_host, B1i_host, x)
    copyto!(s_B1r, B1r_host); copyto!(s_B1i, B1i_host)
    fill!(M_xy, 0f0); fill!(M_z, 1f0)
    excite_only!(M_xy, M_z,
        p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
        s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1r, s_B1i,
        backend)
    KernelAbstractions.synchronize(backend)
    Mr_h = collect(@view M_xy[1:N])
    Mi_h = collect(@view M_xy[N+1:2N])
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

dM_xy = similar(M_xy)
dM_z  = similar(M_z)
∇B1r  = similar(s_B1r)
∇B1i  = similar(s_B1i)

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
    L = sum(loss_err)

    dB1r_host = copy(∇B1r)
    scatter_grad_to_x!(∇x, dB1r_host)
    return L
end

# ---------------- Sanity check (demod) ----------------
let
    x0_ctrl = Float32.(real.(copy(seq.RF[1].A)))
    map_x_to_timeline!(B1r_host, B1i_host, x0_ctrl)
    copyto!(s_B1r, B1r_host); copyto!(s_B1i, B1i_host)
    fill!(M_xy, 0f0); fill!(M_z, 1f0)
    launch_excitation!(
        M_xy, M_z,
        p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
        s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1r, s_B1i,
        N_Spins32, N_Δt32, backend)
    KernelAbstractions.synchronize(backend)
    Mr = collect(@view M_xy[1:N]); Mi = collect(@view M_xy[N+1:2N])
    cosφ, sinφ = compute_phase_arrays(Mr, Mi)
    Mrot_mag = sqrt.((Mr .* cosφ .+ Mi .* sinφ).^2 .+ (Mi .* cosφ .- Mr .* sinφ).^2)
    diff_mag = Mrot_mag .- Float32.(abs.(mag_sinc.xy))
    @show sqrt(mean(diff_mag.^2))
end

# ---------------- Plotting (controls, timeline imag, profile) ----------------
function plot_rf_and_profile(x)
    map_x_to_timeline!(B1r_host, B1i_host, x)
    copyto!(s_B1r, B1r_host); copyto!(s_B1i, B1i_host)

    # simulate once to get achieved profile
    fill!(M_xy, 0f0); fill!(M_z, 1f0)
    launch_excitation!(
        M_xy, M_z,
        p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
        s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1r, s_B1i,
        N_Spins32, N_Δt32, backend)
    KernelAbstractions.synchronize(backend)
    Mr = collect(@view M_xy[1:N]); Mi = collect(@view M_xy[N+1:2N])
    cosφ, sinφ = compute_phase_arrays(Mr, Mi)
    Mrot_r = Mr .* cosφ .+ Mi .* sinφ
    Mrot_i = Mi .* cosφ .- Mr .* sinφ
    mag = sqrt.(Mrot_r .* Mrot_r .+ Mrot_i .* Mrot_i)

    p_r = plot(title="RF real (controls)", ylabel="B1_r (T)")
    plot!(p_r, 1:length(x), x; label="Re(B1) controls", lw=2)

    p_i = plot(title="RF imag (timeline)", ylabel="B1_i (T)")
    plot!(p_i, t_center[rf_idx], B1i_host[rf_idx]; label="Im(B1) taps", lw=2)

    p_prof = plot(title="|M_xy|(x)", xlabel="x (m)", ylabel="magnitude")
    plot!(p_prof, xline, TARGET_MAG; label="Target |M|", lw=3, ls=:dash)
    plot!(p_prof, xline, mag;        label="Achieved", lw=2)

    plot(p_r, p_i, p_prof; layout=(3,1), size=(900,900))
end

# ---------------- Train & plot ----------------
x  = zeros(Float32, Nrf)
∇x = similar(x)

function _cpu_warmup!()
    try; _ = loss_and_grad!(similar(x), x); catch; end
    KernelAbstractions.synchronize(backend)
end
_cpu_warmup!()

# Optimization parameters
nsteps  = 20
λ_init  = 1f-9     # Initial step size for Malitsky-Mishchenko
η_base  = 5f-10     # Base learning rate for simple gradient descent
rf_clip = 2f-5     # RF amplitude clipping

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

# bt = @benchmark train_loop_mm!(xloc, nst, clip; λ_init=$λ_init) setup=(xloc = zeros(Float32, n_ctrl); nst = $nsteps; clip = $rf_clip; λ_init = 1f-9) evals=1 samples=2 seconds=30000

# mean_s = mean(bt).time * 1e-9
# std_s  = std(bt).time  * 1e-9
# @info "Optimization loop time — mean/std" mean_s=round(mean_s, digits=4) std_s=round(std_s, digits=4) samples=length(bt.times)

for _ in 1:nsteps
    _ = loss_and_grad!(∇x, x)
    g = sqrt(mean(abs2, ∇x)) + 1f-20
    η = min(η_base, 1f-6 / (10f0 * g))
    @. x = clamp(x - η*∇x, -rf_clip, rf_clip)
end
KernelAbstractions.synchronize(backend)

plt = plot_rf_and_profile(x)
savefig(plt, "profile_and_rf.png")
@info "Saved plot to profile_and_rf.png"