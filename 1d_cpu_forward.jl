# ============================================================
# 1-D RF slice design — Enzyme JVP via timeline tangents (CPU)
# Stable: Float64 kernel, SU(2) half-angle, Δf=0 during training
# Keeps phase-matched loss; no CUDA usage (CPU backend only).
# ============================================================

using KomaMRICore, Suppressor
using Random: seed!
using KernelAbstractions
using KernelAbstractions: @kernel, @index
using Adapt
import Enzyme
using Statistics: mean, std
using LinearAlgebra: norm
using Plots
using BenchmarkTools

# ---------------- Constants / Params ----------------
seed!(42)

const B1_amp = 4.9e-6
const Trf    = 3.2e-3
const TBP    = 8.0
const Δz     = 6e-3
const zmax   = 8e-3
const fmax   = TBP / Trf
const γ64    = 2π * 42.57747892e6

Nspins = 100
const GROUP_SIZE = 256

z  = collect(range(-zmax, zmax, length=Nspins))
Gz = fmax / (γ64 * Δz)

# ---------------- Koma objects ----------------
sys = Scanner()
seq_full = PulseDesigner.RF_sinc(B1_amp, Trf, sys; G=[Gz; 0; 0], TBP=TBP)
seq = seq_full
obj = Phantom(; x=z)

# Controls on original RF grid
const Nrf = length(seq.RF[1].A)  

# ---------------- Simulation config ----------------
sim_params = KomaMRICore.default_sim_params()
sim_params["Δt_rf"]       = Trf / (2*(Nrf - 1))   # Koma internal ×2 upsample
sim_params["Δt"]          = Inf                   # RF-only window
sim_params["return_type"] = "state"
sim_params["precision"]   = "f64"
sim_params["Nthreads"]    = 1
sim_params["sim_method"]  = KomaMRICore.Bloch()

# Baseline (for demod/compare)
mag_sinc = @suppress simulate(obj, seq, sys; sim_params)

# ---------------- Target profile ----------------
butterworth_degree = 5
target_profile = 0.5im ./ (1 .+ (z ./ (Δz / 2)).^(2*butterworth_degree))
const TARGET_R_h = Float64.(real.(target_profile))
const TARGET_I_h = Float64.(imag.(target_profile))
const TARGET_MAG = Float64.(abs.(target_profile))

# ---------------- Backend ----------------
const backend = KernelAbstractions.CPU()

# =============================================================
# Discretize Koma timeline (RF-only window)
# =============================================================
function get_koma_timeline(seq, sim_params)
    seqd = KomaMRICore.discretize(seq; sampling_params=sim_params)
    Nt = length(seqd.Δt)
    B1i = ComplexF64.(seqd.B1[1:Nt])
    Gxi = Float64.(seqd.Gx[1:Nt])
    Gyi = Float64.(seqd.Gy[1:Nt])
    Gzi = Float64.(seqd.Gz[1:Nt])
    Δti = Float64.(seqd.Δt[1:Nt])
    Δfi = Float64.(seqd.Δf[1:Nt])       # angular freq [rad/s]
    ti  = Float64.(seqd.t[1:Nt])
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
# Host buffers (Float64 everywhere)
# =============================================================
const N         = length(z)
const N_Spins32 = Int32(N)
const N_Δt32    = TL.Nt

adapt_dev(x) = x  # CPU: identity

# State
M_xy  = zeros(Float64, 2N)   # [1:N]=Re, [N+1:2N]=Im
M_z   = ones(Float64, N)

# Spins/params
p_x   = Float64.(z)
p_y   = zeros(Float64, N)
p_z   = zeros(Float64, N)
p_ΔBz = zeros(Float64, N)
p_T1  = fill(Float64(1e9), N)
p_T2  = fill(Float64(1e9), N)
p_ρ   = ones(Float64, N)

# Timeline
s_Gx  = copy(TL.Gx)
s_Gy  = copy(TL.Gy)
s_Gz  = copy(TL.Gz)
s_Δt  = copy(TL.Δt)
s_Δf  = copy(TL.Δf)

# RF (we overwrite only rf_idx; Im stays 0 for real-only controls)
s_B1r = real.(TL.B1)
s_B1i = imag.(TL.B1)

# Targets (host copies for seeding)
target_r_dev = TARGET_R_h
target_i_dev = TARGET_I_h

# Host RF scratch (full timeline)
B1r_host = copy(real.(TL.B1))
B1i_host = copy(imag.(TL.B1))

# =============================================================
# Kernels (Bloch) — Float64 + SU(2) half-angle + tiny guards
# =============================================================
@kernel inbounds=true function excitation_kernel!(
    M_xy::AbstractVector{T}, M_z::AbstractVector{T},
    p_x::AbstractVector{T}, p_y::AbstractVector{T}, p_z::AbstractVector{T},
    p_ΔBz::AbstractVector{T}, p_T1::AbstractVector{T}, p_T2::AbstractVector{T}, p_ρ::AbstractVector{T},
    s_Gx::AbstractVector{T}, s_Gy::AbstractVector{T}, s_Gz::AbstractVector{T},
    s_Δt::AbstractVector{T}, s_Δf::AbstractVector{T}, s_B1r::AbstractVector{T}, s_B1i::AbstractVector{T},
    N_Spins::Int32, N_Δt::Int32
) where {T}
    eps_d = T(1e-12)
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
            Δt  = s_Δt[s_idx]; df = s_Δf[s_idx]          # [rad/s]
            b1r = s_B1r[s_idx]; b1i = s_B1i[s_idx]

            Bz = (x*gx + y*gy + z*gz) + ΔBz - df / T(γ64)

            # |B| and SU(2) half-angle
            B2 = b1r*b1r + b1i*b1i + Bz*Bz
            B  = sqrt(B2)
            φ  = T(0.5) * T(γ64) * B * Δt
            φ  = ifelse(isfinite(φ), φ, zero(T))   # tiny guard
            sφ = sin(φ)
            cφ = cos(φ)
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
            Mz = Mz_new * ΔT1 + ρ * (T(1.0) - ΔT1)

            s_idx += 1
        end

        M_xy[ir] = Mx
        M_xy[ii] = My
        M_z[i]   = Mz
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

# =============================================================
# Phase-matched loss pieces (Float64)
# =============================================================
@kernel function phase_matched_seed_per_spin!(
    dM::AbstractVector{Float64}, M::AbstractVector{Float64},
    target_r::AbstractVector{Float64}, target_i::AbstractVector{Float64},
    cosφ::AbstractVector{Float64},  sinφ::AbstractVector{Float64}, λ::Float64
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
        scale = λ / Float64(N)
        g_rot_r = scale * (Mrot_r - Tr)
        g_rot_i = scale * (Mrot_i - Ti)
        dM[ir] =  g_rot_r * c - g_rot_i * s
        dM[ii] =  g_rot_r * s + g_rot_i * c
    end
end

# Global phase φ0 and spatial demod κ·x
function compute_phase_arrays(Mr::Vector{Float64}, Mi::Vector{Float64})
    N  = length(Mr)
    κ  = γ64 * KX_RESID     # [rad/m]
    xh = Float64.(z)
    acc_r = 0.0; acc_i = 0.0
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
const ctrl_pos = collect(range(0.0, 1.0, length=n_ctrl))
const tap_pos  = collect(range(0.0, 1.0, length=n_taps))

function map_x_to_timeline!(B1r::Vector{Float64}, B1i::Vector{Float64}, x::Vector{Float64})
    @assert length(x) == n_ctrl
    B1r .= real.(TL.B1);  B1i .= imag.(TL.B1)    # start from baseline (Koma's discretization)
    j = 1
    @inbounds for (k, idx) in enumerate(rf_idx)
        t = tap_pos[k]
        while j < n_ctrl && ctrl_pos[j+1] < t
            j += 1
        end
        if j == n_ctrl
            B1r[idx] = x[end]
            B1i[idx] = 0.0
        else
            α = (t - ctrl_pos[j]) / (ctrl_pos[j+1] - ctrl_pos[j] + 1e-20)
            B1r[idx] = (1.0-α)*x[j] + α*x[j+1]
            B1i[idx] = 0.0
        end
    end
    return nothing
end

# Adjoint of the linear interp (timeline → controls), unweighted
function scatter_grad_to_x!(∇x::Vector{Float64}, dB1r_timeline::AbstractVector{Float64})
    @assert length(dB1r_timeline) == length(TL.Δt)
    resize!(∇x, n_ctrl);  fill!(∇x, 0.0)
    j = 1
    @inbounds for (k, idx) in enumerate(rf_idx)
        t   = tap_pos[k]
        while j < n_ctrl && ctrl_pos[j+1] < t
            j += 1
        end
        if j == n_ctrl
            ∇x[end] += dB1r_timeline[idx]
        else
            α  = (t - ctrl_pos[j]) / (ctrl_pos[j+1] - ctrl_pos[j] + 1e-20)
            w0 = (1.0 - α)
            w1 = α
            ∇x[j]   += w0 * dB1r_timeline[idx]
            ∇x[j+1] += w1 * dB1r_timeline[idx]
        end
    end
    return ∇x
end

# Helper: build RF timeline tangent from a control-direction (baseline-subtracted)
function make_timeline_tangent!(dB1r_host::Vector{Float64}, dB1i_host::Vector{Float64},
                                dir_ctrl::Vector{Float64}, B1r_base::Vector{Float64}, B1i_base::Vector{Float64})
    # start from baseline
    copyto!(dB1r_host, B1r_base); copyto!(dB1i_host, B1i_base)
    # map (baseline + dir), then subtract baseline → pure tangent
    map_x_to_timeline!(dB1r_host, dB1i_host, dir_ctrl)
    @inbounds for k in eachindex(dB1r_host)
        dB1r_host[k] -= B1r_base[k]
        dB1i_host[k] -= B1i_base[k]   # will be zero for real-only controls
    end
    return nothing
end

# =============================================================
# Forward, Loss, Grad (timeline-tangent JVPs via Enzyme)
# =============================================================
function forward!(x::Vector{Float64})
    map_x_to_timeline!(B1r_host, B1i_host, x)
    copyto!(s_B1r, B1r_host); copyto!(s_B1i, B1i_host)
    # Disable off-resonance during optimization for stability
    fill!(s_Δf, 0.0)

    fill!(M_xy, 0.0); fill!(M_z, 1.0)
    excite_only!(M_xy, M_z,
        p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
        s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1r, s_B1i,
        backend)
    KernelAbstractions.synchronize(backend)
    Mr_h = view(M_xy, 1:N) |> collect
    Mi_h = view(M_xy, N+1:2N) |> collect
    cosφ, sinφ = compute_phase_arrays(Mr_h, Mi_h)
    loss_sum = 0.0
    @inbounds for i in 1:N
        c = cosφ[i]; s = sinφ[i]
        Mrot_r = Mr_h[i]*c + Mi_h[i]*s
        Mrot_i = Mi_h[i]*c - Mr_h[i]*s
        dr = Mrot_r - TARGET_R_h[i]
        di = Mrot_i - TARGET_I_h[i]
        loss_sum += dr*dr + di*di
    end
    L = 0.5 * loss_sum / Float64(N)
    return L, (cosφ=cosφ, sinφ=sinφ)
end

# Scratch
dM_xy = similar(M_xy)
dM_z  = similar(M_z)
# Tangent buffer for forward-mode JVP of M_xy
dM_xy_tan = similar(M_xy)
tB1r = similar(s_B1r)
tB1i = similar(s_B1i)

function loss_and_grad!(∇x::Vector{Float64}, x::Vector{Float64})
    # ----- primal forward -----
    L, cφ = forward!(x)

    # build ∂L/∂M on device (phase-matched) from the *current* M
    fill!(dM_xy, 0.0)
    phase_matched_seed_per_spin!(backend, GROUP_SIZE)(
        dM_xy, M_xy, target_r_dev, target_i_dev,
        adapt_dev(cφ.cosφ), adapt_dev(cφ.sinφ), 1.0; ndrange=Int(N))
    KernelAbstractions.synchronize(backend)
    dM_seed_host = collect(dM_xy)

    # ----- precompute baseline timeline for tangent extraction -----
    B1r_base = copy(real.(TL.B1))
    B1i_base = copy(imag.(TL.B1))
    zero_ctrl = zeros(Float64, n_ctrl)
    map_x_to_timeline!(B1r_base, B1i_base, zero_ctrl)

    # host scratch for RF timeline tangent
    dB1r_host = similar(B1r_base)
    dB1i_host = similar(B1i_base)

    # gradient accumulator on timeline
    dB1r_timeline = zeros(Float64, length(B1r_host))

    # ----- loop over control directions; real-only controls -----
    for j in 1:n_ctrl
        # control basis direction (real)
        dir = zeros(Float64, n_ctrl); dir[j] = 1.0
        make_timeline_tangent!(dB1r_host, dB1i_host, dir, B1r_base, B1i_base)
        copyto!(tB1r, dB1r_host); copyto!(tB1i, dB1i_host)  # tB1i ≡ 0

        # reset primal & tangent states
        fill!(M_xy, 0.0); fill!(M_z, 1.0)
        fill!(dM_xy_tan, 0.0); fill!(dM_z, 0.0)

        # keep Δf = 0 during JVP as well
        fill!(s_Δf, 0.0)

        Enzyme.autodiff(
            Enzyme.Forward,                      # JVP
            excite_only!,
            Enzyme.Duplicated(M_xy, dM_xy_tan),  # output tangent J*v
            Enzyme.Duplicated(M_z,  dM_z),
            Enzyme.Const(p_x), Enzyme.Const(p_y), Enzyme.Const(p_z),
            Enzyme.Const(p_ΔBz), Enzyme.Const(p_T1), Enzyme.Const(p_T2), Enzyme.Const(p_ρ),
            Enzyme.Const(s_Gx), Enzyme.Const(s_Gy), Enzyme.Const(s_Gz),
            Enzyme.Const(s_Δt), Enzyme.Const(s_Δf),
            Enzyme.Duplicated(s_B1r, tB1r),      # seed input tangents
            Enzyme.Duplicated(s_B1i, tB1i),
            Enzyme.Const(backend),
        )
        KernelAbstractions.synchronize(backend)

        # contract with \bar{M} to get dL/d (timeline dir j)
        dM_tan_host = collect(dM_xy_tan)
        @assert all(isfinite, dM_tan_host)
        dL_d_dir = sum(dM_tan_host .* dM_seed_host)   # scalar

        # distribute scalar onto the timeline basis we used (linear map adjoint)
        # i.e., accumulate timeline gradient components proportionally to the seed
        @inbounds for k in eachindex(dB1r_host)
            dB1r_timeline[k] += dL_d_dir * (abs(dB1r_host[k]) > 0 ? 1.0 : 0.0)
        end
    end

    # scatter timeline gradient back to controls
    scatter_grad_to_x!(∇x, dB1r_timeline)

    return L
end

# =============================================================
# Sanity check vs Koma (forward-only; demod for compare)
# =============================================================
let
    x0_ctrl = Float64.(real.(seq.RF[1].A))
    map_x_to_timeline!(B1r_host, B1i_host, x0_ctrl)
    copyto!(s_B1r, B1r_host); copyto!(s_B1i, B1i_host)
    fill!(M_xy, 0.0); fill!(M_z, 1.0)
    launch_excitation!(
        M_xy, M_z,
        p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
        s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1r, s_B1i,
        N_Spins32, N_Δt32, backend)
    KernelAbstractions.synchronize(backend)
    Mr = view(M_xy, 1:N) |> collect; Mi = view(M_xy, N+1:2N) |> collect
    cosφ, sinφ = compute_phase_arrays(Mr, Mi)
    Mrot_mag = sqrt.((Mr .* cosφ .+ Mi .* sinφ).^2 .+ (Mi .* cosφ .- Mr .* sinφ).^2)
    diff_mag = Mrot_mag .- Float64.(abs.(mag_sinc.xy))
    @show sqrt(mean(diff_mag.^2))
end

function plot_rf_and_profile(x)
    map_x_to_timeline!(B1r_host, B1i_host, x)
    copyto!(s_B1r, B1r_host); fill!(s_B1i, 0.0)
    fill!(M_xy, 0.0); fill!(M_z, 1.0)
    launch_excitation!(
        M_xy, M_z,
        p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
        s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1r, s_B1i,
        N_Spins32, N_Δt32, backend)
    KernelAbstractions.synchronize(backend)

    Mr = view(M_xy, 1:N) |> collect; Mi = view(M_xy, N+1:2N) |> collect
    cosφ, sinφ = compute_phase_arrays(Mr, Mi)
    Mrot_r = Mr .* cosφ .+ Mi .* sinφ
    Mrot_i = Mi .* cosφ .- Mr .* sinφ
    mag = sqrt.(Mrot_r .* Mrot_r .+ Mrot_i .* Mrot_i)

    p_r = plot(title="RF real (controls)", ylabel="B1_r (T)"); plot!(p_r, 1:length(x), x; label="Re(B1) controls", lw=2)
    t_edges  = cumsum(vcat(0.0, TL.Δt[1:end-1])); t_center = t_edges .+ 0.5 .* TL.Δt
    p_i = plot(title="RF imag (timeline)", ylabel="B1_i (T)"); plot!(p_i, t_center[rf_idx], B1i_host[rf_idx]; label="Im(B1) taps", lw=2)
    p_prof = plot(title="|M_xy|(z)", xlabel="z (m)", ylabel="magnitude")
    plot!(p_prof, z, TARGET_MAG; label="Target |M|", lw=3, ls=:dash)
    plot!(p_prof, z, mag;        label="Achieved", lw=2)
    plot(p_r, p_i, p_prof; layout=(3,1), size=(900,900))
end

# =============================================================
# Optimize in control space (Nrf) — with guards
# =============================================================

"""
Malitsky-Mishchenko adaptive gradient descent optimizer.

Parameters:
    λ_i = min{√(1 + θ_{i-1}) * λ_{i-1}, ||x^i - x^{i-1}|| / (2||∇f(x^i) - ∇f(x^{i-1})||)}
    x^{i+1} = x^i - λ_i * ∇f(x^i)
    θ_i = λ_i / λ_{i-1}

Reference: Malitsky & Mishchenko (2020), "Adaptive gradient descent without descent"
"""
function train_loop_mm!(x::Vector{Float64}, nsteps::Int, rf_clip::Float64; λ_init::Float64=1e-9)
    # Preallocate buffers
    ∇x_curr = zeros(Float64, length(x))
    ∇x_prev = zeros(Float64, length(x))
    x_prev = copy(x)

    λ_prev = λ_init
    θ_prev = 1.0

    @inbounds for i in 1:nsteps
        # Compute current gradient
        _ = loss_and_grad!(∇x_curr, x)
        @. ∇x_curr = ifelse(isfinite(∇x_curr), ∇x_curr, 0.0)
        KernelAbstractions.synchronize(backend)

        if i == 1
            # First iteration: simple gradient step
            λ = λ_init
            @. x = clamp(x - λ * ∇x_curr, -rf_clip, rf_clip)
        else
            # Compute step size via Malitsky-Mishchenko formula
            Δx_norm = norm(x .- x_prev)
            Δg_norm = norm(∇x_curr .- ∇x_prev)

            # λ_i = min{√(1 + θ_{i-1}) * λ_{i-1}, ||x^i - x^{i-1}|| / (2||∇f(x^i) - ∇f(x^{i-1})||)}
            if Δg_norm > 1e-20
                λ_candidate = Δx_norm / (2.0 * Δg_norm)
                λ = min(sqrt(1.0 + θ_prev) * λ_prev, λ_candidate)
            else
                λ = sqrt(1.0 + θ_prev) * λ_prev
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

    KernelAbstractions.synchronize(backend)
    return nothing
end

# ---------------- Warmup once ----------------
function _cpu_warmup!()
    x_w = zeros(Float64, Nrf)
    ∇x_w = similar(x_w)
    try
        loss_and_grad!(∇x_w, x_w)
    catch
        # ignore first-call hiccups
    end
    KernelAbstractions.synchronize(backend)
    return nothing
end
_cpu_warmup!()

# ---------------- Optimization parameters ----------------
nsteps  = 20
λ_init  = 1e-9     # Initial step size for Malitsky-Mishchenko
rf_clip = 2e-5     # RF amplitude clipping

# Benchmark Malitsky-Mishchenko optimizer
bt = @benchmark train_loop_mm!(xloc, nst, clip; λ_init=$λ_init) setup=(xloc = zeros(Float64, n_ctrl); nst = $nsteps; clip = $rf_clip) evals=1 samples=20 seconds=3000000000000

mean_s = mean(bt).time * 1e-9
std_s  = std(bt).time  * 1e-9
@info "Optimization loop time — mean/std (CPU, MM)" mean_s=round(mean_s, digits=4) std_s=round(std_s, digits=4) samples=length(bt.times)

# # ---------------- Run once (produce a solution to plot) ----------------
# @info "Running Malitsky-Mishchenko optimization for $nsteps steps (CPU)..."
# x = zeros(Float64, n_ctrl)
# train_loop_mm!(x, nsteps, rf_clip; λ_init=λ_init)
# plt = plot_rf_and_profile(x)
# savefig(plt, "profile_and_rf_cpu.png")
# @info "Saved plot to profile_and_rf_cpu.png"
