# RF slice design w/ Enzyme AD — CPU-ONLY + CairoMakie
# - CPU simulation/AD (KernelAbstractions.CPU)
# - RF window aligned to Gx flat-top
# - Publication-quality GIF + static figure
# - Asymmetric RF y-axis (true min/max)
# - Rounded, "nice" ticks + larger typography

ENV["GKSwstype"]    = "100"
ENV["GKS_ENCODING"] = "utf8"

using KomaMRICore, Suppressor
using Random: seed!
using KernelAbstractions
using KernelAbstractions: @kernel, @index, CPU
import Enzyme
using Statistics: mean, std
using LinearAlgebra: norm
using BenchmarkTools
using Dates
using ImageMagick             # GIF backend for CairoMakie
import Base.Filesystem: mkpath

using CairoMakie

mkpath("figs")

# ---------- Publication theme ----------
set_theme!(
    theme_latexfonts();
    fontsize = 28,                 # base font
    Axis = (
        titlesize       = 30,
        xlabelsize      = 26,
        ylabelsize      = 26,
        xticklabelsize  = 20,
        yticklabelsize  = 20,
        xgridwidth      = 1.1,
        ygridwidth      = 1.1,
        xgridcolor      = (:black, 0.10),
        ygridcolor      = (:black, 0.10),
    ),
    Legend = (labelsize = 20,)
)

# Labels (μ)
const LAB_RF      = "B₁ (μT)"
const LAB_TIME    = "Time (ms)"
const LAB_PROF_Y  = "|M_xy(z)|"
const LAB_PROF_X  = "z (mm)"
const LAB_LOSS    = "Loss"
lettered(title::AbstractString, ch) = "($(ch)) " * title

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

const Nspins = 50_000

xline = collect(range(-zmax, zmax, length=Nspins))  # slice axis along x
Gx_val = fmax / (Float64(γf64) * Δz)

# ---------------- Koma objects ----------------
sys = Scanner()
seq_full = PulseDesigner.RF_sinc(B1, Trf, sys; G=[Gx_val; 0; 0], TBP=TBP)
seq = seq_full
obj = Phantom(; x=xline)

const Nrf = length(seq.RF[1].A)

# ---------------- Simulation config (CPU) ----------------
sim_params = KomaMRICore.default_sim_params()
sim_params["Δt_rf"]       = Trf / (2*(Nrf - 1))
sim_params["Δt"]          = 1f-3          # include gradients
sim_params["return_type"] = "state"
sim_params["precision"]   = "f32"
sim_params["Nthreads"]    = 1
sim_params["sim_method"]  = KomaMRICore.Bloch()
sim_params["precession_groupsize"]  = 64
sim_params["excitation_groupsize"]  = 64

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
const t_center_ms = t_center .* 1e3

# Residual kx moment for demod
const KX_RESID = sum(TL.Gx .* TL.Δt)

# ---------------- Align RF window to Gx flat-top ----------------
gx = TL.Gx
gx_max = maximum(gx)
thr = 0.98f0 * gx_max
flat = findall(gx .>= thr)
if isempty(flat)
    flat = findall(gx .> 0f0)
end
@assert !isempty(flat) "No positive Gx region detected."
i_up = first(flat)
i_dn = last(flat)

const Lrf_taps = length(TL.rf_active_idx)
const rf_window_len = t_center[TL.rf_active_idx[end]] - t_center[TL.rf_active_idx[1]]

t_flat_start = t_center[i_up]
t_flat_end   = t_center[i_dn]
t_flat_len   = max(t_flat_end - t_flat_start, 1f-12)
t_start_desired = t_flat_start + 0.5f0*(t_flat_len - rf_window_len)
t_start = clamp(t_start_desired, t_center[1], t_center[end] - rf_window_len)

start_k = searchsortedfirst(t_center, t_start)
start_k = min(max(start_k, 1), length(t_center) - Lrf_taps + 1)
const ACT_IDX = collect(start_k:(start_k + Lrf_taps - 1))

# normalized tap positions in aligned window
rf_t   = Float32.(t_center[ACT_IDX])
span   = max(rf_t[end] - rf_t[1], 1f-12)
const tap_pos  = (rf_t .- rf_t[1]) ./ span
const ctrl_pos = collect(Float32, range(0f0, 1f0, length=Nrf))

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

# -------- Control <-> timeline mapping inside ACT_IDX --------
const n_ctrl = Nrf

function map_x_to_timeline!(B1r::Vector{Float32}, B1i::Vector{Float32}, x::Vector{Float32})
    @assert length(x) == n_ctrl
    fill!(B1r, 0f0);  fill!(B1i, 0f0)
    j = 1
    @inbounds for (k, idx) in enumerate(ACT_IDX)
        t = (t_center[idx] - t_center[ACT_IDX[1]]) / (t_center[ACT_IDX[end]] - t_center[ACT_IDX[1]] + 1f-20)
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
    @inbounds for (k, idx) in enumerate(ACT_IDX)
        t = (t_center[idx] - t_center[ACT_IDX[1]]) / (t_center[ACT_IDX[end]] - t_center[ACT_IDX[1]] + 1f-20)
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
        if wsum[i] > 0f0;  ∇x[i] /= wsum[i];  end
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
    # magnitude snapshot for plotting
    Mrot_r = Mr_h .* cosφ .+ Mi_h .* sinφ
    Mrot_i = Mi_h .* cosφ .- Mr_h .* sinφ
    Mmag_h = sqrt.(Mrot_r .* Mrot_r .+ Mrot_i .* Mrot_i)
    return L, Mmag_h
end

dM_xy = similar(M_xy)
dM_z  = similar(M_z)
∇B1r  = similar(s_B1r)
∇B1i  = similar(s_B1i)

function loss_and_grad!(∇x::Vector{Float32}, x::Vector{Float32})
    L, _ = forward!(x)
    Mr_h = collect(@view M_xy[1:N])
    Mi_h = collect(@view M_xy[N+1:2N])
    cosφ, sinφ = compute_phase_arrays(Mr_h, Mi_h)

    fill!(dM_xy, 0f0)
    phase_matched_seed_per_spin!(backend)(
        dM_xy, M_xy, target_r_dev, target_i_dev,
        cosφ, sinφ, 1f0; ndrange=Int(N))
    KernelAbstractions.synchronize(backend)
    fill!(∇B1r, 0f0); fill!(∇B1i, 0f0); fill!(dM_z, 0f0)
    Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        excite_only!,
        Enzyme.Duplicated(M_xy, dM_xy),
        Enzyme.Duplicated(M_z,  dM_z),
        Enzyme.Const(p_x), Enzyme.Const(p_y), Enzyme.Const(p_z),
        Enzyme.Const(p_ΔBz), Enzyme.Const(p_T1), Enzyme.Const(p_T2), Enzyme.Const(p_ρ),
        Enzyme.Const(s_Gx), Enzyme.Const(s_Gy), Enzyme.Const(s_Gz),
        Enzyme.Const(s_Δt), Enzyme.Const(s_Δf),
        Enzyme.Duplicated(s_B1r, ∇B1r),
        Enzyme.Duplicated(s_B1i, ∇B1i),
        Enzyme.Const(backend),
    )
    KernelAbstractions.synchronize(backend)
    dB1r_host = copy(∇B1r)
    scatter_grad_to_x!(∇x, dB1r_host)

    # also return magnitude for logging
    Mrot_r = Mr_h .* cosφ .+ Mi_h .* sinφ
    Mrot_i = Mi_h .* cosφ .- Mr_h .* sinφ
    Mmag_h = sqrt.(Mrot_r .* Mrot_r .+ Mrot_i .* Mrot_i)
    return L, Mmag_h
end

# ---------------- Train & log ----------------
x   = zeros(Float32, Nrf)
∇x  = similar(x)

loss_hist = Float32[]
mag_hist  = Vector{Float32}[]   # |M_xy|(x) each iter
x_hist    = Vector{Vector{Float32}}()   # controls each iter

function _cpu_warmup!()
    try; _ = loss_and_grad!(similar(x), x); catch; end
    KernelAbstractions.synchronize(backend)
end
_cpu_warmup!()

nsteps  = 20
η_base  = 5f-10
rf_clip = 2f-5

function train_loop!(x::Vector{Float32}, ∇x::Vector{Float32}, nsteps::Int, η::Float32, clip::Float32)
    for k in 1:nsteps
        L, Mmag = loss_and_grad!(∇x, x)
        push!(loss_hist, L)
        push!(mag_hist,  copy(Mmag))
        push!(x_hist,    copy(x))
        g = sqrt(mean(abs2, ∇x)) + 1f-20
        η = min(η_base, 1f-6 / (10f0 * g))
        @. x = clamp(x - η*∇x, -clip, clip)
    end
    KernelAbstractions.synchronize(backend)
end

train_loop!(x, ∇x, nsteps, η_base, rf_clip)

# ---------------- Helpers for plotting ----------------
# Asymmetric RF limits over all frames (μT), with modest padding
function rf_limits_asym_µT(x_hist::Vector{Vector{Float32}})
    lo =  Inf32
    hi = -Inf32
    for xk in x_hist
        map_x_to_timeline!(B1r_host, B1i_host, xk)
        rf = @view B1r_host[ACT_IDX]
        lo = min(lo, minimum(rf))
        hi = max(hi, maximum(rf))
    end
    loµ = lo * 1e6
    hiµ = hi * 1e6
    rng = max(hiµ - loµ, 1f-6)
    pad_lo = 0.02f0 * rng           # 2% below
    pad_hi = 0.04f0 * rng           # 4% above
    return (loµ - pad_lo, hiµ + pad_hi)
end

# Simple "nice" tick generator (1,2,2.5,5 × 10^k)
function _nice_step(rng::Float32, nticks::Int=5)
    raw = rng / max(nticks - 1, 1)
    e = floor(log10(raw))
    b = raw / 10f0^e
    nb = b < 1.5f0 ? 1f0 : b < 2.5f0 ? 2f0 : b < 4f0 ? 2.5f0 : b < 7.5f0 ? 5f0 : 10f0
    return Float32(nb) * 10f0^e
end
function nice_ticks(lo::Float32, hi::Float32; nticks::Int=5)
    if !(isfinite(lo) && isfinite(hi)) || lo == hi
        return Float32[lo]
    end
    step = _nice_step(hi - lo, nticks)
    s = floor(lo/step) * step
    t = ceil(hi/step) * step
    collect(Float32, s:step:t)
end
nice_labels(ts; digits=6) = [string(round(t, digits=digits)) for t in ts]

rf_xlims = (minimum(t_center_ms[ACT_IDX]), maximum(t_center_ms[ACT_IDX]))
rf_ylims = rf_limits_asym_µT(x_hist)

z_mm = xline .* 1e3
z_pass = Δz * 1e3 / 2

# Profile limits (from target + first achieved)
prof_first = mag_hist[1]
prof_min = min(minimum(prof_first), minimum(TARGET_MAG))
prof_max = max(maximum(prof_first), maximum(TARGET_MAG))
if prof_min == prof_max
    padp = max(abs(prof_min)*0.1, 1e-6)
    prof_min -= padp; prof_max += padp
else
    d = prof_max - prof_min
    prof_min -= 0.05*d; prof_max += 0.05*d
end
prof_xlims = (minimum(z_mm), maximum(z_mm))
prof_ylims = (prof_min, prof_max)

# Loss limits (log scale)
loss_min, loss_max = extrema(loss_hist)
lower = max(loss_min * 0.9, 1e-12)
upper = max(loss_max * 1.1, lower * 1.001)
loss_xlims = (1, length(loss_hist))
loss_ylims = (lower, upper)

# ---------------- Animation (GIF) ----------------
let nframes = length(loss_hist)
    fig = Figure(size = (1500, 1050), backgroundcolor = :white)

    ax_rf   = Axis(fig[1, 1], title = lettered("Optimized RF Pulse", 'A'),
                   xlabel = LAB_TIME, ylabel = LAB_RF)
    ax_prof = Axis(fig[1, 2], title = lettered("Slice profile", 'B'),
                   xlabel = LAB_PROF_X, ylabel = LAB_PROF_Y)
    ax_loss = Axis(fig[2, 1:2], title = lettered("Loss evolution", 'C'),
                   xlabel = "Iteration", ylabel = LAB_LOSS, yscale = Makie.log10)

    # Guides & legend
    lines!(ax_prof, z_mm, TARGET_MAG; linestyle = :dash, color = (:black, 0.78), linewidth = 2.6, label = "Target")
    axislegend(ax_prof, position = :rt)

    # Stable limits
    xlims!(ax_rf, rf_xlims...);   ylims!(ax_rf, rf_ylims...)
    xlims!(ax_prof, prof_xlims...); ylims!(ax_prof, prof_ylims...)
    xlims!(ax_loss, loss_xlims...); ylims!(ax_loss, loss_ylims...)

    # Nice rounded ticks for RF axes
    yl, yh = Float32(rf_ylims[1]), Float32(rf_ylims[2])
    rf_yt = nice_ticks(yl, yh; nticks=5)
    ax_rf.yticks = (rf_yt, nice_labels(rf_yt; digits=6))
    # x ticks in ms (rounded)
    xl, xh = Float32(rf_xlims[1]), Float32(rf_xlims[2])
    rf_xt = nice_ticks(xl, xh; nticks=6)
    ax_rf.xticks = (rf_xt, nice_labels(rf_xt; digits=3))

    # Observables
    map_x_to_timeline!(B1r_host, B1i_host, x_hist[1])
    rf_t  = Observable(t_center_ms[ACT_IDX])
    rf_y  = Observable(B1r_host[ACT_IDX] .* 1e6)
    prof_y = Observable(mag_hist[1])
    loss_x = Observable(1:1)
    loss_y = Observable(loss_hist[1:1])

    # Thicker, clearer lines
    lines!(ax_rf,   rf_t,   rf_y;   linewidth = 3.0)
    lines!(ax_prof, z_mm,   prof_y; linewidth = 4.0, label = "Achieved")
    lines!(ax_loss, loss_x, loss_y; linewidth = 2.4)

    gif_path = joinpath("figs", "enzyme_rf_training_evolution.gif")

    CairoMakie.record(fig, gif_path; framerate = 6) do io
        for k in 1:nframes
            map_x_to_timeline!(B1r_host, B1i_host, x_hist[k])
            rf_y[]   = B1r_host[ACT_IDX] .* 1e6
            prof_y[] = mag_hist[k]
            loss_x[] = 1:k
            loss_y[] = loss_hist[1:k]
            # re-assert asymmetric limits (just in case)
            ylims!(ax_rf, rf_ylims...)
            recordframe!(io)
        end
    end
    @info "Saved GIF" path=gif_path
end

# ---------------- Final static 2x1+1 panel (PDF + PNG) ----------------
let
    fig = Figure(size = (1500, 1050), backgroundcolor = :white)

    ax_rf   = Axis(fig[1, 1], title = lettered("Real-only RF controls (final)", 'A'),
                   xlabel = LAB_TIME, ylabel = LAB_RF)
    ax_prof = Axis(fig[1, 2], title = lettered("Slice profile (final)", 'B'),
                   xlabel = LAB_PROF_X, ylabel = LAB_PROF_Y)
    ax_loss = Axis(fig[2, 1:2], title = lettered("Loss evolution", 'C'),
                   xlabel = "Iteration", ylabel = LAB_LOSS, yscale = Makie.log10)

    # Final RF over time
    map_x_to_timeline!(B1r_host, B1i_host, x_hist[end])
    lines!(ax_rf, t_center_ms[ACT_IDX], B1r_host[ACT_IDX] .* 1e6; linewidth = 3.0)

    # Final profile vs target
    lines!(ax_prof, z_mm, mag_hist[end]; linewidth = 4.0, label = "Achieved")
    lines!(ax_prof, z_mm, TARGET_MAG; linestyle = :dash, color = (:black, 0.78), linewidth = 2.6, label = "Target")
    axislegend(ax_prof, position = :rt)

    # Loss curve
    lines!(ax_loss, 1:length(loss_hist), loss_hist; linewidth = 2.4)

    # Apply consistent limits + nice ticks
    xlims!(ax_rf, rf_xlims...);   ylims!(ax_rf, rf_ylims...)
    yl, yh = Float32(rf_ylims[1]), Float32(rf_ylims[2])
    rf_yt = nice_ticks(yl, yh; nticks=5)
    ax_rf.yticks = (rf_yt, nice_labels(rf_yt; digits=6))
    xl, xh = Float32(rf_xlims[1]), Float32(rf_xlims[2])
    rf_xt = nice_ticks(xl, xh; nticks=6)
    ax_rf.xticks = (rf_xt, nice_labels(rf_xt; digits=3))

    xlims!(ax_prof, prof_xlims...); ylims!(ax_prof, prof_ylims...)
    xlims!(ax_loss, loss_xlims...); ylims!(ax_loss, loss_ylims...)

    pdf_path = joinpath("figs", "enzyme_rf_slice_design_panels.pdf")
    png_path = joinpath("figs", "enzyme_rf_slice_design_panels.png")
    save(pdf_path, fig)
    save(png_path, fig)
    @info "Saved static figure" pdf=pdf_path png=png_path

    println("\nOutputs:")
    println("  • GIF : $(joinpath("figs", "enzyme_rf_training_evolution.gif"))")
    println("  • PDF : $(pdf_path)")
    println("  • PNG : $(png_path)")
end