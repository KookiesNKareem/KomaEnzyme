# ============================================================
# 2-D RF excitation design — physics/params/loss unchanged
# Gradient via Enzyme + custom GPU/CPU Bloch kernel
# No FiniteDiff, no includes: adaptive_gd! implemented below
# Option A: all kernel args are 1-D vectors (vec); no scalar GPU indexing
# ============================================================

using KomaMRICore, Suppressor                  # MRI simulation
using CairoMakie, LaTeXStrings, KomaMRIPlots   # Plotting
using JLD2                                     # Save results
using FileIO, FFTW                             # Images
import Images
using Random
using Statistics: mean, std
using PlotlyJS, PlotlyKaleido
using BenchmarkTools

# ---------------------- Gyromagnetic constants ----------------------
const γ64 = 2π * 42.57747892e6        # rad/(s·T) Float64 for k-space math
const γ32 = Float32(γ64)               # Float32 for GPU kernel

# Fast-mode toggles
const FAST_MODE = true
const ENABLE_DIAGNOSTICS = false
const ENABLE_GIF = false
const RECORD_HISTORY = false
const BB_EVERY = 2  # compute BB step every K iterations (use 1 to disable)
const DO_BENCHMARK = true  # enable BenchmarkTools timing

# ---------------------- Sequence setup (unchanged) ----------------------
sys = Scanner()
sys.Smax = 150.0
sys.Gmax = 100e-3
FOV = 1000e-3
N = 60
seq = PulseDesigner.spiral_base(FOV, N, sys; Nint=1)(0)

# Reverse gradients for RF excitation
x, y = 1, 2
seq.GR[x].A = reverse(seq.GR[x].A)
seq.GR[y].A = reverse(seq.GR[y].A)
seq.GR[x].rise, seq.GR[x].fall = seq.GR[x].fall, seq.GR[x].rise
seq.GR[y].rise, seq.GR[y].fall = seq.GR[y].fall, seq.GR[y].rise
seq.GR[x].delay = max(dur(seq.GR[x]), dur(seq.GR[y])) - dur(seq.GR[x])
seq.GR[y].delay = max(dur(seq.GR[x]), dur(seq.GR[y])) - dur(seq.GR[y])

# RF pulse
Nrf = 350
B1 = ComplexF32.(1f-6 .* ones(Float32, Nrf))
seq.RF[1] = RF(B1, dur(seq.ADC[1]) - seq.ADC[1].delay, 0.0,
               max(dur(seq.GR[x]), dur(seq.GR[y])) - dur(seq.ADC[1]))
Trf = seq.RF[1].T

# ADC for k-space plotting only
seq.ADC[1].N = Nrf
seq.ADC[1].T = Trf
seq.ADC[1].delay = seq.RF[1].delay
# Save sequence plot (works over SSH)
fig_seq = plot_seq(seq)
mkpath("Results")
PlotlyJS.savefig(fig_seq, "Results/sequence_plot.png")

# ---------------------- Excitation k-space (unchanged) ----------------------
seqd = discretize(seq)
mkx = γ64 * cumtrapz(reverse(seqd.Δt), reverse(seqd.Gx))[:]
mky = γ64 * cumtrapz(reverse(seqd.Δt), reverse(seqd.Gy))[:]
kspace = [mkx mky]
adc = reverse(seqd.ADC[1:end-1])

fig = Figure(size = (800, 800))
ax = Axis(fig[1, 1], xlabel = L"$k_x$ (mm$^{-1}$)", ylabel = L"$k_y$ (mm$^{-1}$)")
lines!(ax, mkx, mky)
scatter!(ax, mkx[adc], mky[adc], markersize=10, color=:red)
fig
seq.ADC[1].N = 0

# ---------------------- Phantom / Target (unchanged) ----------------------
Nspins_x = 80
Nspins_y = 80
FOV_sim = 200e-3
xs = range(-FOV_sim/2, FOV_sim/2, Nspins_x)
ys = range(-FOV_sim/2, FOV_sim/2, Nspins_y)
x = [x for (x, y) in Iterators.product(xs, ys)][:]
y = [y for (x, y) in Iterators.product(xs, ys)][:]
obj = Phantom(; x, y)

# GT params (for reference; Enzyme path doesn't call simulate)
sim_params_gt = KomaMRICore.default_sim_params()
sim_params_gt["Δt_rf"] = 1e-6
sim_params_gt["Δt"] = Inf
sim_params_gt["return_type"] = "state"
sim_params_gt["precision"] = "f64"
sim_params_gt["Nthreads"] = 32

# Optimization sim params (unchanged)
sim_params = copy(sim_params_gt)
sim_params["sim_method"] = KomaMRICore.Bloch()
# Use a finite RF timestep to avoid Inf in Δt; match control grid spacing
sim_params["Δt_rf"] = Trf / (Nrf - 1)
sim_params["Δt"] = Inf

# Params tuple (unchanged content & names)
mask = [sqrt(x^2 + y^2) .<= FOV_sim/2.2 for (x, y) in Iterators.product(xs, ys)][:]
params = (
    obj=obj,
    seq=seq,
    sys=sys,
    sim_params=sim_params,
    target_profile=nothing,  # Placeholder, set per image
    x0=copy(seq.RF[1].A),
    mag_x0=nothing,          # Placeholder, set per image
    xs=xs,
    ys=ys,
    Nspins_x=Nspins_x,
    Nspins_y=Nspins_y,
    t=collect(range(0, Trf, Nrf)),
    mask=mask
)

# ---------------------- Enzyme + KernelAbstractions path ----------------------
using KernelAbstractions
using KernelAbstractions: @kernel, @index, @atomic
import Enzyme
using CUDA
using Adapt
using LinearAlgebra: norm

# Backend
CUDA.device!(0) # Change to 0
const backend = CUDA.CUDABackend()
CUDA.limit!(CUDA.CU_LIMIT_MALLOC_HEAP_SIZE, 5*1024^3)
const GROUP_SIZE = 256
adapt_dev(x) = CUDA.adapt(CuArray, x)

# Discretize Koma timeline exactly per your sim_params
function _koma_timeline(seq, sim_params)
    seqd = KomaMRICore.discretize(seq; sampling_params=sim_params)
    Nt   = length(seqd.Δt)
    # Sanitize Δt: replace any non-finite entries with 0 to avoid sin/cos(Inf) -> NaN
    Δt_raw = Float32.(seqd.Δt[1:Nt])
    @inbounds for i in eachindex(Δt_raw)
        if !isfinite(Δt_raw[i])
            Δt_raw[i] = 0.0f0
        end
    end
    return (
        B1   = ComplexF32.(seqd.B1[1:Nt]),
        Gx   = Float32.(seqd.Gx[1:Nt]),
        Gy   = Float32.(seqd.Gy[1:Nt]),
        Gz   = Float32.(seqd.Gz[1:Nt]),
        Δt   = Δt_raw,
        Δf   = Float32.(seqd.Δf[1:Nt]),   # [rad/s]
        t    = Float32.(seqd.t[1:Nt]),
        Nt32 = Int32(Nt),
        rf_active_idx = findall(b1 -> abs(b1) > 1f-10, ComplexF32.(seqd.B1[1:Nt]))
    )
end

const TL = _koma_timeline(params.seq, params.sim_params)
const rf_idx = TL.rf_active_idx
const n_ctrl = length(params.x0)

# RF sign convention (will be calibrated to match Koma simulate)
S_B1R_SIGN = 1.0
S_B1I_SIGN = 1.0

# Control <-> timeline (linear; complex controls preserved)
const ctrl_pos = collect(range(0f0, 1f0, length=n_ctrl))
const tap_pos  = collect(range(0f0, 1f0, length=length(rf_idx)))

# Host RF scratch (preallocate & reuse to avoid per-call allocs)
const B1r_base_h = Float32.(real.(TL.B1))
const B1i_base_h = Float32.(imag.(TL.B1))
B1r_host = copy(B1r_base_h)
B1i_host = copy(B1i_base_h)

function map_x_to_timeline!(B1r::AbstractVector{<:Real}, B1i::AbstractVector{<:Real}, xctrl::AbstractVector{<:Complex})
    @assert length(xctrl) == n_ctrl
    copyto!(B1r, B1r_base_h)
    copyto!(B1i, B1i_base_h)
    j = 1
    @inbounds for (k, idx) in enumerate(rf_idx)
        t = tap_pos[k]
        while j < n_ctrl && ctrl_pos[j+1] < t
            j += 1
        end
        v = if j == n_ctrl
            xctrl[end]
        else
            α = (t - ctrl_pos[j]) / (ctrl_pos[j+1] - ctrl_pos[j] + 1e-20)
            (1 - α) * xctrl[j] + α * xctrl[j+1]
        end
        B1r[idx] = S_B1R_SIGN * real(v)
        B1i[idx] = S_B1I_SIGN * imag(v)
    end
    return nothing
end

# Adjoint scatter: transpose of the linear interpolation used in the forward map
function scatter_grad_to_x!(∇x::AbstractVector{ComplexF64},
                            dB1r_timeline::AbstractVector{<:Real},
                            dB1i_timeline::AbstractVector{<:Real})
    @assert length(dB1r_timeline) == length(TL.Δt) == length(dB1i_timeline)
    resize!(∇x, n_ctrl);  fill!(∇x, 0.0 + 0.0im)
    j = 1
    @inbounds for (k, idx) in enumerate(rf_idx)
        t = tap_pos[k]
        while j < n_ctrl && ctrl_pos[j+1] < t
            j += 1
        end
        wr = Float64(dB1r_timeline[idx])
        wi = Float64(dB1i_timeline[idx])
        if j == n_ctrl
            ∇x[end] += ComplexF64(wr, wi)
        else
            α = (t - ctrl_pos[j]) / (ctrl_pos[j+1] - ctrl_pos[j] + 1e-20)
            w0 = 1 - α
            w1 = α
            ∇x[j]   += ComplexF64(wr, wi) * w0
            ∇x[j+1] += ComplexF64(wr, wi) * w1
        end
    end
    return ∇x
end

# Spins (order matches target_profile and mask) — make sure all are VECTORS
xs32 = Float32.(params.xs); ys32 = Float32.(params.ys)
Nspins_x32 = Int(params.Nspins_x); Nspins_y32 = Int(params.Nspins_y)
const Nspins = Nspins_x32 * Nspins_y32
xgrid = Float32.([xx for (xx, _) in Iterators.product(xs32, ys32)])
ygrid = Float32.([yy for (_, yy) in Iterators.product(xs32, ys32)])
@assert length(xgrid) == length(params.mask)

# ---------------------- Option A: ALL device buffers are 1-D ----------------------
# State
M_xy  = adapt_dev(vec(zeros(Float32, 2Nspins)))
M_z   = adapt_dev(vec(ones(Float32, Nspins)))

# Spins/params
p_x   = adapt_dev(vec(Float32.(xgrid)))
p_y   = adapt_dev(vec(Float32.(ygrid)))
p_z   = adapt_dev(vec(zeros(Float32, Nspins)))
p_ΔBz = adapt_dev(vec(zeros(Float32, Nspins)))
p_T1  = adapt_dev(vec(fill(Float32(1e9), Nspins)))
p_T2  = adapt_dev(vec(fill(Float32(1e9), Nspins)))
p_ρ   = adapt_dev(vec(ones(Float32, Nspins)))

# Timeline
s_Gx  = adapt_dev(vec(TL.Gx))
s_Gy  = adapt_dev(vec(TL.Gy))
s_Gz  = adapt_dev(vec(TL.Gz))
s_Δt  = adapt_dev(vec(TL.Δt))
s_Δf  = adapt_dev(vec(TL.Δf))
# Ignore off-resonance during optimization for stability; set later if needed
fill!(s_Δf, 0.0f0)

# RF (we overwrite only rf_idx per call)
s_B1r = adapt_dev(vec(Float32.(real.(TL.B1))))
s_B1i = adapt_dev(vec(Float32.(imag.(TL.B1))))
# Keep immutable base timeline on device for fast resets
s_B1r_base = adapt_dev(vec(Float32.(real.(TL.B1))))
s_B1i_base = adapt_dev(vec(Float32.(imag.(TL.B1))))
# Device scratch buffers for reverse-mode and RF gradients
dM_xy = similar(M_xy)
dM_z  = similar(M_z)
∇B1r  = similar(s_B1r)
∇B1i  = similar(s_B1i)

# Targets & mask (host and device)
MASK_h = Float32.(params.mask)
mask_d_global = adapt_dev(Float32.(params.mask))

# Device accumulator for loss (single scalar)
acc_loss_d = adapt_dev(zeros(Float32, 1))

# Device grads in control space and pinned host mirrors
gx_d  = adapt_dev(zeros(Float32, n_ctrl))
gi_d  = adapt_dev(zeros(Float32, n_ctrl))
acc_loss_h = CUDA.pin(zeros(Float32, 1))
gx_h = CUDA.pin(zeros(Float32, n_ctrl))
gi_h = CUDA.pin(zeros(Float32, n_ctrl))
x_r_h = CUDA.pin(zeros(Float32, n_ctrl))
x_i_h = CUDA.pin(zeros(Float32, n_ctrl))

# Use Float32 gyromagnetic ratio inside kernels to avoid conversions
const γ_kernel32 = Float32(γ64)

# ---------------------- Bloch excitation kernel ----------------------
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
            Δt_raw  = s_Δt[s_idx]
            # Guard against non-finite Δt (e.g., Inf) to avoid sin/cos(NaN)
            Δt = isfinite(Δt_raw) ? Δt_raw : zero(T)
            df = s_Δf[s_idx]          # [rad/s]
            b1r = s_B1r[s_idx]; b1i = s_B1i[s_idx]

            Bz = (x*gx + y*gy + z*gz) + ΔBz - df / T(γ_kernel32)

            B  = sqrt(b1r*b1r + b1i*b1i + Bz*Bz)
            # Correct SU(2) parameterization: use θ/2 = (γ * |B| * Δt)/2
            φ  = T(0.5) * T(γ_kernel32) * B * Δt
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

# ---------------------- New small helper kernels (device-only seed and scatter) ----------------------
# Fused: seed dL/dM and accumulate loss in one pass
@kernel inbounds=true function seed_and_loss!(acc::AbstractVector{T},
                                             dM_xy::AbstractVector{T},
                                             M_xy::AbstractVector{T},
                                             T_r::AbstractVector{T},
                                             T_i::AbstractVector{T},
                                             mask::AbstractVector{T},
                                             invN::T) where {T<:AbstractFloat}
    i = Int(Tuple(@index(Global))[1])
    N = length(mask)
    if i <= N
        Mr = M_xy[i]
        Mi = M_xy[i + N]
        dr = Mr - T_r[i]
        di = Mi - T_i[i]
        w  = mask[i] * invN
        # loss
        @atomic acc[1] += w * (dr*dr + di*di)
        # gradient seed: 2*w*(M - T)
        scale = T(2) * w
        dM_xy[i]     = scale * dr
        dM_xy[i + N] = scale * di
    end
end

# Precompute mapping from active RF timeline idx -> control indices and weights (host)
j_lo_h = Vector{Int32}(undef, length(rf_idx))
j_hi_h = Vector{Int32}(undef, length(rf_idx))
w0_h   = Vector{Float32}(undef, length(rf_idx))
w1_h   = Vector{Float32}(undef, length(rf_idx))
# Also keep the active timeline indices as Int32 for device lookups
idx_rf_h = Vector{Int32}(undef, length(rf_idx))

let j = 1
    @inbounds for (k, idx) in enumerate(rf_idx)
        t = tap_pos[k]
        while j < n_ctrl && ctrl_pos[j+1] < t
            j += 1
        end
        if j == n_ctrl
            j_lo_h[k] = Int32(n_ctrl);  j_hi_h[k] = Int32(n_ctrl)
            w0_h[k] = 1f0; w1_h[k] = 0f0
        else
            α = Float32((t - ctrl_pos[j]) / (ctrl_pos[j+1] - ctrl_pos[j] + 1e-20))
            j_lo_h[k] = Int32(j)
            j_hi_h[k] = Int32(j+1)
            w0_h[k] = 1f0 - α
            w1_h[k] = α
        end
        idx_rf_h[k] = Int32(idx)
    end
end

# Upload mapping to device once
j_lo_d = adapt_dev(j_lo_h)
j_hi_d = adapt_dev(j_hi_h)
w0_d   = adapt_dev(w0_h)
w1_d   = adapt_dev(w1_h)
idx_rf_d = adapt_dev(idx_rf_h)

# Build CSR for gather: for each control j, list all k (active taps) that touch j with weights
counts = zeros(Int32, n_ctrl)
@inbounds for k in eachindex(j_lo_h)
    counts[Int(j_lo_h[k])] += 1
    counts[Int(j_hi_h[k])] += 1
end
csr_ptr_h = Vector{Int32}(undef, n_ctrl + 1)
csr_ptr_h[1] = Int32(1)
@inbounds for j in 1:n_ctrl
    csr_ptr_h[j+1] = csr_ptr_h[j] + counts[j]
end
nnz = Int(csr_ptr_h[end] - 1)
csr_idx_h = Vector{Int32}(undef, nnz)
csr_w_h   = Vector{Float32}(undef, nnz)
fill!(counts, 0)  # reuse as write cursor
@inbounds for k in eachindex(j_lo_h)
    j0 = Int(j_lo_h[k]); j1 = Int(j_hi_h[k])
    p0 = Int(csr_ptr_h[j0] + counts[j0]); counts[j0] += 1
    csr_idx_h[p0] = Int32(k); csr_w_h[p0] = w0_h[k]
    p1 = Int(csr_ptr_h[j1] + counts[j1]); counts[j1] += 1
    csr_idx_h[p1] = Int32(k); csr_w_h[p1] = w1_h[k]
end
csr_ptr_d = adapt_dev(csr_ptr_h)
csr_idx_d = adapt_dev(csr_idx_h)
csr_w_d   = adapt_dev(csr_w_h)

# ---------------------- Debug helpers ----------------------
function _nonfinite_count(v)
    c = 0
    @inbounds @simd for x in v
        if !isfinite(x)
            c += 1
        end
    end
    return c
end

function _debug_dump!(k::Integer, f::Real; tag::AbstractString="grad_and_loss_dev")
    br = Array(s_B1r); bi = Array(s_B1i)
    mxy = Array(M_xy)
    gx  = Array(gx_d); gi = Array(gi_d)
    gr  = Array(∇B1r); giB = Array(∇B1i)

    b1_max = maximum(@. sqrt(br*br + bi*bi))
    b1_nf  = _nonfinite_count(br) + _nonfinite_count(bi)

    mr = @view mxy[1:Nspins]
    mi = @view mxy[Nspins+1:2Nspins]
    mxy_max  = maximum(@. sqrt(mr*mr + mi*mi))
    mxy_mean = mean(@. sqrt(mr*mr + mi*mi))
    mxy_nf   = _nonfinite_count(mxy)

    # Gradients
    g_norm_dev = sqrt(Float64(norm(gx))^2 + Float64(norm(gi))^2)
    grad_b1_norm = sqrt(sum(@. Float64(gr*gr)) + sum(@. Float64(giB*giB)))

    @info "DBG[$k] $tag" loss=f b1_max=b1_max b1_nonfinite=b1_nf mxy_max=mxy_max mxy_mean=mxy_mean mxy_nonfinite=mxy_nf g_ctrl_norm=g_norm_dev dB1_norm=grad_b1_norm
    if b1_nf > 0 || mxy_nf > 0
        @warn "Non-finite values detected in device buffers" b1_nonfinite=b1_nf mxy_nonfinite=mxy_nf
    end
    return nothing
end

# CSR-style gather kernel: compute control-space grads without atomics
@kernel inbounds=true function gather_ctrl_grads!(gx::AbstractVector{T},
                                                  gi::AbstractVector{T},
                                                  dB1r::AbstractVector{T},
                                                  dB1i::AbstractVector{T},
                                                  idx_rf::AbstractVector{Int32},
                                                  ptr::AbstractVector{Int32},
                                                  csr_idx::AbstractVector{Int32},
                                                  csr_w::AbstractVector{T}) where {T<:AbstractFloat}
    j = Int(Tuple(@index(Global))[1])
    if j <= length(gx)
        s_r = zero(T); s_i = zero(T)
        p0 = Int(ptr[j]); p1 = Int(ptr[j+1]) - 1
        @inbounds for p in p0:p1
            k   = Int(csr_idx[p])
            tid = Int(idx_rf[k])
            w   = csr_w[p]
            s_r += w * dB1r[tid]
            s_i += w * dB1i[tid]
        end
        gx[j] = s_r
        gi[j] = s_i
    end
end

# Map controls -> timeline on device (per active RF index)
@kernel inbounds=true function map_ctrl_to_timeline!(out_B1r::AbstractVector{T},
                                                     out_B1i::AbstractVector{T},
                                                     x_r::AbstractVector{T},
                                                     x_i::AbstractVector{T},
                                                     idx_rf::AbstractVector{Int32},
                                                     jlo::AbstractVector{Int32},
                                                     jhi::AbstractVector{Int32},
                                                     w0::AbstractVector{T},
                                                     w1::AbstractVector{T},
                                                     sgn_r::T,
                                                     sgn_i::T) where {T<:AbstractFloat}
    k = Int(Tuple(@index(Global))[1])
    if k <= length(idx_rf)
        idx = Int(idx_rf[k])
        j0 = Int(jlo[k]); j1 = Int(jhi[k])
        out_B1r[idx] = sgn_r * (w0[k] * x_r[j0] + w1[k] * x_r[j1])
        out_B1i[idx] = sgn_i * (w0[k] * x_i[j0] + w1[k] * x_i[j1])
    end
end

# Per-try update: x_try = x - λ * g (device)
@kernel inbounds=true function axpy_neg!(x_try_r::AbstractVector{T},
                                         x_try_i::AbstractVector{T},
                                         x_r::AbstractVector{T},
                                         x_i::AbstractVector{T},
                                         gx::AbstractVector{T},
                                         gi::AbstractVector{T},
                                         λ::T) where {T<:AbstractFloat}
    j = Int(Tuple(@index(Global))[1])
    if j <= length(x_r)
        x_try_r[j] = x_r[j] - λ * gx[j]
        x_try_i[j] = x_i[j] - λ * gi[j]
    end
end

# Kernel launcher
function _launch_excitation!(
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
    _launch_excitation!(
        M_xy, M_z,
        p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
        s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1r, s_B1i,
        Int32(Nspins), TL.Nt32, backend)
    return nothing
end

# ---------------------- Forward / Loss / Grad (device-optimized) ----------------------
# Forward returns an object with .xy only when explicitly needed (plotting)
function forward(x, params)
    copyto!(B1r_host, B1r_base_h); copyto!(B1i_host, B1i_base_h)
    map_x_to_timeline!(B1r_host, B1i_host, x)
    copyto!(s_B1r, vec(B1r_host)); copyto!(s_B1i, vec(B1i_host))
    fill!(M_xy, 0.0f0); fill!(M_z, 1.0f0)
    excite_only!(M_xy, M_z,
        p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
        s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1r, s_B1i,
        backend)
    KernelAbstractions.synchronize(backend)
    Mr = Array(view(M_xy, 1:Nspins))
    Mi = Array(view(M_xy, Nspins+1:2Nspins))
    return (; xy = ComplexF32.(Mr, Mi))
end

function loss_only!(x, params)
    # Fallback host path retained for API compatibility
    copyto!(B1r_host, B1r_base_h); copyto!(B1i_host, B1i_base_h)
    map_x_to_timeline!(B1r_host, B1i_host, x)
    copyto!(s_B1r, vec(B1r_host)); copyto!(s_B1i, vec(B1i_host))
    fill!(M_xy, 0.0f0); fill!(M_z, 1.0f0)
    excite_only!(M_xy, M_z,
        p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
        s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1r, s_B1i,
        backend)
    fill!(acc_loss_d, 0.0f0)
    seed_and_loss!(backend)(acc_loss_d, dM_xy, M_xy, params.T_xy_r_d, params.T_xy_i_d, params.mask_d, params.INVN;
                            ndrange=Nspins, workgroupsize=GROUP_SIZE)
    CUDA.@sync CUDA.copyto!(acc_loss_h, acc_loss_d)
    return acc_loss_h[1]
end

# Device-optimized loss-only using on-GPU control->timeline mapping
function loss_only_dev!(x_r_d, x_i_d, params)
    # No need to reset entire timeline buffers each iteration; active taps are fully overwritten below
    map_ctrl_to_timeline!(backend)(s_B1r, s_B1i, x_r_d, x_i_d, idx_rf_d, j_lo_d, j_hi_d, w0_d, w1_d,
                                   Float32(S_B1R_SIGN), Float32(S_B1I_SIGN);
                                   ndrange=length(rf_idx), workgroupsize=GROUP_SIZE)
    fill!(M_xy, 0.0f0); fill!(M_z, 1.0f0)
    excite_only!(M_xy, M_z,
        p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
        s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1r, s_B1i,
        backend)
    fill!(acc_loss_d, 0.0f0)
    seed_and_loss!(backend)(acc_loss_d, dM_xy, M_xy, params.T_xy_r_d, params.T_xy_i_d, params.mask_d, params.INVN;
                            ndrange=Nspins, workgroupsize=GROUP_SIZE)
    CUDA.@sync CUDA.copyto!(acc_loss_h, acc_loss_d)
    return acc_loss_h[1]
end

function grad_and_loss!(∇loss::AbstractVector{ComplexF32}, x::AbstractVector{<:Complex}, params)
    # Host fallback retained; prefer grad_and_loss_dev! in hot loops
    copyto!(B1r_host, B1r_base_h); copyto!(B1i_host, B1i_base_h)
    map_x_to_timeline!(B1r_host, B1i_host, x)
    copyto!(s_B1r, vec(B1r_host)); copyto!(s_B1i, vec(B1i_host))
    fill!(M_xy, 0.0f0); fill!(M_z, 1.0f0)
    excite_only!(M_xy, M_z,
        p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
        s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1r, s_B1i,
        backend)
    fill!(acc_loss_d, 0.0f0)
    fill!(dM_z, 0.0f0)
    fill!(∇B1r, 0.0f0); fill!(∇B1i, 0.0f0)
    seed_and_loss!(backend)(acc_loss_d, dM_xy, M_xy, params.T_xy_r_d, params.T_xy_i_d, params.mask_d, params.INVN;
                            ndrange=Nspins, workgroupsize=GROUP_SIZE)
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
    fill!(gx_d, 0.0f0); fill!(gi_d, 0.0f0)
    gather_ctrl_grads!(backend)(gx_d, gi_d, ∇B1r, ∇B1i, idx_rf_d, csr_ptr_d, csr_idx_d, csr_w_d;
                                ndrange=n_ctrl, workgroupsize=GROUP_SIZE)
    CUDA.@sync begin
        CUDA.copyto!(gx_h, gx_d)
        CUDA.copyto!(gi_h, gi_d)
        CUDA.copyto!(acc_loss_h, acc_loss_d)
    end
    @inbounds for j in 1:n_ctrl
        ∇loss[j] = ComplexF32(gx_h[j], gi_h[j])
    end
    return acc_loss_h[1]
end

# Device-optimized gradient+loss using on-GPU mapping and CSR gather
function grad_and_loss_dev!(∇loss::AbstractVector{ComplexF32}, x_r_d, x_i_d, params)
    # No need to reset entire timeline buffers each iteration; active taps are fully overwritten below
    map_ctrl_to_timeline!(backend)(s_B1r, s_B1i, x_r_d, x_i_d, idx_rf_d, j_lo_d, j_hi_d, w0_d, w1_d,
                                   Float32(S_B1R_SIGN), Float32(S_B1I_SIGN);
                                   ndrange=length(rf_idx), workgroupsize=GROUP_SIZE)
    fill!(M_xy, 0.0f0); fill!(M_z, 1.0f0)
    excite_only!(M_xy, M_z,
        p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
        s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1r, s_B1i,
        backend)
    fill!(acc_loss_d, 0.0f0)
    fill!(dM_z, 0.0f0)
    fill!(∇B1r, 0.0f0); fill!(∇B1i, 0.0f0)
    seed_and_loss!(backend)(acc_loss_d, dM_xy, M_xy, params.T_xy_r_d, params.T_xy_i_d, params.mask_d, params.INVN;
                            ndrange=Nspins, workgroupsize=GROUP_SIZE)
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
    fill!(gx_d, 0.0f0); fill!(gi_d, 0.0f0)
    gather_ctrl_grads!(backend)(gx_d, gi_d, ∇B1r, ∇B1i, idx_rf_d, csr_ptr_d, csr_idx_d, csr_w_d;
                                ndrange=n_ctrl, workgroupsize=GROUP_SIZE)
    CUDA.@sync begin
        CUDA.copyto!(gx_h, gx_d)
        CUDA.copyto!(gi_h, gi_d)
        CUDA.copyto!(acc_loss_h, acc_loss_d)
    end
    @inbounds for j in 1:n_ctrl
        ∇loss[j] = ComplexF32(gx_h[j], gi_h[j])
    end
    return acc_loss_h[1]
end

# New: device-only gradient+loss that avoids per-iteration host gradient copies
const DEBUG_NAN = false

function grad_and_loss_dev_device!(x_r_d, x_i_d, params)
    # Overwrite only active RF taps from controls
    map_ctrl_to_timeline!(backend)(s_B1r, s_B1i, x_r_d, x_i_d, idx_rf_d, j_lo_d, j_hi_d, w0_d, w1_d,
                                   Float32(S_B1R_SIGN), Float32(S_B1I_SIGN);
                                   ndrange=length(rf_idx), workgroupsize=GROUP_SIZE)
    # Forward (reset state each call)
    fill!(M_xy, 0.0f0); fill!(M_z, 1.0f0)
    excite_only!(M_xy, M_z,
        p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
        s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1r, s_B1i,
        backend)

    if DEBUG_NAN
        mxy_nf = _nonfinite_count(Array(M_xy))
        if mxy_nf > 0
            _debug_dump!(0, NaN; tag="post_forward")
        end
    end

    # Seed loss and initialize adjoint buffers
    fill!(acc_loss_d, 0.0f0)
    fill!(dM_z, 0.0f0)
    fill!(∇B1r, 0.0f0); fill!(∇B1i, 0.0f0)
    seed_and_loss!(backend)(acc_loss_d, dM_xy, M_xy, params.T_xy_r_d, params.T_xy_i_d, params.mask_d, params.INVN;
                            ndrange=Nspins, workgroupsize=GROUP_SIZE)

    if DEBUG_NAN
        dseed_nf = _nonfinite_count(Array(dM_xy))
        CUDA.@sync CUDA.copyto!(acc_loss_h, acc_loss_d)
        if dseed_nf > 0 || !isfinite(acc_loss_h[1])
            _debug_dump!(0, acc_loss_h[1]; tag="post_seed")
        end
    end

    # Reverse via Enzyme
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

    if DEBUG_NAN
        nb1 = _nonfinite_count(Array(∇B1r)) + _nonfinite_count(Array(∇B1i))
        if nb1 > 0
            _debug_dump!(0, NaN; tag="post_reverse")
        end
    end

    # Gather to control space (on device)
    fill!(gx_d, 0.0f0); fill!(gi_d, 0.0f0)
    gather_ctrl_grads!(backend)(gx_d, gi_d, ∇B1r, ∇B1i, idx_rf_d, csr_ptr_d, csr_idx_d, csr_w_d;
                                ndrange=n_ctrl, workgroupsize=GROUP_SIZE)
    # Return scalar loss
    CUDA.@sync CUDA.copyto!(acc_loss_h, acc_loss_d)
    return acc_loss_h[1]
end

function calc_grad!(∇loss::AbstractVector{ComplexF32}, x::AbstractVector{<:Complex}, params)
    grad_and_loss!(∇loss, x, params)
    return ∇loss
end

# ---------------------- Gradient diagnostic (directional derivative test) ----------------------
function gradient_directional_check(x::AbstractVector{<:Complex}, params;
                                    ε::Real = 1f-6, ntests::Int = 3, seed::Int = 0)
    rng = MersenneTwister(seed)
    g = similar(x); fill!(g, zero(eltype(x)))
    calc_grad!(g, x, params)
    # Use conjugate in inner product so df_adj is real; compare to symmetric FD
    rel_errs = Float64[]
    for _ in 1:ntests
        v = randn(rng, ComplexF32, length(x))
        nv = norm(v)
        nv == 0 && (v[1] = ComplexF32(1, 0); nv = 1.0)
        v ./= nv
        f_plus  = loss_only!(x .+ (eltype(x))(ε) .* v, params)
        f_minus = loss_only!(x .- (eltype(x))(ε) .* v, params)
        df_fd = Float64((f_plus - f_minus) / (2*(eltype(x))(ε)))
        df_adj = Float64(real(sum(conj.(g) .* v)))
        push!(rel_errs, abs(df_fd - df_adj) / max(1e-12, abs(df_fd)))
    end
    return (; mean_rel_err = mean(rel_errs), max_rel_err = maximum(rel_errs))
end

# ---------------------- Minimal optimizer: adaptive_gd! ----------------------
"""
adaptive_gd!(x0, params, Niters, λ0; plateau_tol=1e-12, plateau_patience=5)

Simple adaptive gradient descent that:
- tries step x - λ*g; accepts if loss decreases, else halves λ (backtracking, up to 8 tries),
- on success, slightly grows λ (×1.05),
- records loss_history, x_history, and simple stats.
Stops early if the loss change between consecutive iterations stays below plateau_tol
for plateau_patience consecutive iterations.
Returns: x, loss_history, x_history, stats, performed_iterations
"""
"""
Malitsky-Mishchenko adaptive gradient descent optimizer for 2D RF design.

Parameters:
    λ_i = min{√(1 + θ_{i-1}) * λ_{i-1}, ||x^i - x^{i-1}|| / (2||∇f(x^i) - ∇f(x^{i-1})||)}
    x^{i+1} = x^i - λ_i * ∇f(x^i)
    θ_i = λ_i / λ_{i-1}

Reference: Malitsky & Mishchenko (2020), "Adaptive gradient descent without descent"
"""
function adaptive_gd!(x0, params, Niters::Integer, λ0; plateau_tol::Float64=1e-12, plateau_patience::Int=5, log_every::Int=0, debug::Bool=false, debug_every::Int=10)
    x = copy(x0)
    Nctrl = length(x0)
    loss_history = zeros(Float64, Niters)
    x_history = zeros(eltype(x0), Nctrl, Niters)
    stats = Dict{Symbol,Any}()
    Tλ = eltype(real(x0))
    λ_prev = Tλ(λ0)
    θ_prev = Tλ(Inf)

    # Device controls
    x_r_d = adapt_dev(Float32.(real.(x0)))
    x_i_d = adapt_dev(Float32.(imag.(x0)))
    x_try_r_d = similar(x_r_d); x_try_i_d = similar(x_i_d)

    # Previous (device) for step-size computation
    gx_prev_d = similar(gx_d); gi_prev_d = similar(gi_d)
    fill!(gx_prev_d, 0.0f0); fill!(gi_prev_d, 0.0f0)
    x_prev_r_d = similar(x_r_d); x_prev_i_d = similar(x_i_d)
    CUDA.copyto!(x_prev_r_d, x_r_d); CUDA.copyto!(x_prev_i_d, x_i_d)

    # Track how many iterations actually ran and if early stop happened
    performed_iterations = 0
    early_stopped = false

    for k in 1:Niters
        performed_iterations = k
        # Evaluate f(x^k) and ∇f(x^k) (kept on device)
        f_k = grad_and_loss_dev_device!(x_r_d, x_i_d, params)
        loss_history[k] = f_k

        # Norms via device reductions
        gnormk = sqrt(Float64(CUDA.sum(gx_d .* gx_d .+ gi_d .* gi_d)))
        num = sqrt(Float64(CUDA.sum((x_r_d .- x_prev_r_d) .* (x_r_d .- x_prev_r_d) .+
                                     (x_i_d .- x_prev_i_d) .* (x_i_d .- x_prev_i_d))))
        den = sqrt(Float64(CUDA.sum((gx_d .- gx_prev_d) .* (gx_d .- gx_prev_d) .+
                                     (gi_d .- gi_prev_d) .* (gi_d .- gi_prev_d))))

        # Immediate NaN/Inf guard with debug dump
        if !isfinite(f_k) || !isfinite(gnormk)
            _debug_dump!(k, f_k; tag="nan_guard")
            @warn "Non-finite loss/gradient encountered; stopping" iter=k loss=f_k gnorm=gnormk
            early_stopped = true
            break
        end

        if log_every > 0 && (k % log_every == 0 || k == 1)
            @info "Iter $k: loss = $f_k, λ_prev = $λ_prev, ||g|| = $gnormk"
        end
        if debug && (k % debug_every == 0 || k == 1)
            _debug_dump!(k, f_k; tag="iter")
        end

        ratio = den > 0 ? num / (2 * den) : Inf
        grow  = sqrt(1 + Float64(θ_prev)) * Float64(λ_prev)
        λ_k = Tλ(min(grow, ratio))
        if !(isfinite(λ_k)) || λ_k <= zero(Tλ)
            λ_k = λ_prev
        end

        # Carry previous for next iteration
        CUDA.copyto!(gx_prev_d, gx_d); CUDA.copyto!(gi_prev_d, gi_d)
        CUDA.copyto!(x_prev_r_d, x_r_d); CUDA.copyto!(x_prev_i_d, x_i_d)

        # Update: x^{k+1} = x^k − λ_k ∇f(x^k)
        axpy_neg!(backend)(x_try_r_d, x_try_i_d, x_r_d, x_i_d, gx_d, gi_d, Float32(λ_k); ndrange=Nctrl, workgroupsize=GROUP_SIZE)
        copyto!(x_r_d, x_try_r_d); copyto!(x_i_d, x_try_i_d)

        # Host mirror (for history/plots)
        CUDA.@sync begin
            CUDA.copyto!(x_r_h, x_r_d)
            CUDA.copyto!(x_i_h, x_i_d)
        end
        @inbounds for j in 1:Nctrl
            x[j] = ComplexF32(x_r_h[j], x_i_h[j])
        end
        x_history[:, k] .= x

        # θ_k and carry
        θ_prev = λ_k / λ_prev
        λ_prev = λ_k
    end

    stats[:final_step] = λ_prev
    stats[:theta_last] = θ_prev
    stats[:early_stopped] = early_stopped
    return x, loss_history, x_history, stats, performed_iterations
end

# Benchmark wrapper: run adaptive_gd! for Niters and λ0, then sync GPU
function bench_train_loop!(params, Niters::Integer, λ0)
    x0 = zeros(eltype(params.seq.RF[1].A), length(params.x0))
    _ = adaptive_gd!(x0, params, Niters, λ0)
    CUDA.synchronize()
    return nothing
end

# ---------------------- Animation hook (minimal) ----------------------
function update_x_in_fig!(x_ctrl)
    st = forward(x_ctrl, params)
    mag = abs.(reshape(st.xy, params.Nspins_x, params.Nspins_y))
    contour!(axp, params.xs * 1e2, params.ys * 1e2, mag; levels=5, color=:blue, linewidth=1)
    return nothing
end

# Reference forward using KomaMRICore.simulate for cross-checks
function forward_koma(x, params)
    seq_aux = copy(params.seq)
    seq_aux.RF[1].A .= x
    mag = @suppress simulate(params.obj, seq_aux, params.sys; params.sim_params)
    return mag.xy
end

# Calibrate RF sign convention to match Koma simulate
function calibrate_rf_convention!(params)
    global S_B1R_SIGN, S_B1I_SIGN
    a = 1f-6
    # Two probes: real-only and imag-only impulse at mid control
    mid = max(1, Int(clamp(round(n_ctrl/2), 1, n_ctrl)))
    xt_re = zeros(ComplexF32, n_ctrl); xt_re[mid] = ComplexF32(a, 0)
    xt_im = zeros(ComplexF32, n_ctrl); xt_im[mid] = ComplexF32(0, a)
    combos = [(1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)]
    best_err = Inf
    best = (1.0, 1.0)
    for (sr, si) in combos
        S_B1R_SIGN = sr; S_B1I_SIGN = si
        yk_re = forward(xt_re, params).xy
        yk_im = forward(xt_im, params).xy
        ys_re = forward_koma(xt_re, params)
        ys_im = forward_koma(xt_im, params)
        err = norm(yk_re - ys_re) / max(1e-14, norm(ys_re)) +
              norm(yk_im - ys_im) / max(1e-14, norm(ys_im))
        if err < best_err
            best_err = err
            best = (sr, si)
        end
    end
    S_B1R_SIGN, S_B1I_SIGN = best
    @info "Calibrated RF signs to match Koma" S_B1R_SIGN S_B1I_SIGN best_err
    return nothing
end

# Per-target GPU warmup to JIT-compile kernels and Enzyme path
function _gpu_warmup!(params)
    xw = zeros(eltype(params.seq.RF[1].A), n_ctrl)
    gw = similar(xw)
    try
        grad_and_loss!(gw, xw, params)
    catch
        # ignore first-call issues
    end
    KernelAbstractions.synchronize(backend)
    return nothing
end

# ---------------------- Run optimization for all target images ----------------------
image_paths = [
    "target_images/stanford_logo.png",
    "target_images/julia_logo.png",
    "target_images/circle.png",
]

for img_path in image_paths
    # ----- Build target -----
    img = load(img_path)
    img_bw = reverse(getproperty.(img', :b) .* 1.0, dims=2)
    img_bw .= img_bw ./ maximum(img_bw)

    cx, cy = size(img_bw) .÷ 2 .+ 1
    radius_px = 20
    ft_mask = [(sqrt((i - cx)^2 + (j - cy)^2) <= radius_px) *
               exp(-π * ((i - cx)^2 + (j - cy)^2) / (2 * radius_px^2))
              for i in 1:size(img_bw, 1), j in 1:size(img_bw, 2)]
    img_bw_lowpass = abs.(fftshift(ifft(fftshift(fft(fftshift(img_bw))) .* ft_mask)))

    # Resample to our grid and set target
    img_resized = Images.imresize(img_bw_lowpass, (Nspins_x, Nspins_y))
    target_profile = complex.(zeros(Float32, length(img_resized)), 0.5f0 .* Float32.(img_resized[:]))
    mag_target_2d = reshape(target_profile, Nspins_x, Nspins_y)

    # ----- Device target/mask for this target -----
    T_xy_r_d = adapt_dev(Float32.(real.(target_profile)))
    T_xy_i_d = adapt_dev(Float32.(imag.(target_profile)))
    INVN = inv(Float32(Nspins))

    # ----- Params for this target -----
    params_i = (
        obj=obj,
        seq=seq,
        sys=sys,
        sim_params=sim_params,
        target_profile=target_profile,
        x0=copy(seq.RF[1].A),
        mag_x0=mag_target_2d,
        xs=xs,
        ys=ys,
        Nspins_x=Nspins_x,
        Nspins_y=Nspins_y,
        t=collect(range(0, Trf, Nrf)),
        mask=mask,
        # new device helpers
        T_xy_r_d=T_xy_r_d,
        T_xy_i_d=T_xy_i_d,
        mask_d=mask_d_global,
        INVN=INVN,
    )

    # ----- Diagnostics -----
    if ENABLE_DIAGNOSTICS
        let res = gradient_directional_check(zeros(eltype(seq.RF[1].A), Nrf), params_i; ε=1e-6, ntests=2, seed=42)
            @info "[$(basename(img_path))] Gradient directional check at x0" res
        end
        let m_kernel = forward(zeros(eltype(seq.RF[1].A), Nrf), params_i).xy,
            m_koma = forward_koma(zeros(eltype(seq.RF[1].A), Nrf), params_i),
            rel = norm(m_kernel - m_koma) / max(1e-12, norm(m_koma))
            @info "[$(basename(img_path))] Forward model consistency (kernel vs simulate) at x0" rel
        end
    end

    # Calibrate RF sign convention for this run
    calibrate_rf_convention!(params_i)

    # Warm up GPU kernels + Enzyme once per target
    _gpu_warmup!(params_i)

    # ----- Optimization -----
    x0 = zeros(eltype(seq.RF[1].A), Nrf)
    Niters = 20
    λ0 = 2f-8

    # Benchmark the training loop (3 samples, evals=1)
    if DO_BENCHMARK
        bt = @benchmark bench_train_loop!(p, nst, eta) setup=(p=$params_i; nst=$Niters; eta=$λ0) evals=1 samples=3
        mean_s = mean(bt).time * 1e-9
        std_s  = std(bt).time  * 1e-9
        @info "[$(basename(img_path))] Optimization loop time — mean/std" mean_s=round(mean_s, digits=4) std_s=round(std_s, digits=4) samples=length(bt.times)
    end

    @info "[$(basename(img_path))] Starting optimization..." λ0 Niters
    start_time = time()
    x_opt, loss_history, x_history, stats, performed_iterations = adaptive_gd!(x0, params_i, Niters, λ0)
    end_time = time()
    @info end_time - start_time "seconds elapsed."
    @info "[$(basename(img_path))] Optimization complete."
    seq.RF[1].A .= x_opt

    # ----- Output directory -----
    stem = splitext(basename(img_path))[1]
    outdir = joinpath("Results", stem)
    mkpath(outdir)

    # ----- RF time-course figure -----
    let t_ms = params_i.t .* 1e3
        fig_rf = Figure(size = (1000, 350))
        axrf = Axis(fig_rf[1, 1], xlabel = "Time (ms)", ylabel = "B1 (µT)", title = "Optimized RF (real & imag)")
        lines!(axrf, t_ms, real(x_opt) .* 1e6, color = :blue, label = "Real")
        lines!(axrf, t_ms, imag(x_opt) .* 1e6, color = :red, label = "Imag")
        axislegend(axrf, position = :rt)
        save(joinpath(outdir, "RF_2D_optimized_RF.png"), fig_rf, px_per_unit = 4)
    end

    # ----- Target vs Achieved figure -----
    fig_prof = Figure(size = (1200, 800))
    axp = Axis(fig_prof[1, 1], xlabel = L"$x$ (cm)", ylabel = L"$y$ (cm)", title = "Target |Mₓᵧ|")
    hm_t = heatmap!(axp, xs * 1e2, ys * 1e2, abs.(mag_target_2d); colormap = :grays)
    Colorbar(fig_prof[1, 2], hm_t, ticks = [0.0, 0.25, 0.5, 0.75, 1.0])

    achieved = abs.(reshape(forward(x_opt, params_i).xy, Nspins_x, Nspins_y))
    axp2 = Axis(fig_prof[1, 3], xlabel = L"$x$ (cm)", ylabel = L"$y$ (cm)", title = "Achieved |Mₓᵧ|")
    hm_a = heatmap!(axp2, xs * 1e2, ys * 1e2, achieved; colormap = :grays)
    Colorbar(fig_prof[1, 4], hm_a, ticks = [0.0, 0.25, 0.5, 1.0])

    # Layout adjustments
    colsize!(fig_prof.layout, 1, Auto(1))
    colsize!(fig_prof.layout, 2, Fixed(18))
    colsize!(fig_prof.layout, 3, Auto(1))
    colsize!(fig_prof.layout, 4, Fixed(18))
    colgap!(fig_prof.layout, 8)

    # Match color scales between target and achieved
    cl_hi = max(maximum(abs.(mag_target_2d)), maximum(achieved))
    hm_t.colorrange[] = (0.0, cl_hi)
    hm_a.colorrange[] = (0.0, cl_hi)

    save(joinpath(outdir, "RF_2D_image_profile.png"), fig_prof, px_per_unit=4)

    # Optional: animation per target
    if ENABLE_GIF
        CairoMakie.record(fig_prof, joinpath(outdir, "RF_2D_image_profile.gif"), 1:performed_iterations; framerate = performed_iterations) do i
            st = forward(x_history[:, i], params_i)
            mag = abs.(reshape(st.xy, Nspins_x, Nspins_y))
            contour!(axp, xs * 1e2, ys * 1e2, mag; levels=5, color=:blue, linewidth=1)
        end
    end

    # ----- Save ONLY the optimized RF (real/imag) -----
    B1_r = real.(seq.RF[1].A)
    B1_i = imag.(seq.RF[1].A)
    @save joinpath(outdir, "RF_2D_image_results.jld2") B1_r B1_i
end