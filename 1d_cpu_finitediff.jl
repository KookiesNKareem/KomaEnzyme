# RF slice design — CPU-ONLY (RF window aligned to Gx flat-top)
using KomaMRICore, Suppressor
using Random: seed!
using KernelAbstractions
using KernelAbstractions: @kernel, @index, CPU
# import Enzyme                      # ← removed
using FiniteDiff                    # ← added
using Statistics: mean, std
using LinearAlgebra: norm
using Plots
using BenchmarkTools
using CUDA

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
sim_params["Δt_rf"]       = Trf / (2*(Nrf - 1))
sim_params["Δt"]          = 1f-3
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

const backend = CUDA.CUDABackend()

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

# ---------------- Forward/Loss ----------------
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

# A scalar loss wrapper for FiniteDiff
_loss_only(x::Vector{Float32}) = first(forward!(x))

# ---------------- FiniteDiff gradient (replaces Enzyme) ----------------
function loss_and_grad!(∇x::Vector{Float32}, x::Vector{Float32})
    # in-place FD gradient; tune absstep/relstep if needed
    FiniteDiff.finite_difference_gradient!(∇x, _loss_only, x; absstep=1f-7)
    return _loss_only(x)
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

# ---------------- Plotting (RF & Gx on same graph over full t) ----------------
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

    # --- Combined plot: left y = B1_r (T), right y = Gx (T/m), x = full timeline ---
    p_comb = plot(
        t_center, B1r_host; label="Re(B1) (T)", lw=2,
        xlabel="t (s)", ylabel="B1_r (T)", title="RF & Gx (full timeline, shared x)"
    )
    # second axis for Gx
    plot!(twinx(p_comb), t_center, TL.Gx; label="Gx (T/m)", lw=2, ylabel="Gx (T/m)")

    # visual guides
    vline!(p_comb, [t_center[ACT_IDX[1]], t_center[ACT_IDX[end]]]; ls=:dash, label="RF window")
    plot!(twinx(p_comb), [t_center[i_up], t_center[i_dn]], [NaN, NaN]; label="")  # keep legends tidy
    vline!(p_comb, [t_center[i_up], t_center[i_dn]]; ls=:dot, label="Gx flat-top")

    # RF imaginary (separate, same full x)
    p_i = plot(title="RF imag (timeline)", xlabel="t (s)", ylabel="B1_i (T)")
    plot!(p_i, t_center, B1i_host; label="Im(B1)", lw=2)

    # Profile
    p_prof = plot(title="|M_xy|(x)", xlabel="x (m)", ylabel="magnitude")
    plot!(p_prof, xline, TARGET_MAG; label="Target |M|", lw=3, ls=:dash)
    plot!(p_prof, xline, mag;        label="Achieved", lw=2)

    plot(p_comb, p_i, p_prof; layout=(3,1), size=(950,1000))
end

# ---------------- Train & plot ----------------
x  = zeros(Float32, Nrf)
∇x = similar(x)

function _cpu_warmup!()
    try; _ = loss_and_grad!(similar(x), x); catch; end
    KernelAbstractions.synchronize(backend)
end
_cpu_warmup!()

nsteps  = 20
η_base  = 5f-10
rf_clip = 2f-5

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

bt = @benchmark train_loop_mm!(xloc, nst, clip; λ_init=$λ_init) setup=(xloc = zeros(Float32, n_ctrl); nst = $nsteps; clip = $rf_clip; λ_init = 1f-9) evals=1 samples=20 seconds=30000

mean_s = mean(bt).time * 1e-9
std_s  = std(bt).time  * 1e-9
@info "Optimization loop time — mean/std" mean_s=round(mean_s, digits=4) std_s=round(std_s, digits=4) samples=length(bt.times)

# plt = plot_rf_and_profile(x)
# savefig(plt, "profile_and_rf.png")
# @info "Saved plot to profile_and_rf.png"