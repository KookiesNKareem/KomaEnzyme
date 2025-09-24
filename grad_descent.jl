############################
# Deps & constants
############################
using StaticArrays, LinearAlgebra, Random, Statistics
using Plots
using CUDA
import Enzyme
using Printf

# --- Sim constants ---
const TOTAL_TIME = 4.0f-3
const DT         = 8.0f-5
const NSTEPS     = Int(cld(TOTAL_TIME, DT))
const NUM_TIME_SEGMENTS = 50
const INV_SEG_DUR = NUM_TIME_SEGMENTS / TOTAL_TIME

const MAX_GZ   = 10f-3
const MAX_SLEW = 70f0
const RF_BOUND = 12f-6

const GAMMA  = 2f0π * 42.58f6     # rad/s/T (Hz*2π scaled); we’ll convert per-float
const PARAMS = (γ=GAMMA, T1=1f0, T2=1f0, M0=1f0)

# --- Spatial config ---
const DELTA   = 4.0f-3
const Z_FOV   = 14.0f-3
const NZ      = 121
const Z_BASE  = Float32.(range(-Z_FOV/2, Z_FOV/2; length=NZ))

const TRANSITION_FRAC = 0f0
const DELTA_CORE = DELTA * (1 - TRANSITION_FRAC)

# --- GPU config ---
const TPB = 256  # threads per block

Random.seed!(42)

############################
# Sequence & masks
############################
@inline function gz_at(t; Gmax=MAX_GZ, slew=MAX_SLEW)
    ramp_time = min(Gmax/slew, TOTAL_TIME/2)
    if 2ramp_time == TOTAL_TIME
        Gpeak = slew * ramp_time
        return t <= ramp_time ? (t/ramp_time)*Gpeak :
               t >= TOTAL_TIME - ramp_time ? ((TOTAL_TIME - t)/ramp_time)*Gpeak : Gpeak
    elseif 2ramp_time < TOTAL_TIME
        if t < ramp_time
            return (t/ramp_time)*Gmax
        elseif t <= TOTAL_TIME - ramp_time
            return Gmax
        else
            return ((TOTAL_TIME - t)/ramp_time)*Gmax
        end
    else
        Gpeak = slew * ramp_time
        return t <= ramp_time ? (t/ramp_time)*Gpeak : ((TOTAL_TIME - t)/ramp_time)*Gpeak
    end
end

@inline function rf_at(t, I, Q)
    s = clamp(t * INV_SEG_DUR, 0f0, NUM_TIME_SEGMENTS - 1f-6)
    i1 = clamp(Int(floor(s)) + 1, 1, NUM_TIME_SEGMENTS - 1)
    i2 = i1 + 1
    α = s - (i1 - 1)
    @inbounds bx = I[i1]*(1-α) + I[i2]*α
    @inbounds by = Q[i1]*(1-α) + Q[i2]*α
    return bx, by
end

function generate_masks(z_positions)
    in_roi   = (@. -DELTA/2      <= z_positions <= DELTA/2)
    in_core  = (@. -DELTA_CORE/2 <= z_positions <= DELTA_CORE/2)
    in_trans = in_roi .& .!in_core
    in_stop  = .!in_roi
    target_mag = zeros(Float32, length(z_positions))
    target_mag[in_core] .= 1f0
    target_mag[in_trans] .= NaN32
    return in_roi, in_core, in_trans, in_stop, target_mag
end

############################
# GPU context
############################
function make_kernel_context()
    M_xy = CUDA.zeros(Float32, 2NZ)
    M_z  = CUDA.zeros(Float32, NZ)

    _, in_core_b, _, in_stop_b, target_mag = generate_masks(Z_BASE)

    # Phantom (static along t=1 column for positions)
    obj = (
        p_x  = CUDA.zeros(Float32, NZ, 1),
        p_y  = CUDA.zeros(Float32, NZ, 1),
        p_z  = cu(reshape(Z_BASE, :, 1)),
        p_ΔBz= CUDA.zeros(Float32, NZ),
        p_T1 = CUDA.fill(PARAMS.T1, NZ),
        p_T2 = CUDA.fill(PARAMS.T2, NZ),
        p_ρ  = CUDA.fill(PARAMS.M0, NZ)
    )

    # Sequence (precompute gz)
    tgrid     = collect(0f0:DT:((NSTEPS-1)*DT))
    gz_values = Float32[gz_at(t) for t in tgrid]

    seq = (
        s_Gx = CUDA.zeros(Float32, NSTEPS),
        s_Gy = CUDA.zeros(Float32, NSTEPS),
        s_Gz = cu(gz_values),
        s_Δt = CUDA.fill(DT, NSTEPS),
        s_Δf = CUDA.zeros(Float32, NSTEPS),
        s_B1 = CUDA.zeros(Float32, 2*NSTEPS),   # layout: [I(1..T); Q(1..T)]
        in_core = cu(in_core_b),                # Bool
        in_stop = cu(in_stop_b)                 # Bool
    )

    target = CUDA.zeros(Float32, 2NZ)
    target[1:NZ] .= cu(target_mag)   # real target; imag target left at 0

    return (; M_xy, M_z, obj, seq, target)
end

const KCTX = make_kernel_context()
const IN_ROI_BASE, IN_CORE_BASE, IN_TRANS_BASE, IN_STOP_BASE, TARGET_MAG_BASE = generate_masks(Z_BASE)

############################
# CPU profile simulator (unchanged)
############################
@inline function rot_apply(m::SVector{3}, Ω::SVector{3})
    θ = norm(Ω)
    iszero(θ) && return m
    n = Ω / θ
    c, s = sincos(θ)
    return m*c + cross(n, m)*s + n*(dot(n, m))*(1-c)
end

@inline function bloch_step(dt, t, m, I, Q, z, p)
    bx_n, by_n   = rf_at(t, I, Q)
    gz_n         = gz_at(t)
    bx_n1, by_n1 = rf_at(t+dt, I, Q)
    gz_n1        = gz_at(t+dt)

    B_n  = SVector{3}(bx_n,  by_n,  gz_n  * z)
    B_n1 = SVector{3}(bx_n1, by_n1, gz_n1 * z)

    A_n  = -p.γ * B_n
    A_n1 = -p.γ * B_n1
    Ω1 = (dt/2) * (A_n + A_n1)
    Ω2 = (dt*dt/12) * cross(A_n1, A_n)
    Ω  = Ω1 + Ω2

    m_rot = rot_apply(m, Ω)
    relax = SVector{3}(-m[1]/p.T2*dt, -m[2]/p.T2*dt, -(m[3]-p.M0)/p.T1*dt)
    return m_rot + relax
end

function simulate_profile(I, Q; z_positions=Z_BASE)
    T = promote_type(eltype(I), eltype(Q), Float32)
    Mx = Vector{T}(undef, NZ); My = similar(Mx); Mz = similar(Mx)
    @inbounds for iz in 1:NZ
        mm = SVector{3}(zero(T), zero(T), T(PARAMS.M0))
        t  = zero(T); zz = z_positions[iz]
        for _ in 1:NSTEPS
            mm = bloch_step(DT, t, mm, I, Q, T(zz), PARAMS)
            t += DT
        end
        Mx[iz]=mm[1]; My[iz]=mm[2]; Mz[iz]=mm[3]
    end
    return Mx, My, Mz
end

############################
# CUDA kernels (forward + gradient via Enzyme)
############################
const γ_HZ_PER_μT = 42.58f6  # matches your earlier γ base; we multiply by π below where needed

@inline function safe_den(x::T) where {T<:AbstractFloat}
    ifelse(x > zero(T), x, eps(T))
end

# per-thread device function (one spin, full time-loop)
@inline function excite_one!(
    M_xy::CuDeviceVector{T}, M_z::CuDeviceVector{T},
    p_x::CuDeviceMatrix{T}, p_y::CuDeviceMatrix{T}, p_z::CuDeviceMatrix{T},
    p_ΔBz::CuDeviceVector{T}, p_T1::CuDeviceVector{T}, p_T2::CuDeviceVector{T}, p_ρ::CuDeviceVector{T},
    s_Gx::CuDeviceVector{T}, s_Gy::CuDeviceVector{T}, s_Gz::CuDeviceVector{T},
    s_Δt::CuDeviceVector{T}, s_Δf::CuDeviceVector{T}, s_B1::CuDeviceVector{T},
    N_Spins::Int32, N_Δt::Int32, γT::T, i::Int32
) where {T<:AbstractFloat}
    x   = @inbounds p_x[i,1]
    y   = @inbounds p_y[i,1]
    z   = @inbounds p_z[i,1]
    ΔBz = @inbounds p_ΔBz[i]
    T1  = @inbounds p_T1[i]
    T2  = @inbounds p_T2[i]
    ρ   = @inbounds p_ρ[i]

    Mxr = @inbounds M_xy[i]
    Myi = @inbounds M_xy[i + N_Spins]
    Mzz = @inbounds M_z[i]

    @inbounds for s::Int32 = 1:N_Δt
        gx = s_Gx[s]; gy = s_Gy[s]; gz = s_Gz[s]
        Δt = s_Δt[s]; Δf = s_Δf[s]
        B1r = s_B1[s]
        B1i = s_B1[s + N_Δt]

        Bz  = muladd(x,gx, muladd(y,gy, muladd(z,gz, ΔBz))) - Δf/γT
        Bsq = muladd(B1r,B1r, muladd(B1i,B1i, Bz*Bz))
        B   = sqrt(Bsq)

        φ   = -T(π) * γT * B * Δt
        sφ, cφ = CUDA.sincos(φ)

        den   = safe_den(B)
        scale = sφ/den

        αr = cφ
        αi = -Bz*scale
        βr =  B1i*scale
        βi = -B1r*scale

        αr2=αr*αr; αi2=αi*αi; βr2=βr*βr; βi2=βi*βi

        Mxr_new = T(2) * (Myi*(muladd(αr,αi,-βr*βi)) + Mzz*(muladd(αi,βi,αr*βr))) + Mxr*(αr2 - αi2 - βr2 + βi2)

        Myi_new = -T(2) * (Mxr*(muladd(αr,αi, βr*βi)) - Mzz*(muladd(αr,βi,-αi*βr))) + Myi*(αr2 - αi2 + βr2 - βi2)

        Mzz_new =  Mzz*(αr2 + αi2 - βr2 - βi2) - T(2) * (Mxr*(muladd(αr,βr,-αi*βi)) + Myi*(muladd(αr,βi,αi*βr)))

        ΔT1 = exp(-Δt/T1)
        ΔT2 = exp(-Δt/T2)
        Mxr = Mxr_new * ΔT2
        Myi = Myi_new * ΔT2
        Mzz = muladd(Mzz_new, ΔT1, ρ*(one(T)-ΔT1))
    end

    @inbounds begin
        M_xy[i]           = Mxr
        M_xy[i + N_Spins] = Myi
        M_z[i]            = Mzz
    end
    return
end    

# forward: parallel over spins
function kernel_excite_forward!(
    M_xy::CuDeviceVector{T}, M_z::CuDeviceVector{T},
    p_x::CuDeviceMatrix{T}, p_y::CuDeviceMatrix{T}, p_z::CuDeviceMatrix{T},
    p_ΔBz::CuDeviceVector{T}, p_T1::CuDeviceVector{T}, p_T2::CuDeviceVector{T}, p_ρ::CuDeviceVector{T},
    s_Gx::CuDeviceVector{T}, s_Gy::CuDeviceVector{T}, s_Gz::CuDeviceVector{T},
    s_Δt::CuDeviceVector{T}, s_Δf::CuDeviceVector{T}, s_B1::CuDeviceVector{T},
    N_Spins::Int32, N_Δt::Int32, γT::T
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i >= 1 && i <= N_Spins
        @inbounds excite_one!(M_xy, M_z,
                              p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
                              s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1,
                              N_Spins, N_Δt, γT, Int32(i))
    end
    return
end


# gradient: run Enzyme on excite_one! per-thread to backprop from (M_xȳ,M_z̄) into s_B1̄
function kernel_excite_grad!(
    M_xy::CuDeviceVector{T}, M_xȳ::CuDeviceVector{T},
    M_z::CuDeviceVector{T},  M_z̄::CuDeviceVector{T},
    p_x::CuDeviceMatrix{T}, p_y::CuDeviceMatrix{T}, p_z::CuDeviceMatrix{T},
    p_ΔBz::CuDeviceVector{T}, p_T1::CuDeviceVector{T}, p_T2::CuDeviceVector{T}, p_ρ::CuDeviceVector{T},
    s_Gx::CuDeviceVector{T}, s_Gy::CuDeviceVector{T}, s_Gz::CuDeviceVector{T},
    s_Δt::CuDeviceVector{T}, s_Δf::CuDeviceVector{T},
    s_B1::CuDeviceVector{T}, s_B1̄::CuDeviceVector{T},
    N_Spins::Int32, N_Δt::Int32, γT::T
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if !(i >= 1 && i <= N_Spins); return; end

    Enzyme.autodiff(Enzyme.Reverse,
        excite_one!,
        Enzyme.Duplicated(M_xy,  M_xȳ),
        Enzyme.Duplicated(M_z,   M_z̄),
        Enzyme.Const(p_x), Enzyme.Const(p_y), Enzyme.Const(p_z),
        Enzyme.Const(p_ΔBz), Enzyme.Const(p_T1), Enzyme.Const(p_T2), Enzyme.Const(p_ρ),
        Enzyme.Const(s_Gx), Enzyme.Const(s_Gy), Enzyme.Const(s_Gz),
        Enzyme.Const(s_Δt), Enzyme.Const(s_Δf),
        Enzyme.Duplicated(s_B1, s_B1̄),
        Enzyme.Const(N_Spins), Enzyme.Const(N_Δt), Enzyme.Const(γT),
        Enzyme.Const(Int32(i)))
    return
end


function excitation_forward!(M_xy::CuArray{T,1}, M_z::CuArray{T,1}, obj, seq) where {T<:AbstractFloat}
    Nsp = Int32(length(M_z)); Ndt = Int32(length(seq.s_Δt)); γT = T(42.58f6)
    @cuda threads=256 blocks=cld(Nsp,256) kernel_excite_forward!(
        M_xy, M_z, obj.p_x, obj.p_y, obj.p_z, obj.p_ΔBz, obj.p_T1, obj.p_T2, obj.p_ρ,
        seq.s_Gx, seq.s_Gy, seq.s_Gz, seq.s_Δt, seq.s_Δf, seq.s_B1,
        Nsp, Ndt, γT)
    CUDA.synchronize()
end

function excitation_grad!(M_xy, M_xȳ, M_z, M_z̄, obj, seq, s_B1̄::CuArray{T,1}) where {T<:AbstractFloat}
    Nsp = Int32(length(M_z)); Ndt = Int32(length(seq.s_Δt)); γT = T(42.58f6)
    @cuda threads=256 blocks=cld(Nsp,256) kernel_excite_grad!(
        M_xy, M_xȳ, M_z, M_z̄,
        obj.p_x, obj.p_y, obj.p_z, obj.p_ΔBz, obj.p_T1, obj.p_T2, obj.p_ρ,
        seq.s_Gx, seq.s_Gy, seq.s_Gz, seq.s_Δt, seq.s_Δf,
        seq.s_B1, s_B1̄,
        Nsp, Ndt, γT)
    CUDA.synchronize()
end


############################
# Loss + gradient (GPU forward + GPU RF-grad)
############################
# Simple slice-creation loss (pass/stop); RF regularizers handled on CPU and added to RF grad.
function loss_and_grad_gpu!(x::Vector{Float32};
    obj=KCTX.obj, seq=KCTX.seq, M_xy=KCTX.M_xy, M_z=KCTX.M_z, target=KCTX.target,
    in_core=seq.in_core, in_stop=seq.in_stop,
    λ_profile=200f0, λ_smooth=1f10, λ_power=10000f0, λ_profile_tv=0f0
)
    # set RF on device: s_B1 = [I; Q]
    copyto!(seq.s_B1, x)

    # forward
    fill!(M_xy, 0); fill!(M_z, 0)
    excitation_forward!(M_xy, M_z, obj, seq)

    Nsp = length(M_z)
    Mx  = view(M_xy, 1:Nsp)
    My  = view(M_xy, Nsp+1:2Nsp)
    tgtx = view(target, 1:Nsp)
    tgty = view(target, Nsp+1:2Nsp)

    # pass/stop losses (device reductions, not differentiated)
    dmx = Mx .- tgtx
    dmy = My .- tgty
    mag2 = Mx.^2 .+ My.^2

    pass_loss = CUDA.sum((dmx.^2 .+ dmy.^2)[in_core])
    stop_loss = CUDA.sum(mag2[in_stop])

    core_count = max(1, Int(CUDA.sum(in_core)))
    stop_count = max(1, Int(CUDA.sum(in_stop)))

    L_slice = λ_profile * (pass_loss / core_count) + 2f0*λ_profile * (stop_loss / stop_count)

    # seeds for states: dL/dMx, dL/dMy (Mz is not used → zero)
    Mx̄ = CUDA.zeros(Float32, Nsp)
    Mȳ = CUDA.zeros(Float32, Nsp)
    Mz̄ = CUDA.zeros(Float32, Nsp)

    @. Mx̄[in_core] += (2f0*λ_profile/core_count) * dmx[in_core]
    @. Mȳ[in_core] += (2f0*λ_profile/core_count) * dmy[in_core]
    @. Mx̄[in_stop] += (4f0*λ_profile/stop_count) * Mx[in_stop]
    @. Mȳ[in_stop] += (4f0*λ_profile/stop_count) * My[in_stop]

    # pack adjoint for M_xy (SoA in a vector)
    M_xȳ = similar(M_xy); fill!(M_xȳ, 0)
    copyto!(view(M_xȳ, 1:Nsp), Mx̄)
    copyto!(view(M_xȳ, Nsp+1:2Nsp), Mȳ)

    # RF gradient buffer on device
    s_B1̄ = CUDA.zeros(Float32, 2*NSTEPS)

    # backprop through excitation on GPU (Enzyme-inside-kernel)
    excitation_grad!(M_xy, M_xȳ, M_z, Mz̄, obj, seq, s_B1̄)

    # bring RF gradient to host
    g = Array(s_B1̄)  # g = [∂L/∂I; ∂L/∂Q]

    # add RF regularizers (CPU, analytical)
    n = length(x) ÷ 2
    I = view(x, 1:n); Q = view(x, n+1:2n)
    gI = view(g, 1:n); gQ = view(g, n+1:2n)

    # power: λ_power * (||I||^2+||Q||^2)/(2n) → grad = λ_power/n * {I,Q}
    @. gI += (λ_power/Float32(n)) * I
    @. gQ += (λ_power/Float32(n)) * Q

    # TV on RF (L2 on diffs): λ_smooth * (||ΔI||^2 + ||ΔQ||^2)
    if n > 1 && λ_smooth != 0f0
        # grad equivalent to discrete Laplacian with Neumann ends
        @inbounds begin
            # I
            gI[1]     += 2f0*λ_smooth*(I[1]-I[2])
            for k in 2:n-1
                gI[k] += 2f0*λ_smooth*(2f0*I[k] - I[k-1] - I[k+1])
            end
            gI[n]     += 2f0*λ_smooth*(I[n]-I[n-1])
            # Q
            gQ[1]     += 2f0*λ_smooth*(Q[1]-Q[2])
            for k in 2:n-1
                gQ[k] += 2f0*λ_smooth*(2f0*Q[k] - Q[k-1] - Q[k+1])
            end
            gQ[n]     += 2f0*λ_smooth*(Q[n]-Q[n-1])
        end
    end

    # (optional) profile TV term omitted from gradient for brevity; add if needed.

    # scalar loss (add RF regs to slice loss)
    rf_power = sum(abs2, I) + sum(abs2, Q)
    tv_rf    = (n>1) ? (sum(abs2, @views diff(I)) + sum(abs2, @views diff(Q))) : 0f0
    L = L_slice + λ_power * rf_power / (2f0*n) + λ_smooth * tv_rf + λ_profile_tv * 0f0

    return L, g
end

############################
# Optimizer (Adam)
############################
function optimize_adam(x0;
    max_iters=200, lr=5e-6, lr_decay=0.995, β1=0.9, β2=0.999, ϵ=1e-8,
    grad_clip=5e2, patience=50, min_delta=1e-6, rel_tol=1e-4,
    plot_every=1, save_plots=true,
    obj_params = (λ_profile=200f0, λ_smooth=1f10, λ_power=10000f0, λ_profile_tv=0f0)
)
    gr()
    x = Float32.(x0); m = zero.(x); v = zero.(x)
    best_loss = Inf32; best_x = copy(x); best_iter = 0; patience_ctr = 0
    loss_hist = Float32[]; plots_array = Any[]

    for t in 1:max_iters
        L, g = loss_and_grad_gpu!(x; λ_profile=obj_params.λ_profile,
                                     λ_smooth=obj_params.λ_smooth,
                                     λ_power=obj_params.λ_power,
                                     λ_profile_tv=obj_params.λ_profile_tv)

        push!(loss_hist, L)

        # clip
        gnorm = norm(g)
        if gnorm > grad_clip
            @. g = g * (grad_clip/gnorm)
            gnorm = grad_clip
        end

        # Adam
        @. m = β1*m + (1-β1)*g
        @. v = β2*v + (1-β2)*g*g
        mhat = @. m / (1 - β1^t)
        vhat = @. v / (1 - β2^t)
        lr_t = lr * lr_decay^t
        @. x -= lr_t * mhat / (sqrt(vhat) + ϵ)
        @. x = clamp(x, -RF_BOUND, RF_BOUND)

        # best tracking
        improv = best_loss - L
        if improv > min_delta || abs(improv / max(abs(best_loss), 1e-10)) > rel_tol
            best_loss = L; best_x .= x; best_iter = t; patience_ctr = 0
        else
            patience_ctr += 1
        end

        # plotting
        if t % plot_every == 0
            n = length(x) ÷ 2; I = x[1:n]; Q = x[n+1:end]
            p = plot_progress(I, Q, t, L)
            display(p); save_plots && push!(plots_array, p)
            @printf("iter %d: loss=%.6g |g|=%.3g lr=%.3g best=%.6g@%d patience=%d/%d\n",
                    t, L, gnorm, lr_t, best_loss, best_iter, patience_ctr, patience)
        end

        if patience_ctr >= patience
            println("Early stopping at iter $t"); break
        end
        if t > 50 && gnorm < 1e-7
            println("Converged at iter $t"); break
        end
    end

    return best_loss < loss_hist[end] ? best_x : x, loss_hist, plots_array, []
end

############################
# Plotting (unchanged)
############################
function init_iq(nseg; Ttot=TOTAL_TIME, tbw=3.0, flip_deg=90.0)
    tmid = Ttot/2
    σ = Ttot / (2.5f0 * tbw)
    tgrid = range(0f0, Ttot, length=nseg)
    s = @. exp(-0.5f0 * ((tgrid - tmid) / σ)^2)

    θ = deg2rad(flip_deg)
    dt_seg = Ttot / (nseg - 1)
    area = sum(abs, s) * dt_seg
    scale = (θ / PARAMS.γ) / area

    I = @. clamp(s * scale, -RF_BOUND*0.8f0, RF_BOUND*0.8f0)
    Q = zeros(Float32, length(I))
    return I, Q
end

function plot_progress(I, Q, iter, loss_val)
    Mx, My, _ = simulate_profile(I, Q)
    mag = hypot.(Mx, My)
    t_rf = range(0, TOTAL_TIME, length=length(I)) .* 1e3

    p = plot(layout=(1,2), size=(900, 400))
    plot!(p[1], t_rf, I*1e6, label="Bx (I)", lw=2)
    plot!(p[1], t_rf, Q*1e6, label="By (Q)", lw=2, alpha=0.7)
    hline!(p[1], [RF_BOUND*1e6, -RF_BOUND*1e6], ls=:dot, alpha=0.5, label="")
    plot!(p[1], xlabel="Time (ms)", ylabel="RF (μT)",
          title="iter $iter | loss $(round(loss_val, sigdigits=4))", legend=:topright)

    vspan!(p[2], [-DELTA*1e3/2, -DELTA_CORE*1e3/2], alpha=0.12, color=:yellow, label="")
    vspan!(p[2], [ DELTA_CORE*1e3/2,  DELTA*1e3/2], alpha=0.12, color=:yellow, label="")
    plot!(p[2], Z_BASE*1e3, mag, label="|Mxy|", lw=2, xlabel="z (mm)", ylabel="|Mxy|",
          title="Slice Profile", ylims=(-0.1, 1.2), legend=:topright)
    return p
end

############################
# Main
############################
function run_optim(; max_iters=150, lr=2f-6, patience=40, plot_every=1)
    println("Initializing RF profile...")
    I0, Q0 = init_iq(NUM_TIME_SEGMENTS)
    x0 = vcat(I0, Q0)

    println("Starting optimization (GPU fwd + GPU RF-grad)...")
    x_opt, L_hist, P_list, M_hist = optimize_adam(
        x0; max_iters, lr, lr_decay=0.998, patience,
        min_delta=1e-6, rel_tol=1e-4, grad_clip=1e2, plot_every, save_plots=true
    )
    return (xopt=x_opt, loss_hist=L_hist, plots=P_list, metrics=M_hist)
end

println("Running optimization...")
@time best = run_optim()

xopt = best.xopt; loss_hist = best.loss_hist
n = length(xopt) ÷ 2; Iopt = xopt[1:n]; Qopt = xopt[n+1:end]

Mx, My, Mz = simulate_profile(Iopt, Qopt)
mag = hypot.(Mx, My); phase = atan.(My, Mx)
mag_core = mag[IN_CORE_BASE]; mag_stop = mag[IN_STOP_BASE]

println("\nFINAL PERFORMANCE METRICS")
println("Optimization iters: $(length(loss_hist))")
println("Final loss: $(round(loss_hist[end], sigdigits=5))")
println("Passband mean: $(round(mean(mag_core), digits=3))")
println("Passband ripple: $(round(maximum(mag_core) - minimum(mag_core), digits=3))")
println("Max stopband: $(round(maximum(mag_stop), digits=4))")
println("Mean stopband: $(round(mean(mag_stop), digits=4))")
println("Phase RMS (core): $(round(sqrt(mean((phase[IN_CORE_BASE]).^2)), digits=3)) rad")
println("Max RF amplitude: $(round(maximum(hypot.(Iopt, Qopt))*1e6, digits=2)) μT")

p_final = plot_progress(Iopt, Qopt, length(loss_hist), loss_hist[end])
display(p_final)
savefig(p_final, "final_optimized_profile.png")
