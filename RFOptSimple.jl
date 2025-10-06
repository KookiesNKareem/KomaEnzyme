using CUDA
using KernelAbstractions
using Adapt
import Enzyme
using Random: seed!
using Plots
using Statistics: mean, maximum
using LinearAlgebra: norm

ENV["GKSwstype"] = "100" 

CUDA.allowscalar(false)

const GROUP_SIZE = 256
const Nspins     = 129
const Nt         = 256
const dt         = 1.0f-5
const γ          = Float32(2π) * 42.58f6
const N_Spins32 = Int32(Nspins)
const N_Δt32    = Int32(Nt)

# segmented RF
const NUM_TIME_SEGMENTS = 50
const INV_SEG_DUR = NUM_TIME_SEGMENTS / (Nt*dt)

# slew-limited Gz
const MAX_GZ   = 10.0f-3     # T/m
const MAX_SLEW = 70.0f0      # T/m/s
@inline function gz_at(t; Gmax=MAX_GZ, slew=MAX_SLEW)
    Ttot = Nt*dt
    ramp_time = min(Gmax/slew, Ttot/2)
    if 2ramp_time == Ttot
        Gpeak = slew * ramp_time
        return t <= ramp_time ? (t/ramp_time)*Gpeak :
               t >= Ttot - ramp_time ? ((Ttot - t)/ramp_time)*Gpeak : Gpeak
    elseif 2ramp_time < Ttot
        return t < ramp_time ? (t/ramp_time)*Gmax :
               t <= Ttot - ramp_time ? Gmax :
               ((Ttot - t)/ramp_time)*Gmax
    else
        Gpeak = slew * ramp_time
        return t <= ramp_time ? (t/ramp_time)*Gpeak :
                                ((Ttot - t)/ramp_time)*Gpeak
    end
end

backend = CUDA.CUDABackend()
seed!(42)

M_xy_h = zeros(Float32, 2Nspins)
M_z_h  = zeros(Float32, Nspins)

p_x_h   = zeros(Float32, Nspins)
p_y_h   = zeros(Float32, Nspins)

const half_FOVz = 10.0f-3
p_z_h = collect(LinRange(-half_FOVz, half_FOVz, Nspins)) .|> Float32
p_ΔBz_h = zeros(Float32, Nspins)
p_T1_h  = fill(1.0f0, Nspins)
p_T2_h  = fill(0.5f0, Nspins)
p_ρ_h   = fill(1.0f0, Nspins)

s_Gx_h  = zeros(Float32, Nt)
s_Gy_h  = zeros(Float32, Nt)
s_Gz_h  = [gz_at((k-1)*dt) for k in 1:Nt] .|> Float32
s_Δt_h  = fill(dt, Nt)
s_Δf_h  = zeros(Float32, Nt)
s_B1r_h = fill(1.0f-6, Nt)
s_B1i_h = zeros(Float32, Nt)

const box_halfwidth = 2.5f-3
target_mag_h = map(z -> abs(z) <= box_halfwidth ? 1.0f0 : 0.0f0, p_z_h) |> collect

# pass/stop masks (metrics only)
const CORE = box_halfwidth
in_core_h = (@. -CORE <= p_z_h <= CORE)
in_stop_h = .!(@. -box_halfwidth <= p_z_h <= box_halfwidth)

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
target_mag = adapt(backend, target_mag_h)

RF_I = 1.0f-6 .* (2rand(Float32, NUM_TIME_SEGMENTS) .- 1.0f0)
RF_Q = zeros(Float32, NUM_TIME_SEGMENTS)

function upsample_rf!(s_B1r::CUDA.CuArray{Float32,1}, s_B1i::CUDA.CuArray{Float32,1},
    I::Vector{Float32}, Q::Vector{Float32})
        r = Vector{Float32}(undef, Nt)
        i = Vector{Float32}(undef, Nt)

        @inbounds for k in 1:Nt
            t = (k-1)*dt
            s   = clamp(t*INV_SEG_DUR, 0.0f0, NUM_TIME_SEGMENTS - 1.0f-6)
            i1  = clamp(Int(floor(s)) + 1, 1, NUM_TIME_SEGMENTS - 1)
            i2  = i1 + 1
            α   = s - (i1 - 1)
            r[k] = I[i1]*(1.0f0-α) + I[i2]*α
            i[k] = Q[i1]*(1.0f0-α) + Q[i2]*α
        end

        copyto!(s_B1r, r)
        copyto!(s_B1i, i)

        return nothing
end

function accumulate_seg_grads!(gI::Vector{Float32}, gQ::Vector{Float32},
                               dB1r::CUDA.CuArray{Float32,1}, dB1i::CUDA.CuArray{Float32,1})
    hgr = Array(dB1r); hgi = Array(dB1i)
    fill!(gI, 0.0f0); fill!(gQ, 0.0f0)
    @inbounds for k in 1:Nt
        t  = (k-1)*dt
        s  = clamp(t*INV_SEG_DUR, 0.0f0, NUM_TIME_SEGMENTS - 1.0f-6)
        i1 = clamp(Int(floor(s)) + 1, 1, NUM_TIME_SEGMENTS - 1)
        i2 = i1 + 1
        α  = s - (i1 - 1)
        w1 = (1.0f0 - α) * dt
        w2 = α * dt
        gI[i1] += w1 * hgr[k]; gI[i2] += w2 * hgr[k]
        gQ[i1] += w1 * hgi[k]; gQ[i2] += w2 * hgi[k]
    end
    invT = 1.0f0 / (Nt*dt)
    @. gI *= invT
    @. gQ *= invT
    return nothing
end


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
            ϕ  = T(γ) * B * Δt
            sϕ = sin(ϕ); cϕ = cos(ϕ)
            denom = B + T(1.0f-20)
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
            Mz    = Mz_new * ΔT1 + ρ * (T(1.0f0) - ΔT1)
            s_idx += 1
        end
        M_xy[ir] = Mxy_r
        M_xy[ii] = Mxy_i
        M_z[i]   = Mz
    end
end

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

@kernel function mag_loss_grad_M!(dM, M, TGTmag::AbstractVector{Float32})
    i = @index(Global, Linear)
    N = length(TGTmag)
    if i <= N
        ir = i; ii = i + N
        @inbounds begin
            mr = M[ir]; mi = M[ii]
            mag = sqrt(mr*mr + mi*mi)
            e   = mag - TGTmag[i]          # mean-squared: L = mean(e^2)
            if mag < 1.0f-12
                # subgradient pointing along 45° when mag≈0
                g = 2.0f0 * e / Float32(N)
                dM[ir] = g * 0.70710677f0
                dM[ii] = g * 0.70710677f0
            else
                invmag = 1.0f0 / mag
                gfac = (2.0f0 * e / Float32(N)) * invmag
                dM[ir] = gfac * mr
                dM[ii] = gfac * mi
            end
        end
    end
end

function gpu_loss_mag(M_xy::CUDA.CuArray{T}, target_mag::CUDA.CuArray{T}) where {T<:AbstractFloat}
    N = length(target_mag)
    Mr = view(M_xy, 1:N)
    Mi = view(M_xy, N+1:2N)
    mag = sqrt.(Mr .* Mr .+ Mi .* Mi .+ T(1.0f-20))
    err = mag .- target_mag
    return CUDA.sum(err .* err) / T(N)   # ← mean, not sum
end


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

function grad_rf!(
    ∇B1r::CUDA.CuArray{Float32}, ∇B1i::CUDA.CuArray{Float32},
    M_xy, M_z,
    p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
    s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1r, s_B1i,
    target_mag,
    backend,
)
    fill!(M_xy, 0.0f0); fill!(M_z, 1.0f0)
    excite_only!(M_xy, M_z,
        p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
        s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1r, s_B1i,
        backend)
    KernelAbstractions.synchronize(backend)

    dM_xy = similar(M_xy)
    mag_loss_grad_M!(backend, GROUP_SIZE)(dM_xy, M_xy, target_mag; ndrange=Int(length(target_mag)))
    KernelAbstractions.synchronize(backend)

    dM_z = similar(M_z);  fill!(dM_z, 0.0f0)
    dΔt  = similar(s_Δt); fill!(dΔt,  0.0f0)
    dΔf  = similar(s_Δf); fill!(dΔf,  0.0f0)
    fill!(∇B1r, 0.0f0); fill!(∇B1i, 0.0f0)

    Enzyme.autodiff(Enzyme.Reverse, excite_only!,
        Enzyme.Duplicated(M_xy, dM_xy),
        Enzyme.Duplicated(M_z,  dM_z),
        Enzyme.Const(p_x), Enzyme.Const(p_y), Enzyme.Const(p_z),
        Enzyme.Const(p_ΔBz), Enzyme.Const(p_T1), Enzyme.Const(p_T2), Enzyme.Const(p_ρ),
        Enzyme.Const(s_Gx), Enzyme.Const(s_Gy), Enzyme.Const(s_Gz),
        Enzyme.Duplicated(s_Δt, dΔt),
        Enzyme.Duplicated(s_Δf, dΔf),
        Enzyme.Duplicated(s_B1r, ∇B1r),
        Enzyme.Duplicated(s_B1i, ∇B1i),
        Enzyme.Const(backend),
    )
    KernelAbstractions.synchronize(backend)
    return gpu_loss_mag(M_xy, target_mag)
end

mutable struct AdamWState
    t::Int
    mI::Vector{Float32}
    vI::Vector{Float32}
    mQ::Vector{Float32}
    vQ::Vector{Float32}
end

function adamw_step_segments!(
    I, Q, gI, gQ, st::AdamWState;
    lr::Float32=3.0f-3, beta1::Float32=0.9f0, beta2::Float32=0.999f0,
    eps::Float32=1.0f-8, weight_decay::Float32=1.0f-4
)
    st.t += 1
    @. st.mI = beta1*st.mI + (1.0f0-beta1)*gI
    @. st.vI = beta2*st.vI + (1.0f0-beta2)*(gI*gI)
    @. st.mQ = beta1*st.mQ + (1.0f0-beta1)*gQ
    @. st.vQ = beta2*st.vQ + (1.0f0-beta2)*(gQ*gQ)
    b1t = 1.0f0 - beta1^st.t
    b2t = 1.0f0 - beta2^st.t
    @. I = (1.0f0 - lr*weight_decay)*I
    @. Q = (1.0f0 - lr*weight_decay)*Q
    invI = 1.0f0 ./ (sqrt.(st.vI ./ b2t) .+ eps)
    invQ = 1.0f0 ./ (sqrt.(st.vQ ./ b2t) .+ eps)
    @. I = I - lr * ((st.mI / b1t) * invI)
    @. Q = Q - lr * ((st.mQ / b1t) * invQ)
    return nothing
end

function clip_seg_grad!(gI, gQ, clip::Float32)
    nrm = sqrt(sum(@. gI*gI + gQ*gQ))
    if nrm > clip
        sc = clip/nrm
        @. gI *= sc; @. gQ *= sc
    end
end

# RF regularizers (segments)
const RF_BOUND  = 12.0f-6
const λ_power   = 1.0f3     # was 1e4
const λ_smooth  = 1.0f6     # was 1e10

function add_rf_regs!(gI, gQ, I, Q)
    n = length(I)
    @. gI += (λ_power/n) * I
    @. gQ += (λ_power/n) * Q
    if n > 1 && λ_smooth != 0.0f0
        gI[1]     += 2.0f0*λ_smooth*(I[1]-I[2])
        for k in 2:n-1
            gI[k] += 2.0f0*λ_smooth*(2.0f0*I[k] - I[k-1] - I[k+1])
        end
        gI[n]     += 2.0f0*λ_smooth*(I[n]-I[n-1])
        gQ[1]     += 2.0f0*λ_smooth*(Q[1]-Q[2])
        for k in 2:n-1
            gQ[k] += 2.0f0*λ_smooth*(2.0f0*Q[k] - Q[k-1] - Q[k+1])
        end
        gQ[n]     += 2.0f0*λ_smooth*(Q[n]-Q[n-1])
    end
end

∇B1r = similar(s_B1r);  ∇B1i = similar(s_B1i)

function evaluate_and_grad!(target_mag)
    upsample_rf!(s_B1r, s_B1i, RF_I, RF_Q)
    L = grad_rf!(∇B1r, ∇B1i,
                 M_xy, M_z,
                 p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
                 s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1r, s_B1i,
                 target_mag, backend)
    KernelAbstractions.synchronize(backend)
    return Float32(L)
end

function run_adamw!(num_iters::Int=300; lr=3.0f-3, beta1=0.9f0, beta2=0.999f0,
                    eps=1.0f-8, weight_decay=1.0f-4, grad_clip=5.0f1)
    st = AdamWState(0, zeros(Float32, NUM_TIME_SEGMENTS), zeros(Float32, NUM_TIME_SEGMENTS),
                       zeros(Float32, NUM_TIME_SEGMENTS), zeros(Float32, NUM_TIME_SEGMENTS))
    gI = similar(RF_I); gQ = similar(RF_Q)
    cur_lr = lr; clip_hits = 0

    losses = Vector{Float32}(undef, num_iters)
    grad_hist = similar(losses)
    lr_hist = similar(losses)

    @info "Starting AdamW" iters=num_iters lr=lr
    for it in 1:num_iters
        L = evaluate_and_grad!(target_mag)
        accumulate_seg_grads!(gI, gQ, ∇B1r, ∇B1i)
        add_rf_regs!(gI, gQ, RF_I, RF_Q)

        gn = sqrt(sum(@. gI*gI + gQ*gQ))
        if gn > grad_clip
            sc = grad_clip/gn
            @. gI *= sc; @. gQ *= sc
            clip_hits += 1
        else
            clip_hits = max(clip_hits - 1, 0)
        end
        if clip_hits >= 3
            cur_lr *= 0.5f0
            clip_hits = 0
        end

        adamw_step_segments!(RF_I, RF_Q, gI, gQ, st; lr=cur_lr, beta1=beta1, beta2=beta2, eps=eps, weight_decay=weight_decay)
        @. RF_I = clamp(RF_I, -RF_BOUND, RF_BOUND)
        @. RF_Q = clamp(RF_Q, -RF_BOUND, RF_BOUND)

        losses[it] = L
        grad_hist[it] = gn
        lr_hist[it] = cur_lr
        @info "iter" it=it L=L grad_norm=gn lr=cur_lr
    end
    return evaluate_and_grad!(target_mag), losses, grad_hist, lr_hist
end

function final_profile_mag!(backend)
    upsample_rf!(s_B1r, s_B1i, RF_I, RF_Q)
    fill!(M_xy, 0.0f0); fill!(M_z, 1.0f0)      
    excite_only!(M_xy, M_z,
                 p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
                 s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1r, s_B1i,
                 backend)
    KernelAbstractions.synchronize(backend)
    N = Int(N_Spins32)
    Mr = Array(view(M_xy, 1:N))
    Mi = Array(view(M_xy, N+1:2N))
    return sqrt.(Mr .* Mr .+ Mi .* Mi)
end

function plot_training_history(losses::AbstractVector{<:Real};
                               grad_hist::Union{Nothing,AbstractVector{<:Real}}=nothing,
                               lr_hist::Union{Nothing,AbstractVector{<:Real}}=nothing)
    ENV["GKSwstype"] = "100"
    it = 1:length(losses)
    if grad_hist === nothing && lr_hist === nothing
        return plot(it, losses, lw=2, xlabel="iter", ylabel="loss", title="Loss")
    else
        p1 = plot(it, losses, lw=2, xlabel="iter", ylabel="loss", title="Loss")
        p2 = grad_hist === nothing ? plot() :
             plot(it, grad_hist, lw=2, xlabel="iter", ylabel="‖grad‖", yscale=:log10, title="Grad norm")
        p3 = lr_hist === nothing ? plot() :
             plot(it, lr_hist, lw=2, xlabel="iter", ylabel="lr", yscale=:log10, title="LR")
        return plot(p1, p2, p3; layout=(1,3), size=(1200,350))
    end
end

function plot_results()
    upsample_rf!(s_B1r, s_B1i, RF_I, RF_Q)
    t = collect(0:dt:(Nt-1)*dt)
    p_rf = plot(t, Array(s_B1r), label="B1r", xlabel="time (s)", ylabel="B1", title="Final RF pulse")
    plot!(p_rf, t, Array(s_B1i), label="B1i")
    z_mm = 1.0f3 .* p_z_h
    prof_mag = final_profile_mag!(backend)
    p_prof = plot(z_mm, target_mag_h; label="Target box", xlabel="z (mm)", ylabel="|M_xy|",
                  title="Slice profile magnitude", lw=3, ls=:dash)
    plot!(p_prof, z_mm, prof_mag; label="Achieved", lw=3)
    plot(p_rf, p_prof; layout=(2,1), size=(900,800))
end

final_loss, losses, grad_hist, lr_hist = run_adamw!(300; lr=3.0f-3, weight_decay=1.0f-4, grad_clip=5.0f1)
@info "Final" loss=final_loss

p_hist = plot_training_history(losses; grad_hist=grad_hist, lr_hist=lr_hist)
savefig(p_hist, "training_history.png")

prof = final_profile_mag!(backend)
core = prof[in_core_h]; stop = prof[in_stop_h]
@info "Passband" mean=mean(core) ripple=(maximum(core)-minimum(core))
@info "Stopband" max=maximum(stop) mean=mean(stop)

savefig(plot_results(), "final_rf_profile.png")