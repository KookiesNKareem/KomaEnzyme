###############################
# RF slice design (SPG-BB)
###############################

using KernelAbstractions
using KernelAbstractions: CPU
using CUDA
using Adapt
import Enzyme
using Random: seed!
using Plots
using Statistics: mean, maximum, minimum
import Atomix         

const backend = CUDA.CUDABackend()
# const backend = CPU()
CUDA.allowscalar(false)

Plots.gr()
Plots.default(fmt=:png)
ENV["GKSwstype"] = "100"
seed!(42)

const GROUP_SIZE = 256
const Nspins     = 129
const Nt         = 300 #  Nt*dt = 3ms    
const dt         = 1.0f-5 # 10 µs
const γ          = Float32(2π) * 42.58f6
const N_Spins32  = Int32(Nspins)
const N_Δt32     = Int32(Nt)

const NUM_TIME_SEGMENTS = 200
const INV_SEG_DUR = NUM_TIME_SEGMENTS / (Nt*dt)

const BW_CUTOFF = 2.5f-3
const BW_ORDER  = 16       

const half_FOVz = 10.0f-3
const voxel_resolution = (2 * half_FOVz) / (Nspins - 1)
const position_offset_range = voxel_resolution / 4

const MAX_GZ   = 10.0f-3   
const MAX_SLEW = 70.0f0   

const RF_BOUND = 12.0f-6
const λ_power  = 1.0f-3
const λ_smooth = 0.2f0

@inline function gz_at(t; Gmax=MAX_GZ, slew=MAX_SLEW)
    Ttot = Nt * dt
    ramp_time = min(Gmax / slew, Ttot / 2)

    if 2 * ramp_time == Ttot
        Gpeak = slew * ramp_time
        if t <= ramp_time
            return (t / ramp_time) * Gpeak
        elseif t >= Ttot - ramp_time
            return ((Ttot - t) / ramp_time) * Gpeak
        else
            return Gpeak
        end
    elseif 2 * ramp_time < Ttot
        if t < ramp_time
            return (t / ramp_time) * Gmax
        elseif t <= Ttot - ramp_time
            return Gmax
        else
            return ((Ttot - t) / ramp_time) * Gmax
        end
    else
        Gpeak = slew * ramp_time
        if t <= ramp_time
            return (t / ramp_time) * Gpeak
        else
            return ((Ttot - t) / ramp_time) * Gpeak
        end
    end
end

p_z_base_h = collect(LinRange(-half_FOVz, half_FOVz, Nspins)) .|> Float32

function butterworth_target(z::Vector{Float32}; cutoff::Float32=BW_CUTOFF, n::Int=BW_ORDER)
    r = abs.(z) ./ cutoff
    return 1f0 ./ sqrt.(1f0 .+ (r .^ (2n)))
end

function make_target_and_masks(z::Vector{Float32})
    target_mag = butterworth_target(z; cutoff=BW_CUTOFF, n=BW_ORDER)

    in_core = Float32.(target_mag .> 0.5f0)
    in_stop = Float32.(target_mag .< 0.05f0)

    return in_core, in_stop, target_mag
end

M_xy   = adapt(backend, zeros(Float32, 2Nspins))
M_z    = adapt(backend, zeros(Float32, Nspins))
p_x    = adapt(backend, zeros(Float32, Nspins))
p_y    = adapt(backend, zeros(Float32, Nspins))
p_z    = adapt(backend, copy(p_z_base_h))
p_ΔBz  = adapt(backend, zeros(Float32, Nspins))
p_T1   = adapt(backend, fill(1.0f0, Nspins))
p_T2   = adapt(backend, fill(0.5f0, Nspins))
p_ρ    = adapt(backend, fill(1.0f0, Nspins))
s_Gx   = adapt(backend, zeros(Float32, Nt))
s_Gy   = adapt(backend, zeros(Float32, Nt))
s_Gz   = adapt(backend, [gz_at((k-1)*dt) for k in 1:Nt] .|> Float32)
s_Δt   = adapt(backend, fill(dt, Nt))
s_Δf   = adapt(backend, zeros(Float32, Nt))
s_B1r  = adapt(backend, fill(1.0f-6, Nt))  # time upsamled version
s_B1i  = adapt(backend, zeros(Float32, Nt))

function init_rf_gaussian(nseg::Int)
    tgrid = collect(range(0.0f0, Nt*dt, length=nseg))
    tmid = (Nt*dt)/2
    σ = (Nt*dt) / 7.5f0

    envelope = exp.(-(0.5f0 .* ((tgrid .- tmid) ./ σ).^2))

    θ = deg2rad(90.0f0)
    dt_seg = (Nt*dt) / (nseg - 1)

    area = sum(abs.(envelope)) * dt_seg
    scale = (θ / γ) / area

    I = clamp.(envelope .* scale * 0.3f0, -RF_BOUND*0.8f0, RF_BOUND*0.8f0)
    Q = zeros(Float32, length(I))
    return I, Q
end

RF_I_h, RF_Q_h = init_rf_gaussian(NUM_TIME_SEGMENTS)
seg_I_dev = adapt(backend, similar(RF_I_h))
seg_Q_dev = adapt(backend, similar(RF_Q_h))


@inline function cubic_weights_scalar(α::Float32)
    t=α
    t2=t*t
    t3=t2*t

    w0 = -0.5f0*t + t2 - 0.5f0*t3
    w1 = 1.0f0   - 2.5f0*t2 + 1.5f0*t3
    w2 = 0.5f0*t + 2.0f0*t2 - 1.5f0*t3
    w3 = -0.5f0*t2 + 0.5f0*t3

    return w0, w1, w2, w3
end

function precompute_catmull_rom(Nt::Int, Nseg::Int, dt::Float32, inv_seg_dur::Float32)
    i0 = Array{Int32}(undef, Nt)
    i1 = similar(i0)
    i2 = similar(i0)
    i3 = similar(i0)

    w0 = Array{Float32}(undef, Nt)
    w1 = similar(w0)
    w2 = similar(w0)
    w3 = similar(w0)

    @inbounds for ksample in 1:Nt
        t = Float32(ksample-1)*dt
        s = clamp(t*inv_seg_dur, 0.0f0, Float32(Nseg - 1) - 1f-6)
        k = clamp(Int(floor(s)), 1, Nseg - 3)
        α = Float32(s - k)

        i0[ksample] = Int32(k)
        i1[ksample] = Int32(k+1)
        i2[ksample] = Int32(k+2)
        i3[ksample] = Int32(k+3)

        w0[ksample], w1[ksample], w2[ksample], w3[ksample] = cubic_weights_scalar(α)
    end
    return i0, i1, i2, i3, w0, w1, w2, w3
end

i0_h, i1_h, i2_h, i3_h, w0_h, w1_h, w2_h, w3_h = precompute_catmull_rom(Nt, NUM_TIME_SEGMENTS, dt, INV_SEG_DUR)
d_i0 = adapt(backend, i0_h)
d_i1 = adapt(backend, i1_h)
d_i2 = adapt(backend, i2_h)
d_i3 = adapt(backend, i3_h)
d_w0 = adapt(backend, w0_h)
d_w1 = adapt(backend, w1_h)
d_w2 = adapt(backend, w2_h)
d_w3 = adapt(backend, w3_h)

@kernel function upsample_kernel!(B1r_t, B1i_t, I, Q, i0,i1,i2,i3, w0,w1,w2,w3)
    k = Int(Tuple(@index(Global))[1])
    B1r_t[k] = w0[k]*I[i0[k]] + w1[k]*I[i1[k]] + w2[k]*I[i2[k]] + w3[k]*I[i3[k]]
    B1i_t[k] = w0[k]*Q[i0[k]] + w1[k]*Q[i1[k]] + w2[k]*Q[i2[k]] + w3[k]*Q[i3[k]]
end

@kernel function adjoint_kernel!(gI, gQ, dB1r_t, dB1i_t, i0,i1,i2,i3, w0,w1,w2,w3, scale::Float32)
    k = Int(Tuple(@index(Global))[1])

    gr = dB1r_t[k]*scale
    gi = dB1i_t[k]*scale

    Atomix.@atomic gI[Int(i0[k])] += w0[k]*gr
    Atomix.@atomic gI[Int(i1[k])] += w1[k]*gr
    Atomix.@atomic gI[Int(i2[k])] += w2[k]*gr
    Atomix.@atomic gI[Int(i3[k])] += w3[k]*gr
    Atomix.@atomic gQ[Int(i0[k])] += w0[k]*gi
    Atomix.@atomic gQ[Int(i1[k])] += w1[k]*gi
    Atomix.@atomic gQ[Int(i2[k])] += w2[k]*gi
    Atomix.@atomic gQ[Int(i3[k])] += w3[k]*gi
end

function device_upsample!(B1r_t, B1i_t, I_dev, Q_dev)
    upsample_kernel!(backend, GROUP_SIZE)(B1r_t, B1i_t, I_dev, Q_dev,
        d_i0,d_i1,d_i2,d_i3, d_w0,d_w1,d_w2,d_w3; ndrange=Nt)
end

function device_adjoint_to_segments!(gI_dev, gQ_dev, dB1r_t, dB1i_t; scale=dt)
    fill!(gI_dev, 0f0); fill!(gQ_dev, 0f0)
    adjoint_kernel!(backend, GROUP_SIZE)(gI_dev, gQ_dev, dB1r_t, dB1i_t,
        d_i0,d_i1,d_i2,d_i3, d_w0,d_w1,d_w2,d_w3, Float32(scale); ndrange=Nt)
end

@kernel inbounds=true function excitation_kernel!(
    M_xy::AbstractVector{T}, M_z::AbstractVector{T},
    p_x::AbstractVector{T}, p_y::AbstractVector{T}, p_z::AbstractVector{T},
    p_ΔBz::AbstractVector{T}, p_T1::AbstractVector{T}, p_T2::AbstractVector{T}, p_ρ::AbstractVector{T},
    s_Gx::AbstractVector{T}, s_Gy::AbstractVector{T}, s_Gz::AbstractVector{T},
    s_Δt::AbstractVector{T}, s_Δf::AbstractVector{T}, s_B1r::AbstractVector{T}, s_B1i::AbstractVector{T},
    N_Spins::Int32, N_Δt::Int32
) where {T}
    i = Int(Tuple(@index(Global))[1])
    if i <= Int(N_Spins)
        N   = Int(N_Spins)
        ir  = i
        ii  = i + N
        x   = p_x[i]
        y = p_y[i]
        z = p_z[i]
        ΔBz = p_ΔBz[i]
        ρ   = p_ρ[i]
        T1  = p_T1[i]
        T2  = p_T2[i]
        Mxy_r = M_xy[ir]
        Mxy_i = M_xy[ii]
        Mz    = M_z[i]
        s_idx = 1
        @inbounds while s_idx <= Int(N_Δt)
            gx  = s_Gx[s_idx]
            gy = s_Gy[s_idx]
            gz = s_Gz[s_idx]
            Δt  = s_Δt[s_idx]
            df  = s_Δf[s_idx]
            b1r = s_B1r[s_idx]
            b1i = s_B1i[s_idx]
            Bz = (x*gx + y*gy + z*gz) + ΔBz - df / T(γ)
            B  = sqrt(b1r*b1r + b1i*b1i + Bz*Bz)
            ϕ  = T(γ) * B * Δt
            sϕ = sin(ϕ)
            cϕ = cos(ϕ)
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

    k = excitation_kernel!(backend)
    k(M_xy, M_z,
      p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
      s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1r, s_B1i,
      N_Spins, N_Δt;
      ndrange=Int(N_Spins),
      workgroupsize=GROUP_SIZE)
    return nothing
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

@kernel function mag_loss_grad_M!(
    dM::AbstractVector{Float32}, M::AbstractVector{Float32},
    in_core::AbstractVector{Float32}, in_stop::AbstractVector{Float32},
    λ_core::Float32, λ_stop::Float32, target_mag::AbstractVector{Float32}
)
    i = Int(Tuple(@index(Global))[1])
    N = length(in_core)
    if i <= N
        ir = i
        ii = i + N
        Mx = M[ir]
        My = M[ii]
        mag2 = Mx*Mx + My*My
        tgt  = target_mag[i]
        mag  = sqrt(mag2 + 1f-12)

        n_core = max(sum(in_core), 1f-12)
        n_stop = max(sum(in_stop), 1f-12)

        if in_core[i] > 0.5f0
            e = (mag - tgt)
            s = (λ_core / n_core) * (e / max(mag, 1f-12)) 
            dM[ir] = s * Mx
            dM[ii] = s * My
        elseif in_stop[i] > 0.5f0
            s = (2f0 * λ_stop) / n_stop                  
            dM[ir] = s * Mx
            dM[ii] = s * My
        else
            dM[ir] = 0f0; dM[ii] = 0f0
        end
    end
end

function gpu_loss_mag(M_xy,
                      in_core, in_stop, target_mag;
                      λ_core::Float32 = 1f0, λ_stop::Float32 = 1f0)
    N  = length(in_core)
    Mx = view(M_xy, 1:N)
    My = view(M_xy, N+1:2N)

    mag = sqrt.(Mx .* Mx .+ My .* My .+ 1f-12)

    n_core = CUDA.sum(in_core) + 1f-12
    n_stop = CUDA.sum(in_stop) + 1f-12

    loss_core = λ_core * CUDA.sum(((mag .- target_mag).^2) .* in_core) / n_core
    loss_stop = λ_stop * CUDA.sum((Mx .* Mx .+ My .* My) .* in_stop) / n_stop
    return Float32(loss_core + loss_stop)
end

struct Batch
    NSC::Int
    z_scen::Vector{Vector{Float32}}
    in_core_dev::Vector             
    in_stop_dev::Vector
    target_dev::Vector             
end

function make_scenarios(NSC::Int, jitter::Float32)
    zs_host = Vector{Vector{Float32}}(undef, NSC)
    core_dev = Vector{Any}(undef, NSC)
    stop_dev = Vector{Any}(undef, NSC)
    tgt_dev  = Vector{Any}(undef, NSC)

    for s in 1:NSC
        pert = (rand(Float32, Nspins) .- 0.5f0) .* (2f0 * position_offset_range * jitter)
        z = p_z_base_h .+ pert
        zs_host[s] = z
        in_core_h, in_stop_h, tgt_h = make_target_and_masks(z)
        core_dev[s] = adapt(backend, in_core_h)
        stop_dev[s] = adapt(backend, in_stop_h)
        tgt_dev[s]  = adapt(backend, tgt_h)
    end
    return Batch(NSC, zs_host, core_dev, stop_dev, tgt_dev)
end

function add_rf_regs!(gI::AbstractVector{Float32}, gQ::AbstractVector{Float32},
                      I::AbstractVector{Float32}, Q::AbstractVector{Float32})
    n = length(I)
    @. gI += (2.0f0*λ_power/n) * I
    @. gQ += (2.0f0*λ_power/n) * Q

    if n > 1 && λ_smooth != 0.0f0
        gI[1] += λ_smooth*(I[1]-I[2])

        for k in 2:n-1
            gI[k] += λ_smooth*(2.0f0*I[k] - I[k-1] - I[k+1])
        end

        gI[n] += λ_smooth*(I[n]-I[n-1])
        gQ[1] = λ_smooth*(Q[1]-Q[2])

        for k in 2:n-1
            gQ[k] += λ_smooth*(2.0f0*Q[k] - Q[k-1] - Q[k+1])
        end

        gQ[n] += λ_smooth*(Q[n]-Q[n-1])
    end
    return nothing
end

function rf_reg_value(I::AbstractVector{Float32}, Q::AbstractVector{Float32})
    n = length(I)
    pwr = λ_power/Float32(n) * (sum(@. I*I) + sum(@. Q*Q))
    smi = 0.0f0
    smq = 0.0f0

    @inbounds for k in 1:n-1
        di = I[k] - I[k+1]
        dq = Q[k] - Q[k+1]
        smi += di*di
        smq += dq*dq
    end

    sm = 0.5f0*λ_smooth*(smi + smq)
    return pwr + sm
end

# temps
dM_xy_buf = similar(M_xy)
dM_z_buf = similar(M_z)
tmp_dB1r = similar(s_B1r)
tmp_dB1i = similar(s_B1i)
∇B1r = similar(s_B1r)
∇B1i = similar(s_B1i)
gI_dev = adapt(backend, zeros(Float32, NUM_TIME_SEGMENTS))
gQ_dev = adapt(backend, zeros(Float32, NUM_TIME_SEGMENTS))

function forward_excitation!(z_host, in_core_dev, in_stop_dev, target_dev)
    copyto!(p_z, z_host)
    device_upsample!(s_B1r, s_B1i, seg_I_dev, seg_Q_dev)

    fill!(M_xy, 0.0f0)
    fill!(M_z, 1.0f0)
    excite_only!(M_xy, M_z,
        p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
        s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1r, s_B1i,
        backend)

    KernelAbstractions.synchronize(backend)
    return gpu_loss_mag(M_xy, in_core_dev, in_stop_dev, target_dev)
end

function loss_only_batched!(θ::Vector{Float32}, batch::Batch)
    copyto!(seg_I_dev, 1, θ, 1, NUM_TIME_SEGMENTS)
    copyto!(seg_Q_dev, 1, θ, NUM_TIME_SEGMENTS+1, NUM_TIME_SEGMENTS)

    Lacc = 0.0f0
    @inbounds for s in 1:batch.NSC
        Lacc += forward_excitation!(batch.z_scen[s], batch.in_core_dev[s], batch.in_stop_dev[s], batch.target_dev[s])
    end
    return Lacc / Float32(batch.NSC) + rf_reg_value(view(θ, 1:NUM_TIME_SEGMENTS),
                                                    view(θ, NUM_TIME_SEGMENTS+1:2NUM_TIME_SEGMENTS))
end

function loss_and_grad_batched!(θ::Vector{Float32}, batch::Batch)
    copyto!(seg_I_dev, 1, θ, 1, NUM_TIME_SEGMENTS)
    copyto!(seg_Q_dev, 1, θ, NUM_TIME_SEGMENTS+1, NUM_TIME_SEGMENTS)

    fill!(∇B1r, 0f0)
    fill!(∇B1i, 0f0)
    Lacc = 0.0f0

    @inbounds for s in 1:batch.NSC
        Ls = forward_excitation!(batch.z_scen[s], batch.in_core_dev[s], batch.in_stop_dev[s], batch.target_dev[s])
        Lacc += Ls

        fill!(dM_xy_buf, 0f0)
        mag_loss_grad_M!(backend, GROUP_SIZE)(
            dM_xy_buf, M_xy, batch.in_core_dev[s], batch.in_stop_dev[s], 1f0, 1f0, batch.target_dev[s];
            ndrange=Int(Nspins))

        fill!(tmp_dB1r, 0f0)
        fill!(tmp_dB1i, 0f0)
        fill!(dM_z_buf, 0f0)
        Enzyme.autodiff(Enzyme.Reverse, excite_only!,
            Enzyme.Duplicated(M_xy, dM_xy_buf),
            Enzyme.Duplicated(M_z,  dM_z_buf),
            Enzyme.Const(p_x), Enzyme.Const(p_y), Enzyme.Const(p_z),
            Enzyme.Const(p_ΔBz), Enzyme.Const(p_T1), Enzyme.Const(p_T2), Enzyme.Const(p_ρ),
            Enzyme.Const(s_Gx), Enzyme.Const(s_Gy), Enzyme.Const(s_Gz),
            Enzyme.Const(s_Δt),
            Enzyme.Const(s_Δf),
            Enzyme.Duplicated(s_B1r, tmp_dB1r),
            Enzyme.Duplicated(s_B1i, tmp_dB1i),
            Enzyme.Const(backend),
        )
        KernelAbstractions.synchronize(backend)

        @. ∇B1r += tmp_dB1r
        @. ∇B1i += tmp_dB1i
    end

    invNSC = 1f0 / Float32(batch.NSC)
    @. ∇B1r *= invNSC
    @. ∇B1i *= invNSC
    L_data = Lacc * invNSC

    device_adjoint_to_segments!(gI_dev, gQ_dev, ∇B1r, ∇B1i; scale=dt)
    gI = Array(gI_dev)
    gQ = Array(gQ_dev)

    Iview = view(θ, 1:NUM_TIME_SEGMENTS)
    Qview = view(θ, NUM_TIME_SEGMENTS+1:2NUM_TIME_SEGMENTS)
    add_rf_regs!(gI, gQ, Iview, Qview)
    L = L_data + rf_reg_value(Iview, Qview)

    return L, vcat(gI, gQ)
end

@inline function project_box!(θ::Vector{Float32}, bnd::Float32)
    @inbounds @simd for i in eachindex(θ)
        x = θ[i]
        θ[i] = ifelse(x >  bnd, bnd, ifelse(x < -bnd, -bnd, x))
    end
    return θ
end

@inline function cap_inf!(dθ::Vector{Float32}, maxstep::Float32)
    m = maximum(abs, dθ)
    if m > maxstep
        s = maxstep / m
        @inbounds @simd for i in eachindex(dθ)
            dθ[i] *= s
        end
    end
    return dθ
end

function run_spg_bb!(θ::Vector{Float32}, batch::Batch;
    max_iters::Int=150, m_hist::Int=10, c::Float32=1f-4, beta::Float32=0.5f0,
    step_cap::Float32=0.15f0*RF_BOUND, α_min::Float32=1f-6, α_max::Float32=1f3,
    grad_tol::Float32=1f-6, loss_tol_rel::Float32=5f-4, patience::Int=15)

    L, g = loss_and_grad_batched!(θ, batch)

    loss_win = fill(L, m_hist)
    win_len  = 1
    θ_prev = copy(θ)
    g_prev = copy(g)
    α = 1f-2
    best_L = L
    stall  = 0

    losses = Vector{Float32}(undef, max_iters)
    @info "Starting optimization" max_iters=max_iters m_hist=m_hist

    dθ = similar(θ)
    θ_trial = similar(θ)

    for it in 1:max_iters
        gnorm = maximum(abs, g)
        losses[it] = L
        if gnorm <= grad_tol
            @info "Converged (grad_tol)" it=it L=L grad=gnorm
            return θ, L, losses[1:it]
        end

        sTy = 0.0f0
        yTy = 0.0f0
        @inbounds @simd for i in eachindex(θ)
            s = θ[i] - θ_prev[i]
            y = g[i] - g_prev[i]
            sTy += s*y
            yTy += y*y
        end
        α = (yTy > 0f0 && sTy > 0f0) ? clamp(sTy / yTy, α_min, α_max) : clamp(1f-2 / (1f0 + gnorm), α_min, α_max)

        # Trial step
        @inbounds @simd for i in eachindex(θ) 
            dθ[i] = -α * g[i] 
        end
        cap_inf!(dθ, step_cap)

        @inbounds @simd for i in eachindex(θ) 
            θ_trial[i] = θ[i] + dθ[i] 
        end
        project_box!(θ_trial, RF_BOUND)

        # Armijo backtrack
        gTd = 0.0f0
        @inbounds @simd for i in eachindex(θ) 
            gTd += g[i] * (θ_trial[i] - θ[i]) 
        end

        f_ref = maximum(view(loss_win, 1:win_len))
        n_back = 0
        L_trial = loss_only_batched!(θ_trial, batch)
        
        while L_trial > f_ref + c * gTd && n_back < 8
            @inbounds @simd for i in eachindex(dθ) 
                dθ[i] *= beta 
            end

            cap_inf!(dθ, step_cap)
            @inbounds @simd for i in eachindex(θ_trial) 
                θ_trial[i] = θ[i] + dθ[i] 
            end

            project_box!(θ_trial, RF_BOUND)
            gTd = 0.0f0
            @inbounds @simd for i in eachindex(θ) 
                gTd += g[i] * (θ_trial[i] - θ[i]) 
            end

            L_trial = loss_only_batched!(θ_trial, batch)
            n_back += 1
        end

        L_trial, g_trial = loss_and_grad_batched!(θ_trial, batch)

        # Accept
        θ_prev .= θ
        g_prev .= g
        θ      .= θ_trial
        L       = L_trial
        g       = g_trial

        # nonmonotone window
        if win_len < m_hist
            win_len += 1
            loss_win[win_len] = L
        else
            @inbounds @simd for i in 1:m_hist-1
                loss_win[i] = loss_win[i+1]
            end
            loss_win[m_hist] = L
        end

        # Early stop
        if L < best_L * (1f0 - loss_tol_rel)
            best_L = L
            stall = 0
        else
            stall += 1
            if stall >= patience
                @info "Early stop (loss plateau)" it=it best_L=best_L current_L=L patience=patience
                return θ, L, losses[1:it]
            end
        end

        if (it % 10 == 0) || (it <= 5)
            @info "iter" it=it L=L grad=gnorm α=α backtracks=n_back
        end
    end

    @info "Training complete (SPG-BB)" final_loss=L
    return θ, L, losses
end

function final_profile_mag!(Ih, Qh)
    copyto!(seg_I_dev, Ih)
    copyto!(seg_Q_dev, Qh)
    copyto!(p_z, p_z_base_h)
    device_upsample!(s_B1r, s_B1i, seg_I_dev, seg_Q_dev)

    fill!(M_xy, 0.0f0)
    fill!(M_z, 1.0f0)
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

function plot_results(RF_Ih, RF_Qh; title_rf="Final RF", title_prof="Slice profile", path=nothing)
    copyto!(seg_I_dev, RF_Ih)
    copyto!(seg_Q_dev, RF_Qh)
    device_upsample!(s_B1r, s_B1i, seg_I_dev, seg_Q_dev)

    t = collect(0:dt:(Nt-1)*dt)
    z_mm = 1.0f3 .* p_z_base_h
    prof_mag = final_profile_mag!(RF_Ih, RF_Qh)
    tgt_plot = butterworth_target(p_z_base_h; cutoff=BW_CUTOFF, n=BW_ORDER)

    plt = plot(layout=(2,1), size=(900,800))
    plot!(plt[1], t, Array(s_B1r); label="B1r", xlabel="time (s)", ylabel="B1", title=title_rf, lw=2)
    plot!(plt[1], t, Array(s_B1i); label="B1i", lw=2)
    plot!(plt[2], z_mm, tgt_plot; label="Target", xlabel="z (mm)", ylabel="|M_xy|",
          title=title_prof, lw=3, ls=:dash)
    plot!(plt[2], z_mm, prof_mag; label="Achieved", lw=3)
    if path !== nothing
        savefig(plt, path)
    end
    return plt
end

function save_training_history_png(losses; path="training_history.png", ttl="Loss")
    p = plot(1:length(losses), losses; lw=2, xlabel="iter", ylabel="loss", title=ttl, size=(900,350))
    savefig(p, path)
end

const NSC = 3
batch = make_scenarios(NSC, 1.0f0)

θ0 = Float32.(vcat(RF_I_h, RF_Q_h))

timed = @timed run_spg_bb!(copy(θ0), batch;
    max_iters=150, m_hist=10, loss_tol_rel=5f-4, patience=15)
θ_exc, L_exc, losses_exc = timed.value
t_sec = timed.time
@info "Optimization time" seconds=t_sec minutes=t_sec/60

RF_I_h .= θ_exc[1:NUM_TIME_SEGMENTS]
RF_Q_h .= θ_exc[NUM_TIME_SEGMENTS+1:2NUM_TIME_SEGMENTS]
@info "Excitation complete" loss=L_exc

save_training_history_png(losses_exc; path="training_history.png", ttl="Loss")
plot_results(RF_I_h, RF_Q_h; path="final_rf_profile.png")

in_core_base, in_stop_base, target_base = make_target_and_masks(p_z_base_h)
prof_exc = final_profile_mag!(RF_I_h, RF_Q_h)
core = prof_exc[in_core_base .> 0.5f0]; stop = prof_exc[in_stop_base .> 0.5f0]
@info "Excitation Passband" mean=mean(core) ripple=(maximum(core)-minimum(core))
@info "Excitation Stopband" max=maximum(stop) mean=mean(stop)

println("Saved: training_history.png, final_rf_profile.png")