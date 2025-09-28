import CUDA
using KernelAbstractions
using Adapt
import Enzyme
using Random
using Plots
using Printf
using Statistics

CUDA.allowscalar(false)
ENV["GKSwstype"] = "100"

const T = Float32
const GROUP_SIZE = 256
const Nspins = 201
const Nt     = 256
const dt     = T(8e-6)
const γ      = T(2π) * T(42.58e6)
const N_Spins32 = Int32(Nspins)
const N_Δt32    = Int32(Nt)
const T1 = T(1.0)
const T2 = T(1.0)
const ρ0 = T(1.0)

const Gz = T(10e-3)
const zFOV   = T(0.05)
const zgrid  = range(-zFOV/2, zFOV/2; length=Nspins) |> collect .|> T
const thick  = T(0.01)
const passband = (-thick/2, thick/2)

const p_cap  = T(2)
const tv_eps = T(1e-9)
const rf_bound = T(30e-6)                # allow a wider hard box (was 20e-6)
const TRF = T(Nt)*dt
const EPS_MAG = T(1e-6)                  # |M| floor in loss-grad

const backend = CUDA.CUDABackend()
Random.seed!(42)

mutable struct Regs{T}
    λ_rf::T
    λ_cap::T
    λ_smooth::T
    λ_tv::T
    cap_B1::T
end

# ↑ raise the soft cap to let optimizer find a stronger solution
make_regs_warmup(::Type{T}) where {T} = Regs(T(0), T(0), T(0), T(0), T(20e-6))
make_regs_final(::Type{T})  where {T} = Regs(T(5e0), T(5e3), T(1e3), T(0), T(20e-6))

@inline function pack_reim(v::AbstractVector{Complex{T}}) where {T}
    N = length(v); out = Vector{T}(undef, 2N)
    @inbounds for i in 1:N
        out[i] = real(v[i]); out[i+N] = imag(v[i])
    end
    out
end

@inline function unpack_reim(v2::AbstractVector{T}) where {T}
    N = length(v2) ÷ 2; out = Vector{Complex{T}}(undef, N)
    @inbounds for i in 1:N
        out[i] = complex(v2[i], v2[i+N])
    end
    out
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
        N  = Int(N_Spins); ir = i; ii = i + N
        x = p_x[i]; y = p_y[i]; z = p_z[i]
        ΔBz = p_ΔBz[i]; ρ = p_ρ[i]; T1 = p_T1[i]; T2 = p_T2[i]
        Mxy_r = M_xy[ir]; Mxy_i = M_xy[ii]; Mz = M_z[i]
        s_idx = 1
        @inbounds while s_idx <= Int(N_Δt)
            gx = s_Gx[s_idx]; gy = s_Gy[s_idx]; gz = s_Gz[s_idx]
            Δt = s_Δt[s_idx]; df = s_Δf[s_idx]
            b1r = s_B1r[s_idx]; b1i = s_B1i[s_idx]
            Bz = (x*gx + y*gy + z*gz) + ΔBz - (T(2π) * df) / T(γ)
            B  = sqrt(b1r*b1r + b1i*b1i + Bz*Bz)
            ϕ  = T(-1) * T(γ) * B * Δt
            sϕ = sin(ϕ); cϕ = cos(ϕ)
            denom = max(B, T(1e-12))
            α_r =  cϕ
            α_i = -(Bz/denom) * sϕ
            β_r =  (b1i/denom) * sϕ
            β_i = -(b1r/denom) * sϕ
            Mxy_new_r = 2T(1) * (Mxy_i * (α_r*α_i - β_r*β_i) + Mz * (α_i*β_i + α_r*β_r)) +
                        Mxy_r * (α_r*α_r - α_i*α_i - β_r*β_r + β_i*β_i)
            Mxy_new_i = -2T(1) * (Mxy_r * (α_r*α_i + β_r*β_i) - Mz * (α_r*β_i - α_i*β_r)) +
                         Mxy_i * (α_r*α_r - α_i*α_i + β_r*β_r - β_i*β_i)
            Mz_new = Mz * (α_r*α_r + α_i*α_i - β_r*β_r - β_i*β_i) -
                     2T(1) * (Mxy_r * (α_r*β_r - α_i*β_i) + Mxy_i * (α_r*β_i + α_i*β_r))
            ΔT1 = exp(-Δt / T1)
            ΔT2 = exp(-Δt / T2)
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

@inline function excitation_caller!(
    M_xy, M_z,
    p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
    s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1r, s_B1i,
    N_Spins::Int32, N_Δt::Int32, backend
)
    k = excitation_kernel!(backend, GROUP_SIZE)
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
    backend
)
    excitation_caller!(M_xy, M_z,
        p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
        s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1r, s_B1i,
        N_Spins32, N_Δt32, backend)
    return nothing
end

@kernel function demod_M!(Md::AbstractVector{T}, M::AbstractVector{T}, z::AbstractVector{T}, γ::T, Gz::T, TRF::T) where {T}
    i = @index(Global, Linear)
    N = length(z)
    if i <= N
        ϕ = γ * Gz * z[i] * TRF
        c = cos(ϕ); s = sin(ϕ)
        ir = i; ii = i+N
        Mx = M[ir]; My = M[ii]
        Md[ir] = c*Mx - s*My
        Md[ii] = s*Mx + c*My
    end
end

@kernel function remod_grad!(dM::AbstractVector{T}, dMd::AbstractVector{T}, z::AbstractVector{T}, γ::T, Gz::T, TRF::T) where {T}
    i = @index(Global, Linear)
    N = length(z)
    if i <= N
        ϕ = γ * Gz * z[i] * TRF
        c = cos(ϕ); s = sin(ϕ)
        ir = i; ii = i+N
        gx = dMd[ir]; gy = dMd[ii]
        dM[ir] =  c*gx + s*gy
        dM[ii] = -s*gx + c*gy
    end
end

# ---------- Weighted |M| loss ----------
@kernel function loss_grad_Mmag_weighted!(dM, M, TGTMAG, W, floor::T) where {T}
    i = @index(Global, Linear)
    N = length(TGTMAG)
    if i <= N
        ir = i; ii = i + N
        @inbounds begin
            Mx = M[ir]; My = M[ii]
            mag = sqrt(Mx*Mx + My*My)
            mag_eff = max(mag, floor)
            diff = (mag - TGTMAG[i])
            w = W[i]
            scale = w * (2f0 / N) * diff / mag_eff
            dM[ir] = scale * Mx
            dM[ii] = scale * My
        end
    end
end

function weighted_mag_loss(Md_xy::CUDA.CuArray{T}, target_mag::CUDA.CuArray{T}, W::CUDA.CuArray{T}) where {T<:AbstractFloat}
    N = length(target_mag)
    @views Mx = Md_xy[1:N]; My = Md_xy[N+1:end]
    mags = sqrt.(Mx .* Mx .+ My .* My .+ T(0))
    CUDA.sum(W .* (mags .- target_mag).^2) / T(N)
end

@inline function reset_ics!(M_xy, M_z, p_ρ)
    M_xy .= zero(eltype(M_xy))
    M_z  .= p_ρ
    return nothing
end

rf_penalty(reg::Regs, s_B1r, s_B1i) = reg.λ_rf * dt * (CUDA.sum(s_B1r .* s_B1r) + CUDA.sum(s_B1i .* s_B1i))
rf_softcap_penalty(reg::Regs, s_B1r, s_B1i) = begin
    excess_r = max.(abs.(s_B1r) .- reg.cap_B1, zero(T))
    excess_i = max.(abs.(s_B1i) .- reg.cap_B1, zero(T))
    reg.λ_cap * dt * (CUDA.sum(excess_r .^ p_cap) + CUDA.sum(excess_i .^ p_cap))
end
@views function rf_smooth_l2_penalty(reg::Regs, s_B1r, s_B1i)
    dr = s_B1r[2:end] .- s_B1r[1:end-1]
    di = s_B1i[2:end] .- s_B1i[1:end-1]
    reg.λ_smooth * dt * (CUDA.sum(dr .* dr) + CUDA.sum(di .* di))
end
@views function rf_tv_penalty(reg::Regs, s_B1r, s_B1i)
    if reg.λ_tv == 0; return zero(eltype(s_B1r)); end
    dr = s_B1r[2:end] .- s_B1r[1:end-1]
    di = s_B1i[2:end] .- s_B1i[1:end-1]
    reg.λ_tv * dt * (CUDA.sum(sqrt.(dr .* dr .+ tv_eps)) + CUDA.sum(sqrt.(di .* di .+ tv_eps)))
end
total_penalty(reg::Regs, s_B1r, s_B1i) =
    rf_penalty(reg, s_B1r, s_B1i) + rf_softcap_penalty(reg, s_B1r, s_B1i) +
    rf_smooth_l2_penalty(reg, s_B1r, s_B1i) + rf_tv_penalty(reg, s_B1r, s_B1i)

@inline function add_rf_softcap_grad!(reg::Regs, ∇B1r, ∇B1i, s_B1r, s_B1i)
    abs_r   = abs.(s_B1r); excessr = max.(abs_r .- reg.cap_B1, zero(T))
    sgn_r   = ifelse.(s_B1r .>= 0, one(T), -one(T))
    mask_r  = ifelse.(excessr .> 0, one(T), zero(T))
    @. ∇B1r += reg.λ_cap * dt * p_cap * (excessr .+ eps(T)).^(p_cap - one(T)) * sgn_r * mask_r
    abs_i   = abs.(s_B1i); excessi = max.(abs_i .- reg.cap_B1, zero(T))
    sgn_i   = ifelse.(s_B1i .>= 0, one(T), -one(T))
    mask_i  = ifelse.(excessi .> 0, one(T), zero(T))
    @. ∇B1i += reg.λ_cap * dt * p_cap * (excessi .+ eps(T)).^(p_cap - one(T)) * sgn_i * mask_i
    return nothing
end
@views function add_rf_smooth_l2_grad!(reg::Regs, ∇, s)
    N = length(s); N <= 1 && return nothing
    dr = s[2:end] .- s[1:end-1]
    tmp = similar(s); fill!(tmp, zero(eltype(s)))
    tmp[1:end-1] .-= dr; tmp[2:end] .+= dr
    @. ∇ += (2f0 * reg.λ_smooth * dt) * tmp
    return nothing
end
@views function add_rf_tv_grad!(reg::Regs, ∇, s)
    reg.λ_tv == 0 && return nothing
    N = length(s); N <= 1 && return nothing
    dr = s[2:end] .- s[1:end-1]
    w  = dr ./ sqrt.(dr .* dr .+ tv_eps)
    tmp = similar(s); fill!(tmp, zero(eltype(s)))
    tmp[1:end-1] .-= w; tmp[2:end] .+= w
    @. ∇ += reg.λ_tv * dt * tmp
    return nothing
end

function forward_only!(M_xy, M_z,
    p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
    s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1r, s_B1i, backend)
    reset_ics!(M_xy, M_z, p_ρ)
    excite_only!(M_xy, M_z,
        p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
        s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1r, s_B1i, backend)
    KernelAbstractions.synchronize(backend)
    return nothing
end

function grad_rf!(
    ∇B1r::CUDA.CuArray{T}, ∇B1i::CUDA.CuArray{T},
    M_xy::CUDA.CuArray{T}, M_z::CUDA.CuArray{T},
    p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
    s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1r, s_B1i,
    target_mag::CUDA.CuArray{T}, W::CUDA.CuArray{T},
    backend
) where {T<:AbstractFloat}
    reset_ics!(M_xy, M_z, p_ρ)
    excite_only!(M_xy, M_z,
        p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
        s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1r, s_B1i,
        backend)
    KernelAbstractions.synchronize(backend)

    Md_xy = similar(M_xy)
    demod_M!(backend, GROUP_SIZE)(Md_xy, M_xy, p_z, γ, Gz, TRF; ndrange=Nspins)
    KernelAbstractions.synchronize(backend)

    dMd_xy = similar(Md_xy)
    loss_grad_Mmag_weighted!(backend, GROUP_SIZE)(dMd_xy, Md_xy, target_mag, W, EPS_MAG; ndrange=length(target_mag))
    KernelAbstractions.synchronize(backend)

    dM_xy = similar(M_xy)
    remod_grad!(backend, GROUP_SIZE)(dM_xy, dMd_xy, p_z, γ, Gz, TRF; ndrange=Nspins)
    KernelAbstractions.synchronize(backend)

    dM_z = similar(M_z);  fill!(dM_z, zero(T))
    fill!(∇B1r, zero(T)); fill!(∇B1i, zero(T))

    Enzyme.autodiff(Enzyme.Reverse, excite_only!,
        Enzyme.Duplicated(M_xy, dM_xy),
        Enzyme.Duplicated(M_z,  dM_z),
        Enzyme.Const(p_x), Enzyme.Const(p_y), Enzyme.Const(p_z),
        Enzyme.Const(p_ΔBz), Enzyme.Const(p_T1), Enzyme.Const(p_T2), Enzyme.Const(p_ρ),
        Enzyme.Const(s_Gx), Enzyme.Const(s_Gy), Enzyme.Const(s_Gz),
        Enzyme.Const(s_Δt),
        Enzyme.Const(s_Δf),
        Enzyme.Duplicated(s_B1r, ∇B1r),
        Enzyme.Duplicated(s_B1i, ∇B1i),
        Enzyme.Const(backend),
    )
    KernelAbstractions.synchronize(backend)

    Ldata = weighted_mag_loss(Md_xy, target_mag, W)
    return Ldata
end

# ----------------- Alloc & init -----------------
M_xy = adapt(backend, zeros(T, 2Nspins))
p_rho = adapt(backend, fill(ρ0, Nspins))
M_z  = copy(p_rho)
p_x   = adapt(backend, zeros(T, Nspins))
p_y   = adapt(backend, zeros(T, Nspins))
p_z   = adapt(backend, zgrid)
p_ΔBz = adapt(backend, zeros(T, Nspins))
p_T1  = adapt(backend, fill(T1, Nspins))
p_T2  = adapt(backend, fill(T2, Nspins))
p_ρ   = p_rho
s_Gx  = adapt(backend, zeros(T, Nt))
s_Gy  = adapt(backend, zeros(T, Nt))
s_Gz  = adapt(backend, fill(Gz, Nt))
s_Δt  = adapt(backend, fill(dt, Nt))
s_Δf  = adapt(backend, zeros(T, Nt))

# Time-domain RF init with correct slice bandwidth
t_cont = ((-(Nt-1)/2):((Nt-1)/2)) .* Float64(dt)
Δf_hz  = 42.58e6 * Float64(Gz) * Float64(thick)
rf_base = sinc.(t_cont .* Δf_hz)
w = 0.54 .- 0.46 .* cos.(2π .* (0:Nt-1) ./ (Nt-1))
rf0 = T(12e-6) .* T.(w .* rf_base)               # start closer to soft cap
s_B1r = adapt(backend, rf0)
s_B1i = adapt(backend, fill(T(0), Nt))

# Targets and weights (heavier in passband)
target_mag_h = T[]
weight_h = T[]
@inbounds for zi in zgrid
    inpb = (passband[1] ≤ zi ≤ passband[2])
    push!(target_mag_h, inpb ? T(0.5) : T(0))
    push!(weight_h,     inpb ? T(3.0) : T(1.0))
end
target_mag = adapt(backend, target_mag_h)
Wz         = adapt(backend, weight_h)

function total_loss_current(reg::Regs, M_xy, s_B1r, s_B1i, target_mag, Wz)
    Md_xy = similar(M_xy)
    demod_M!(backend, GROUP_SIZE)(Md_xy, M_xy, p_z, γ, Gz, TRF; ndrange=Nspins)
    KernelAbstractions.synchronize(backend)
    Ldata = Float64(weighted_mag_loss(Md_xy, target_mag, Wz))
    Lreg  = Float64(total_penalty(reg, s_B1r, s_B1i))
    Ldata + Lreg, Ldata, Lreg
end

function optimize_rf!(
    s_B1r::CUDA.CuArray{T}, s_B1i::CUDA.CuArray{T};
    iters::Int=250, η::T=T(2e-5), clip::T=rf_bound,
    betas::Tuple{T,T}=(T(0.9),T(0.999)), eps::T=T(1e-8),
    gn_clip::T=T(2e3), fps::Int=12, reg_warmup::Int=80
) where {T<:AbstractFloat}
    reg_final = make_regs_final(T)
    reg = make_regs_warmup(T)

    ∇B1r = similar(s_B1r); ∇B1i = similar(s_B1i)
    m_r  = similar(s_B1r); v_r  = similar(s_B1r); fill!(m_r,0); fill!(v_r,0)
    m_i  = similar(s_B1i); v_i  = similar(s_B1i); fill!(m_i,0); fill!(v_i,0)
    losses = Vector{Float64}(undef, iters)
    gradn  = Vector{Float64}(undef, iters)
    anim = Animation()
    t_ms = collect(0:Nt-1) .* Float64(dt) .* 1e3
    z_mm = Array(zgrid) .* 1e3
    target_curve = Array(target_mag_h)
    β1, β2 = betas
    β1pow = one(T); β2pow = one(T)
    no_clip_iters = 80
    base_gn_clip  = T(1e5)                      # much higher ceiling

    function step_and_eval!(ηloc, mhat_r, vhat_r, mhat_i, vhat_i)
        @. s_B1r = s_B1r - ηloc * mhat_r / (sqrt(vhat_r) + eps)
        @. s_B1i = s_B1i - ηloc * mhat_i / (sqrt(vhat_i) + eps)
        # AdamW-style decoupled L2
        wd = reg.λ_rf * dt
        if wd != 0
            @. s_B1r = s_B1r * (1 - ηloc * wd)
            @. s_B1i = s_B1i * (1 - ηloc * wd)
        end
        s_B1r .= min.(max.(s_B1r, -clip), clip)
        s_B1i .= min.(max.(s_B1i, -clip), clip)
        forward_only!(M_xy, M_z, p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
                      s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1r, s_B1i, backend)
        total_loss_current(reg, M_xy, s_B1r, s_B1i, target_mag, Wz)
    end

    Ldata = grad_rf!(∇B1r, ∇B1i,
                     M_xy, M_z,
                     p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
                     s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1r, s_B1i,
                     target_mag, Wz,
                     backend)

    for k in 1:iters
        if k ≤ reg_warmup
            τ = T(k)/T(reg_warmup)
            τcap = τ * sqrt(τ)  # τ^1.5 in-type
            reg = Regs( τ    * reg_final.λ_rf,
                        τcap * reg_final.λ_cap,
                        τ    * reg_final.λ_smooth,
                        τ    * reg_final.λ_tv,
                        reg_final.cap_B1)
        else
            reg = reg_final
        end

        Ldata = grad_rf!(∇B1r, ∇B1i,
                         M_xy, M_z,
                         p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
                         s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1r, s_B1i,
                         target_mag, Wz,
                         backend)

        add_rf_softcap_grad!(reg, ∇B1r, ∇B1i, s_B1r, s_B1i)
        add_rf_smooth_l2_grad!(reg, ∇B1r, s_B1r)
        add_rf_smooth_l2_grad!(reg, ∇B1i, s_B1i)
        add_rf_tv_grad!(reg, ∇B1r, s_B1r)
        add_rf_tv_grad!(reg, ∇B1i, s_B1i)

        # gradient clipping (disabled first 80 iters, higher later)
        gn_limit = (k <= no_clip_iters) ? T(Inf) : base_gn_clip
        gnr = sqrt(Float64(CUDA.sum(∇B1r .* ∇B1r) + CUDA.sum(∇B1i .* ∇B1i)))
        if gnr > gn_limit
            scale = T(gn_limit / gnr)
            @. ∇B1r *= scale
            @. ∇B1i *= scale
            gnr = sqrt(Float64(CUDA.sum(∇B1r .* ∇B1r) + CUDA.sum(∇B1i .* ∇B1i)))
        end

        @. m_r = β1 * m_r + (1 - β1) * ∇B1r
        @. m_i = β1 * m_i + (1 - β1) * ∇B1i
        @. v_r = β2 * v_r + (1 - β2) * (∇B1r * ∇B1r)
        @. v_i = β2 * v_i + (1 - β2) * (∇B1i * ∇B1i)
        β1pow *= β1;  β2pow *= β2
        mhat_r = m_r ./ (1 .- β1pow); mhat_i = m_i ./ (1 .- β1pow)
        vhat_r = v_r ./ (1 .- β2pow); vhat_i = v_i ./ (1 .- β2pow)

        # constant LR (no cosine decay)
        ηloc = η

        L_before, _, _ = total_loss_current(reg, M_xy, s_B1r, s_B1i, target_mag, Wz)
        L_try, Lmag_try, Lreg_try = step_and_eval!(ηloc, mhat_r, vhat_r, mhat_i, vhat_i)
        tries = 0
        while L_try > L_before && tries < 6
            # revert step and reduce gently
            @. s_B1r = s_B1r + ηloc * mhat_r / (sqrt(vhat_r) + eps)
            @. s_B1i = s_B1i + ηloc * mhat_i / (sqrt(vhat_i) + eps)
            ηloc *= T(0.7)
            L_try, Lmag_try, Lreg_try = step_and_eval!(ηloc, mhat_r, vhat_r, mhat_i, vhat_i)
            tries += 1
        end

        losses[k] = L_try
        gradn[k]  = gnr

        rf_r = Array(s_B1r); rf_i = Array(s_B1i)
        Md_xy_plot_dev = similar(M_xy)
        demod_M!(backend, GROUP_SIZE)(Md_xy_plot_dev, M_xy, p_z, γ, Gz, TRF; ndrange=Nspins)
        KernelAbstractions.synchronize(backend)
        prof_mag = abs.(unpack_reim(Array(Md_xy_plot_dev)))

        p1 = plot(t_ms, rf_r, label="B1 real (T)", xlabel="Time (ms)", ylabel="B1 (T)", lw=2, title="RF (iter $k)")
        plot!(p1, t_ms, rf_i, label="B1 imag (T)", lw=2, ls=:dash, legend=:topright)
        p2 = plot(1:k, losses[1:k], lw=2, xlabel="Iteration", ylabel="Total Loss", title="Loss")
        p3 = plot(z_mm, prof_mag, lw=2, label="|Mxy| (demod)", xlabel="z (mm)", ylabel="Mag", title="Slice Profile")
        plot!(p3, z_mm, target_curve, lw=2, ls=:dash, label="Target")
        plt = plot(p1, p2, p3; layout=(3,1), size=(900,900))
        frame(anim, plt)

        if k == 1 || k % 10 == 0
            _, Ld, Lr = total_loss_current(reg, M_xy, s_B1r, s_B1i, target_mag, Wz)
            @info "iter $k  L=$(round(losses[k]; sigdigits=6))  Ldata=$(round(Ld; sigdigits=6))  Lreg=$(round(Lr; sigdigits=6))  ‖∇RF‖₂=$(round(gnr; sigdigits=6))  η=$(round(Float64(ηloc); sigdigits=6))"
        end
    end
    gif(anim, "rf_opt.gif"; fps=fps)
    return (losses, gradn)
end

println("Starting RF optimization (box slice)...")
losses, gradn = optimize_rf!(s_B1r, s_B1i; iters=250, η=T(2e-5), fps=12, reg_warmup=80)

final_Ldata = grad_rf!(similar(s_B1r), similar(s_B1i),
                       M_xy, M_z,
                       p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
                       s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1r, s_B1i,
                       target_mag, Wz,
                       backend)
final_loss = Float64(final_Ldata + total_penalty(make_regs_final(T), s_B1r, s_B1i))
println("\nFinal total loss = ", final_loss)

t_ms = collect(0:Nt-1) .* Float64(dt) .* 1e3
rf_r = Array(s_B1r); rf_i = Array(s_B1i)
Md_xy_final_dev = similar(M_xy)
demod_M!(backend, GROUP_SIZE)(Md_xy_final_dev, M_xy, p_z, γ, Gz, TRF; ndrange=Nspins)
KernelAbstractions.synchronize(backend)
Mxy_final = unpack_reim(Array(Md_xy_final_dev)); prof_mag = abs.(Mxy_final)
target_curve = target_mag_h; z_mm = Array(zgrid) .* 1e3
p1 = plot(t_ms, rf_r, label="B1 real (T)", xlabel="Time (ms)", ylabel="B1 (T)", lw=2, title="Optimized RF")
plot!(p1, t_ms, rf_i, label="B1 imag (T)", lw=2, ls=:dash)
p2 = plot(1:length(losses), losses, lw=2, xlabel="Iteration", ylabel="Total Loss", title="Loss vs Iteration")
p3 = plot(z_mm, prof_mag, lw=2, label="|Mxy| (final, demod)", xlabel="z (mm)", ylabel="Mag", title="Slice Profile")
plot!(p3, z_mm, target_curve, lw=2, ls=:dash, label="Target")
savefig(plot(p1,p2,p3; layout=(3,1), size=(900,900)), "rf_final.png")
println("\nSaved animation rf_opt.gif and final plot rf_final.png")
