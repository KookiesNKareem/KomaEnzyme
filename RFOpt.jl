using KomaMRICore, Suppressor
using Random: seed!
using KernelAbstractions
using KernelAbstractions: @kernel, @index
using Adapt
using CUDA
import Enzyme
using Statistics: mean
using LinearAlgebra: norm
using Plots

seed!(42)

const B1   = 4.9e-6
const Trf  = 3.2e-3
const TBP  = 8.0
const Δz   = 6e-3
const zmax = 8e-3
const fmax = TBP / Trf
const γf64 = 2π * 42.57747892e6       # rad/(s·T)
const γ    = Float32(γf64)

const Nspins = 512
const GROUP_SIZE = 512

z  = collect(range(-zmax, zmax, length=Nspins))
Gz = fmax / (Float64(γf64) * Δz)

sys = Scanner()
seq_full = PulseDesigner.RF_sinc(B1, Trf, sys; G=[Gz; 0; 0], TBP=TBP)
seq = seq_full
obj = Phantom(; x=z)

const Nrf = length(seq.RF[1].A)

sim_params = KomaMRICore.default_sim_params()
sim_params["Δt_rf"] = Trf / (2*(Nrf - 1)) 
sim_params["Δt"] = Inf                 
sim_params["return_type"] = "state"
sim_params["precision"] = "f32"
sim_params["Nthreads"] = 1
sim_params["sim_method"] = KomaMRICore.Bloch()

mag_sinc = @suppress simulate(obj, seq, sys; sim_params)

## Target
butterworth_degree = 5
target_profile = 0.5im ./ (1 .+ (z ./ (Δz / 2)).^(2*butterworth_degree))
const TARGET_R_h = Float32.(real.(target_profile))
const TARGET_I_h = Float32.(imag.(target_profile))
const TARGET_MAG = Float32.(abs.(target_profile))

const backend = CUDA.CUDABackend()

function get_koma_timeline(seq, sim_params)
    seqd = KomaMRICore.discretize(seq; sampling_params=sim_params)
    Nt = length(seqd.Δt)
    B1i = ComplexF32.(seqd.B1[1:Nt])
    Gxi = Float32.(seqd.Gx[1:Nt])
    Gyi = Float32.(seqd.Gy[1:Nt])
    Gzi = Float32.(seqd.Gz[1:Nt])
    Δti = Float32.(seqd.Δt[1:Nt])
    Δfi = Float32.(seqd.Δf[1:Nt])    
    ti  = Float32.(seqd.t[1:Nt])
    rf_active_idx = findall(x -> abs(x) > 1e-10, B1i)
    return (B1=B1i, Gx=Gxi, Gy=Gyi, Gz=Gzi, Δt=Δti, Δf=Δfi, t=ti,
            rf_active_idx=rf_active_idx, Nt=Int32(Nt), Nrf_original=Nrf)
end

const TL = get_koma_timeline(seq, sim_params)
const rf_idx = TL.rf_active_idx
const Lrf_taps = length(rf_idx)

const KX_RESID = sum(TL.Gx .* TL.Δt)     

const N = length(z)
const N_Spins32 = Int32(N)
const N_Δt32 = TL.Nt

adapt_dev(x) = CUDA.adapt(CuArray, x)

M_xy = adapt_dev(zeros(Float32, 2N)) 
M_z = adapt_dev(ones(Float32, N))

p_x = adapt_dev(Float32.(z))
p_y = adapt_dev(zeros(Float32, N))
p_z = adapt_dev(zeros(Float32, N))
p_ΔBz = adapt_dev(zeros(Float32, N))
p_T1 = adapt_dev(fill(Float32(1e9), N))
p_T2 = adapt_dev(fill(Float32(1e9), N))
p_ρ = adapt_dev(ones(Float32, N))

s_Gx = adapt_dev(TL.Gx)
s_Gy = adapt_dev(TL.Gy)
s_Gz = adapt_dev(TL.Gz)
s_Δt = adapt_dev(TL.Δt)
s_Δf = adapt_dev(TL.Δf)

s_B1r = adapt_dev(real.(TL.B1))
s_B1i = adapt_dev(imag.(TL.B1))

target_r_dev = adapt_dev(TARGET_R_h)
target_i_dev = adapt_dev(TARGET_I_h)

B1r_host = copy(real.(TL.B1))
B1i_host = copy(imag.(TL.B1))

@kernel inbounds=true function excitation_kernel!(
    M_xy::AbstractVector{T}, M_z::AbstractVector{T},
    p_x::AbstractVector{T}, p_y::AbstractVector{T}, p_z::AbstractVector{T},
    p_ΔBz::AbstractVector{T}, p_T1::AbstractVector{T}, p_T2::AbstractVector{T}, p_ρ::AbstractVector{T},
    s_Gx::AbstractVector{T}, s_Gy::AbstractVector{T}, s_Gz::AbstractVector{T},
    s_Δt::AbstractVector{T}, s_Δf::AbstractVector{T}, s_B1r::AbstractVector{T}, s_B1i::AbstractVector{T},
    N_Spins::Int, N_Δt::Int
) where {T}
    eps_d = T(1f-20)
    i = @index(Global, Linear)  # 1-based

    if i <= N_Spins
        N  = N_Spins
        ir = i
        ii = i + N

        x   = p_x[ir]; y = p_y[ir]; z = p_z[ir]
        ΔBz = p_ΔBz[ir]; ρ = p_ρ[ir]
        T1  = p_T1[ir];  T2 = p_T2[ir]

        Mx  = M_xy[ir];  My = M_xy[ii];  Mz = M_z[ir]

        @inbounds for s_idx in 1:N_Δt
            gx  = s_Gx[s_idx]; gy = s_Gy[s_idx]; gz = s_Gz[s_idx]
            Δt  = s_Δt[s_idx]; df = s_Δf[s_idx]
            b1r = s_B1r[s_idx]; b1i = s_B1i[s_idx]

            Bz = (x*gx + y*gy + z*gz) + ΔBz - df / T(γ)

            B   = sqrt(b1r*b1r + b1i*b1i + Bz*Bz)
            φ   = T(-π) * T(γ) * B * Δt

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
        end

        M_xy[ir] = Mx
        M_xy[ii] = My
        M_z[ir]  = Mz
    end
end

function launch_excitation!(
    M_xy, M_z,
    p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
    s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1r, s_B1i,
    N_Spins::Int, N_Δt::Int, backend)
    k = excitation_kernel!(backend)
    wgs = min(GROUP_SIZE, N_Spins)
    k(M_xy, M_z,
      p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
      s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1r, s_B1i,
      N_Spins, N_Δt; ndrange=N_Spins, workgroupsize=wgs)
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
        Int(N_Spins32), Int(N_Δt32), backend)
    return nothing
end

@kernel function phase_loss!(
    dM::AbstractVector{Float32}, M::AbstractVector{Float32},
    target_r::AbstractVector{Float32}, target_i::AbstractVector{Float32},
    cosφ::AbstractVector{Float32},  sinφ::AbstractVector{Float32}, λ::Float32
)
    i = Int(Tuple(@index(Global))[1])
    N = length(target_r)
    if i <= N
        ir = i
        ii = i + N

        Mr = M[ir]
        Mi = M[ii]
        Tr = target_r[i]
        Ti = target_i[i]
        c = cosφ[i]
        s = sinφ[i]

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
    xh = Float32.(z)
    acc_r = 0.0f0
    acc_i = 0.0f0

    @inbounds for i in 1:N
        θ = κ * xh[i]
        c = cos(θ)
        s = sin(θ)

        Mr_dem =  Mr[i]*c + Mi[i]*s
        Mi_dem =  Mi[i]*c - Mr[i]*s
        Tr = TARGET_R_h[i]
        Ti = TARGET_I_h[i]
        acc_r +=  Tr*Mr_dem + Ti*Mi_dem
        acc_i +=  Tr*Mi_dem - Ti*Mr_dem
    end

    φ0 = atan(acc_i, acc_r)
    cosφ = similar(Mr)
    sinφ = similar(Mr)
    c0 = cos(φ0)
    s0 = sin(φ0)

    @inbounds for i in 1:N
        θ = κ * xh[i]
        cθ = cos(θ)
        sθ = sin(θ)

        cosφ[i] = c0*cθ - s0*sθ
        sinφ[i] = s0*cθ + c0*sθ
    end
    return cosφ, sinφ
end

const n_ctrl = Nrf
const n_taps = Lrf_taps
const ctrl_pos = collect(range(0f0, 1f0, length=n_ctrl))
const tap_pos = collect(range(0f0, 1f0, length=n_taps))

function x_to_timeline!(B1r::Vector{Float32}, B1i::Vector{Float32}, x::Vector{Float32})
    @assert length(x) == n_ctrl
    B1r .= real.(TL.B1)
    fill!(B1i, 0f0)
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

function grad_to_x!(∇x::Vector{Float32}, dB1r_timeline::AbstractVector{Float32})
    @assert length(dB1r_timeline) == length(TL.Δt)
    resize!(∇x, n_ctrl)
    fill!(∇x, 0f0)
    wsum = zeros(Float32, n_ctrl)

    j = 1
    @inbounds for (k, idx) in enumerate(rf_idx)
        t   = tap_pos[k]
        Δtw = TL.Δt[idx]         
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
            ∇x[j] += w0 * dB1r_timeline[idx]
            wsum[j] += w0
            ∇x[j+1] += w1 * dB1r_timeline[idx]
            wsum[j+1] += w1
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
    x_to_timeline!(B1r_host, B1i_host, x)
    copyto!(s_B1r, B1r_host)
    copyto!(s_B1i, B1i_host)

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

dM_xy = similar(M_xy)
dM_z  = similar(M_z)
∇B1r  = similar(s_B1r)
∇B1i  = similar(s_B1i)

function loss_and_grad!(∇x::Vector{Float32}, x::Vector{Float32})
    L, cφ = forward!(x)
    fill!(dM_xy, 0f0)
    phase_loss!(backend, GROUP_SIZE)(
        dM_xy, M_xy, target_r_dev, target_i_dev,
        adapt_dev(cφ.cosφ), adapt_dev(cφ.sinφ), 1f0; ndrange=Int(N))
    CUDA.synchronize()

    fill!(∇B1r, 0f0)
    fill!(∇B1i, 0f0)
    fill!(dM_z, 0f0)
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

    dB1r_host = Array(∇B1r)
    grad_to_x!(∇x, dB1r_host)
    return L
end

x = zeros(Float32, n_ctrl)
∇x = similar(x)
L0 = loss_and_grad!(∇x, x)

nsteps  = 100
η_base  = 5f-10
rf_clip = 2f-5

t_start = time_ns()
for k in 1:nsteps
    Lk = loss_and_grad!(∇x, x)
    g_rms = sqrt(mean(abs2, ∇x)) + 1f-20
    η = min(η_base, 1f-6 / (10f0 * g_rms))
    @. x = clamp(x - η * ∇x, -rf_clip, rf_clip)
    @info "Step $k: L=$Lk, η=$(round(η, sigdigits=3)), |g|_rms=$(round(g_rms, sigdigits=3))"
end
t_total = (time_ns() - t_start) / 1e9
@info "Total loop time: $(round(t_total, digits=3)) s ($(round(t_total/60, digits=2)) min)"

function plot_rf_and_profile(x)
    x_to_timeline!(B1r_host, B1i_host, x)
    copyto!(s_B1r, B1r_host)
    fill!(s_B1i, 0f0)
    fill!(M_xy, 0f0)
    fill!(M_z, 1f0)

    launch_excitation!(
        M_xy, M_z,
        p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
        s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1r, s_B1i,
        Int(N_Spins32), Int(N_Δt32), backend)
    KernelAbstractions.synchronize(backend)

    Mr = Array(view(M_xy, 1:N))
    Mi = Array(view(M_xy, N+1:2N))
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

plt = plot_rf_and_profile(x)
savefig(plt, "profile_and_rf.png")