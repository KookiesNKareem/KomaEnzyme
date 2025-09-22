using StaticArrays, LinearAlgebra, Random, Statistics
using Plots
import Enzyme
using CUDA
using Printf
using KernelAbstractions

include("miniKomaCore.jl") 

# Physical and Simulation Constants
const TOTAL_TIME = 4.0f-3
const DT = 8.0f-5
const NSTEPS = Int(cld(TOTAL_TIME, DT))
const NUM_TIME_SEGMENTS = 50
const INV_SEG_DUR = NUM_TIME_SEGMENTS / TOTAL_TIME

const MAX_GZ = 10f-3
const MAX_SLEW = 70f0
const RF_BOUND = 12f-6

const GAMMA = 2f0π * 42.58f6
const PARAMS = (γ=GAMMA, T1=1f0, T2=1f0, M0=1f0)

# Spatial Configuration
const DELTA = 4.0f-3
const Z_FOV = 14.0f-3
const NZ = 121
const Z_BASE = Float32.(collect(range(-Z_FOV/2, Z_FOV/2; length=NZ)))

const TRANSITION_FRAC = 0f0
const DELTA_CORE = DELTA * (1 - TRANSITION_FRAC)

const VOXEL_RESOLUTION = Z_FOV / (NZ - 1)
const POSITION_OFFSET_RANGE = VOXEL_RESOLUTION / 2

# GPU Configuration
const GROUP_SIZE = 256

Random.seed!(42)

# Gradient Functions
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
    
    @inbounds begin
        bx = I[i1]*(1-α) + I[i2]*α
        by = Q[i1]*(1-α) + Q[i2]*α
    end
    return bx, by
end


function generate_masks(z_positions)
    in_roi = (@. -DELTA/2 <= z_positions <= DELTA/2)
    in_core = (@. -DELTA_CORE/2 <= z_positions <= DELTA_CORE/2)
    in_trans = in_roi .& .!in_core
    in_stop = .!in_roi
    
    target_mag = zeros(Float32, length(z_positions))
    target_mag[in_core] .= 1f0
    target_mag[in_trans] .= NaN32
    
    return in_roi, in_core, in_trans, in_stop, target_mag
end

# Create GPU context for kernel
function make_kernel_context()
    M_xy = CUDA.zeros(Float32, 2NZ)
    M_z = CUDA.zeros(Float32, NZ)
    
    _, in_core_bool, _, in_stop_bool, target_mag = generate_masks(Z_BASE)
    
    # Convert masks to Float32 once
    in_core = Float32.(in_core_bool)
    in_stop = Float32.(in_stop_bool)

    # Phantom
    obj = (
        p_x = CUDA.zeros(Float32, NZ),
        p_y = CUDA.zeros(Float32, NZ),
        p_z = cu(Z_BASE),
        p_ΔBz = CUDA.zeros(Float32, NZ),
        p_T1 = CUDA.fill(PARAMS.T1, NZ),
        p_T2 = CUDA.fill(PARAMS.T2, NZ),
        p_ρ = CUDA.fill(PARAMS.M0, NZ)
    )

    # Base sequence components - create once
    tgrid = collect(0f0:DT:((NSTEPS-1)*DT))
    gz_values = Float32[gz_at(t) for t in tgrid]
    
    seq = (
        s_Gx = CUDA.zeros(Float32, NSTEPS),
        s_Gy = CUDA.zeros(Float32, NSTEPS),
        s_Gz = cu(gz_values),
        s_Δt = CUDA.fill(DT, NSTEPS),
        s_Δf = CUDA.zeros(Float32, NSTEPS),
        s_B1 = CUDA.zeros(Float32, NSTEPS*2),
        in_core = cu(in_core),
        in_stop = cu(in_stop)
    )

    # Target
    target = CUDA.zeros(Float32, 2NZ)
    target[1:NZ] = cu(target_mag)

    return (; M_xy, M_z, obj, seq, target, in_core=cu(in_core), in_stop=cu(in_stop))
end

# Global kernel context
const KCTX = make_kernel_context()
const IN_ROI_BASE, IN_CORE_BASE, IN_TRANS_BASE, IN_STOP_BASE, TARGET_MAG_BASE = generate_masks(Z_BASE)

# CPU Simulation Functions
@inline function rot_apply(m::SVector{3}, Ω::SVector{3})
    θ = norm(Ω)
    iszero(θ) && return m
    n = Ω / θ
    c, s = sincos(θ)
    return m*c + cross(n, m)*s + n*(dot(n, m))*(1-c)
end

@inline function bloch_step(dt, t, m, I, Q, z, p)
    bx_n, by_n = rf_at(t, I, Q)
    gz_n = gz_at(t)
    bx_n1, by_n1 = rf_at(t+dt, I, Q)
    gz_n1 = gz_at(t+dt)

    B_n = SVector{3}(bx_n, by_n, gz_n * z)
    B_n1 = SVector{3}(bx_n1, by_n1, gz_n1 * z)

    A_n = -p.γ * B_n
    A_n1 = -p.γ * B_n1
    Ω1 = (dt/2) * (A_n + A_n1)
    Ω2 = (dt*dt/12) * cross(A_n1, A_n)
    Ω = Ω1 + Ω2

    m_rot = rot_apply(m, Ω)
    relax = SVector{3}(-m[1]/p.T2*dt, -m[2]/p.T2*dt, -(m[3]-p.M0)/p.T1*dt)
    return m_rot + relax
end

function simulate_profile(I, Q; z_positions=Z_BASE)
    T = promote_type(eltype(I), eltype(Q), Float32)
    Mx = Vector{T}(undef, NZ)
    My = Vector{T}(undef, NZ)
    Mz = Vector{T}(undef, NZ)
    
    @inbounds for iz in 1:NZ
        mm = SVector{3}(zero(T), zero(T), T(PARAMS.M0))
        t = zero(T)
        zz = z_positions[iz]
        for _ in 1:NSTEPS
            mm = bloch_step(DT, t, mm, I, Q, T(zz), PARAMS)
            t += DT
        end
        Mx[iz] = mm[1]
        My[iz] = mm[2]
        Mz[iz] = mm[3]
    end
    return Mx, My, Mz
end

function init_iq(nseg; Ttot=TOTAL_TIME, tbw=3.0, flip_deg=90.0)
    tgrid = range(0f0, Ttot, length=nseg)
    tmid = Ttot/2

    σ = Ttot / (2.5f0 * tbw)
    s = @. exp(-0.5f0 * ((tgrid - tmid) / σ)^2)

    θ = deg2rad(flip_deg)
    dt_seg = Ttot / (nseg - 1)
    area = sum(abs, s) * dt_seg
    scale = (θ / PARAMS.γ) / area

    I = @. clamp(s * scale, -RF_BOUND*0.8f0, RF_BOUND*0.8f0)
    Q = zeros(Float32, length(I))
    return I, Q
end

# Kernel caller
function excitation_caller!(M_xy, M_z, obj, seq, x, backend)
    nthreads = cld(length(M_z), GROUP_SIZE) * GROUP_SIZE
    x_reshaped = reshape(x, 2, (length(x) ÷ 2))
    excitation_kernel!(backend, GROUP_SIZE)(
        M_xy, M_z, 
        obj.p_x, obj.p_y, obj.p_z, obj.p_ΔBz, obj.p_T1, obj.p_T2, obj.p_ρ, length(obj.p_x),
        seq.s_Gx, seq.s_Gy, seq.s_Gz, seq.s_Δt, seq.s_Δf, x_reshaped, length(seq.s_Δt);
        ndrange = nthreads
    )
    KernelAbstractions.synchronize(backend)
end

function loss_with_rf_first(x, M_xy, M_z, obj, seq, target, hyperparams, backend)
    copyto!(seq.s_B1, x)
    excitation_caller!(M_xy, M_z, obj, seq, backend)
    
    T = eltype(M_xy)
    Nspins = length(M_z)
    n = length(x) ÷ 2
    
    λ_profile = hyperparams.λ_profile
    λ_smooth = hyperparams.λ_smooth
    λ_power = hyperparams.λ_power
    λ_profile_tv = hyperparams.λ_profile_tv
    
    # Use views for clarity
    Mx = @view M_xy[1:Nspins]
    My = @view M_xy[Nspins+1:2*Nspins]
    target_x = @view target[1:Nspins]
    target_y = @view target[Nspins+1:2*Nspins]
    I = @view x[1:n]
    Q = @view x[n+1:end]
    
    # Vectorized GPU operations
    dmx = Mx .- target_x
    dmy = My .- target_y
    pass_err = dmx.^2 .+ dmy.^2
    mag_squared = Mx.^2 .+ My.^2
    
    # Use ifelse instead of type conversion for masks
    # This avoids the problematic broadcast conversion
    pass_masked = pass_err .* ifelse.(seq.in_core, one(T), zero(T))
    stop_masked = mag_squared .* ifelse.(seq.in_stop, one(T), zero(T))
    
    # GPU-friendly reductions
    pass_loss = sum(pass_masked)
    core_count = sum(ifelse.(seq.in_core, one(T), zero(T)))
    stop_loss = sum(stop_masked)
    stop_count = sum(ifelse.(seq.in_stop, one(T), zero(T)))
    
    # TV of profile
    if Nspins > 1
        mag_diff = @view(mag_squared[2:end]) .- @view(mag_squared[1:end-1])
        tv_profile_acc = sum(abs, mag_diff)
    else
        tv_profile_acc = zero(T)
    end
    
    # RF regularization
    rf_power_acc = sum(abs2, I) + sum(abs2, Q)
    
    # TV for RF
    if n > 1
        dI = @view(I[2:end]) .- @view(I[1:end-1])
        dQ = @view(Q[2:end]) .- @view(Q[1:end-1])
        tv_rf_acc = sum(abs2, dI) + sum(abs2, dQ)
    else
        tv_rf_acc = zero(T)
    end
    
    # Final loss computation
    eps_val = eps(T)
    loss = λ_profile * pass_loss / (core_count + eps_val) +
           2 * λ_profile * stop_loss / (stop_count + eps_val) +
           λ_profile_tv * tv_profile_acc +
           λ_smooth * tv_rf_acc +
           λ_power * rf_power_acc / T(2n)
    
    return loss
end

# Optimization
function optimize_adam(x0;
    max_iters=200,
    lr=5e-6,
    lr_decay=0.995,
    β1=0.9, β2=0.999, ϵ=1e-8,
    grad_clip=5e2,
    patience=50,
    min_delta=1e-6,
    rel_tol=1e-4,
    plot_every=1,
    save_plots=true,
    obj_params = (λ_profile=200f0, λ_smooth=1f10, λ_power=10000f0, λ_profile_tv=5f0, target_mag=1f0)
)
    gr()
    
    x = Float32.(x0)
    m = zeros(Float32, length(x))
    v = zeros(Float32, length(x))
    
    x_dev = cu(copy(x0))
    x̄ = similar(x_dev)
    Mxȳ = similar(KCTX.M_xy)
    Mz̄ = similar(KCTX.M_z)
    g = similar(x)
    
    best_loss = Inf32
    best_x = copy(x)
    best_iter = 0
    patience_counter = 0
    
    loss_hist = Float32[]
    plots_array = []
    
    for t in 1:max_iters
        # Copy to GPU (no allocation)
        copyto!(x_dev, x)
        
        # Clear gradients
        fill!(x̄, 0)
        fill!(Mxȳ, 0)
        fill!(Mz̄, 0)
        
        # Combined forward and backward pass
        res = Enzyme.autodiff(
            Enzyme.set_runtime_activity(Enzyme.ReverseWithPrimal),
            loss_with_rf_first,
            Enzyme.Active,
            Enzyme.Duplicated(x_dev, x̄),
            Enzyme.Duplicated(KCTX.M_xy, Mxȳ),
            Enzyme.Duplicated(KCTX.M_z, Mz̄),
            Enzyme.Const(KCTX.obj),
            Enzyme.Const(KCTX.seq),
            Enzyme.Const(KCTX.target),
            Enzyme.Const(obj_params),
            Enzyme.Const(CUDABackend())
        )
        
        # Extract the primal (loss value) from the result
        loss_val = res[2]
        
        # Copy gradient back to CPU
        copyto!(g, x̄)
        
        push!(loss_hist, loss_val)
        
        # Gradient clipping
        gnorm = norm(g)
        if gnorm > grad_clip
            rmul!(g, grad_clip / gnorm)
            gnorm = grad_clip
        end

        # Adam update
        @. m = β1 * m + (1 - β1) * g
        @. v = β2 * v + (1 - β2) * g^2
        mhat = @. m / (1 - β1^t)
        vhat = @. v / (1 - β2^t)

        lr_t = lr * lr_decay^t
        @. x -= lr_t * mhat / (sqrt(vhat) + ϵ)
        @. x = clamp(x, -RF_BOUND, RF_BOUND)

        # Track best
        improvement = best_loss - loss_val
        if improvement > min_delta || abs(improvement / max(abs(best_loss), 1e-10)) > rel_tol
            best_loss = loss_val
            best_x .= x
            best_iter = t
            patience_counter = 0
        else
            patience_counter += 1
        end

        # Plotting
        if t % plot_every == 0
            n = length(x) ÷ 2
            I_curr = x[1:n]
            Q_curr = x[n+1:end]
            p = plot_progress(I_curr, Q_curr, t, loss_val)
            display(p)
            save_plots && push!(plots_array, p)
            
            println("iter $t: loss=$(round(loss_val,sigdigits=5)) |g|=$(round(gnorm,sigdigits=3)) " *
                   "lr=$(round(lr_t,sigdigits=3)) best=$(round(best_loss,sigdigits=5))@$best_iter " *
                   "patience=$patience_counter/$patience")
        end

        # Early stopping
        if patience_counter >= patience
            println("Early stopping at iter $t")
            break
        end
        if t > 50 && gnorm < 1e-7
            println("Converged at iter $t")
            break
        end
    end

    return best_loss < loss_hist[end] ? best_x : x, loss_hist, plots_array, []
end

function plot_progress(I, Q, iter, loss_val)
    Mx, My, Mz = simulate_profile(I, Q)
    mag = hypot.(Mx, My)

    t_rf = range(0, TOTAL_TIME, length=length(I)) * 1e3

    p = plot(layout=(1,2), size=(900, 400))
    
    plot!(p[1], t_rf, I*1e6, label="Bx (I)", lw=2)
    plot!(p[1], t_rf, Q*1e6, label="By (Q)", lw=2, alpha=0.7)
    hline!(p[1], [RF_BOUND*1e6, -RF_BOUND*1e6], ls=:dot, alpha=0.5, label="")
    plot!(p[1], xlabel="Time (ms)", ylabel="RF (μT)",
          title="iter $iter | loss $(round(loss_val, sigdigits=4))", legend=:topright)

    plot!(p[2], Z_BASE*1e3, mag, label="|Mxy|", lw=2)
    plot!(p[2], Z_BASE*1e3, TARGET_MAG_BASE, label="Target", lw=2, ls=:dash)
    vspan!(p[2], [-DELTA*1e3/2, -DELTA_CORE*1e3/2], alpha=0.12, color=:yellow, label="")
    vspan!(p[2], [DELTA_CORE*1e3/2, DELTA*1e3/2], alpha=0.12, color=:yellow, label="")
    plot!(p[2], xlabel="z (mm)", ylabel="|Mxy|", title="Slice Profile", 
          ylims=(-0.1, 1.2), legend=:topright)

    return p
end

function run_optim(; max_iters=150, lr=1e-6, patience=40, plot_every=1)
    println("Initializing RF profile...")
    I0, Q0 = init_iq(NUM_TIME_SEGMENTS)
    x0 = vcat(I0, Q0)

    println("Starting optimization with Adam...")
    x_opt, L_hist, P_list, M_hist = optimize_adam(
        x0; max_iters, lr, lr_decay=0.998, patience,
        min_delta=1e-6, rel_tol=1e-4, grad_clip=1e2, plot_every,
        save_plots=true
    )

    return (xopt=x_opt, loss_hist=L_hist, plots=P_list, metrics=M_hist)
end

# Main execution
println("Running optimization...")
@time best = run_optim(max_iters=150, lr=2f-6, patience=40, plot_every=1)

xopt = best.xopt
loss_hist = best.loss_hist

# Final results
n = length(xopt) ÷ 2
Iopt = xopt[1:n]
Qopt = xopt[n+1:end]

Mx, My, Mz = simulate_profile(Iopt, Qopt)
mag = hypot.(Mx, My)
phase = atan.(My, Mx)
mag_core = mag[IN_CORE_BASE]
mag_stop = mag[IN_STOP_BASE]

println("\nFINAL PERFORMANCE METRICS")
println("Optimization iters: $(length(loss_hist))")
println("Final loss: $(round(loss_hist[end], sigdigits=5))")
println("Passband mean: $(round(mean(mag_core), digits=3))")
println("Passband ripple: $(round(maximum(mag_core) - minimum(mag_core), digits=3))")
println("Max stopband: $(round(maximum(mag_stop), digits=4))")
println("Mean stopband: $(round(mean(mag_stop), digits=4))")
println("Phase RMS (core): $(round(sqrt(mean(phase[IN_CORE_BASE].^2)), digits=3)) rad")
println("Max RF amplitude: $(round(maximum(hypot.(Iopt, Qopt))*1e6, digits=2)) μT")

p_final = plot_progress(Iopt, Qopt, length(loss_hist), loss_hist[end])
display(p_final)
savefig(p_final, "final_optimized_profile.png")