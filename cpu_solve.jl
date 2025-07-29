using StaticArrays
using LinearAlgebra
using DifferentiationInterface
using Enzyme
using Plots

target = rand(3)
x = [0.1, 0.3, 0.2]
params = (
    γ = 42.58e6,
    B = 1e-6,      
    T1 = 1.0,     
    T2 = 0.5,     
    M0 = 1.0,   
)
dt = 0.001 
struct ForwardEuler
end

function bloch(m, params)
    γ, B, T1, T2, M0 = params
    cross_term = γ * cross(m, [0, 0, B])
    relaxation_term = [m[1] / T2, m[2] / T2, (m[3] - M0) / T1]
    dM = cross_term - relaxation_term
    return dM
end

function step(dt, m, params, ::ForwardEuler)
    dM = bloch(m, params)
    m_next = m + dM .* dt
    return m_next
end

function solve(m0, dt, tmax, params, method)
    Nsteps = Int(round(tmax / dt))
    mt = zeros(3, Nsteps + 1)
    mt[:, 1] = m0
    for i in 1:Nsteps
        mt[:, i + 1] = step(dt, mt[:, i], params, method)
    end
    return mt
end

f(x) = sum(abs2, step(dt, x, params, ForwardEuler()) .- target)

function solve_objective(x)
    mt = solve(x, dt, 1.0, params, ForwardEuler())
    final_state = mt[:, end] 
    return sum(abs2, final_state .- target)
end

function gradient_descent_step(x, t, objective)
    value, grad = value_and_gradient(objective, AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Forward)), x)
    x_new = x .- t .* grad
    return x_new, value
end

function optimize(x, step_size, iters, objective)
    println("Starting optimization...")
    println("Initial state: ", x)
    println("Target state: ", target)
    
    loss = zeros(iters)
    distances = zeros(iters)
    x_path = zeros(3, iters + 1)
    x_path[:, 1] = x 
    
    for i in 1:iters 
        x, loss[i] = gradient_descent_step(x, step_size, objective)
        x_path[:, i + 1] = x
        distances[i] = norm(x - target)
        
        # if i % 50 == 0 || i <= 5 
        #     println("Iteration $i: Loss = $(loss[i]), x = $x")
        #     println("  Distance to target: $(distances[i])")
        # end
    end
    
    return x, loss, distances, x_path
end

x_solution, loss, distances, x_path = optimize(x, 1e-2, 5000, solve_objective)

println("\n=== OPTIMAL PATH ANALYSIS ===")
println("Optimal initial conditions: ", x_solution)

## 
mt_optimal = solve(x_solution, dt, 1.0, params, ForwardEuler())
mt_original = solve(x, dt, 1.0, params, ForwardEuler())
@show mt_original[:, end]

final_state = mt_optimal[:, end]

println("Final state reached: ", final_state)
println("Target state: ", target)
println("Final error: ", norm(final_state - target))

# p1 = plot(loss, title="Optimization Loss", xlabel="Iteration", ylabel="Loss", 
#           label="Loss", legend=:topright, lw=2)

time_steps = 0:dt:1.0
p2 = plot(time_steps, mt_optimal[1, :], label="Mx", lw=2, xlabel="Time (s)", ylabel="Magnetization", legend=:topright)
plot!(time_steps, mt_optimal[2, :], label="My", lw=2)
plot!(time_steps, mt_optimal[3, :], label="Mz", lw=2)
hline!([target[1]], ls=:dash, color=1, alpha=0.7, label=nothing)
hline!([target[2]], ls=:dash, color=2, alpha=0.7, label=nothing)
hline!([target[3]], ls=:dash, color=3, alpha=0.7, label=nothing)
title!("Optimal Magnetization Trajectory")

final_plot = plot(p2, layout=(1,1), size=(800, 600))
display(final_plot)