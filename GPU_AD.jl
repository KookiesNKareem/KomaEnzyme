using CUDA
using KernelAbstractions
using Adapt
using DifferentiationInterface
import Enzyme
import ChainRulesCore: rrule, NoTangent, ProjectTo
using LinearAlgebra: norm
import Random: seed!

include("miniKomaCore.jl") 


println("Starting script...")
# backend = CPU()
backend = CUDABackend()
CUDA.allowscalar(false)

const GROUP_SIZE = 256
seed!(42)


function excitation_caller!(M_xy, M_z, obj, seq, backend)
    nthreads = cld(length(M_xy), GROUP_SIZE) * GROUP_SIZE
    excitation_kernel!(backend, GROUP_SIZE)(M_xy, M_z, obj.p_x, obj.p_y, obj.p_z, obj.p_ΔBz, obj.p_T1, obj.p_T2, obj.p_ρ, length(obj.p_x),
        seq.s_Gx, seq.s_Gy, seq.s_Gz, seq.s_Δt, seq.s_f, seq.s_B1,length(seq.s_Δt);
        ndrange = nthreads)
end

function loss_fn(M_xy, M_z, obj, seq, target, backend)
    println("Calling excitation_caller!")
    excitation_caller!(M_xy, M_z, obj, seq, backend)
    println("Finished excitation_caller!")
    loss = sum(abs.(M_xy .- target).^2)
    return loss
end

# Magnetization
Nspins = 10 # i used a smaller number to be able to read the output better
M_xy = zeros(Float32, 2*Nspins) # real + imag
M_z  = zeros(Float32, Nspins)

# Phantom
T1 = 1.0f0; T2 = 0.5f0; M0 = 1.0f0
obj = (;
    p_x = zeros(Float32, Nspins),
    p_y = zeros(Float32, Nspins),
    p_z = zeros(Float32, Nspins),
    p_ΔBz = zeros(Float32, Nspins),
    p_T1 = fill(T1, Nspins),
    p_T2 = fill(T2, Nspins),
    p_ρ  = fill(M0, Nspins)
)

# Sequence
Nt = 100; dt = 1f-5;
seq = (;
    s_Gx = zeros(Float32, Nt),
    s_Gy = zeros(Float32, Nt),
    s_Gz = zeros(Float32, Nt),
    s_Δt = fill(dt, Nt),
    s_f  = zeros(Float32, Nt),
    s_B1 = 1f-6 * ones(Float32, 2 * Nt)
)

# Target
target = rand(Float32, Nspins * 2)

# Move to GPU 
M_xy = M_xy |> adapt(backend)
M_z  = M_z  |> adapt(backend)
obj    = obj    |> adapt(backend)
seq    = seq    |> adapt(backend)
target = target |> adapt(backend)

# Gradient via Enzyme
enzyme_mode = Enzyme.set_runtime_activity(Enzyme.Reverse)
ad_backend  = AutoEnzyme(; mode = enzyme_mode)

println("Testing kernel no AD")
loss_fn(M_xy, M_z, obj, seq, target, backend)
@info M_xy

# println("Running Enzyme AD")
# loss, ∇X = value_and_gradient(loss_fn, ad_backend, M_xy, Constant(M_z), Constant(obj), Constant(seq),
#                               Constant(target), Constant(backend))

# @info "loss" loss
# @info "‖grad‖" norm(Array(∇X))