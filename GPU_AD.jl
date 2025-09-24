using CUDA
using KernelAbstractions
using Adapt
using DifferentiationInterface
import Enzyme
using LinearAlgebra: norm
import Random: seed!

include("miniKomaCore.jl") 

# println("Starting script...")
# # backend = CPU()
backend = CUDABackend()
CUDA.allowscalar(false)

const GROUP_SIZE = 256
const Nspins = 10               
const Nt     = 100            
const dt     = 1f-5
seed!(42)
N_Spins32 = Int32(length(obj.p_x))
N_Δt32    = Int32(length(seq.s_Δt))

function excitation_caller!(M_xy, M_z,
    p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
    s_Gx, s_Gy, s_Gz, s_Δt, s_f, s_B1,
    N_Spins::Int32, N_Δt::Int32,
    backend)

    k = excitation_kernel!(backend, GROUP_SIZE)
    k(M_xy, M_z,
      p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ, N_Spins,
      s_Gx, s_Gy, s_Gz, s_Δt, s_f, s_B1, N_Δt;
      ndrange=ndrange)

    KernelAbstractions.synchronize(backend)
end


# loss for AD w.r.t. M_xy
function loss_fn(M_xy, M_z, obj, seq, target, backend)
    excitation_caller!(M_xy, M_z, obj.p_x, obj.p_y, obj.p_z, obj.p_ΔBz, obj.p_T1, obj.p_T2, obj.p_ρ, seq.s_Gx, seq.s_Gy, seq.s_Gz, seq.s_Δt, seq.s_f, seq.s_B1, N_Spins32, N_Δt32, backend)
    KernelAbstractions.synchronize(backend)
    return sum(abs.(M_xy .- target).^2)
end

function loss_fn_RF(RF, M_xy, M_z, p_x,p_y,p_z,p_ΔBz,p_T1,p_T2,p_ρ,
    s_Gx,s_Gy,s_Gz,s_Δt,s_f, N_Spins32, N_Δt32, target, backend)
excitation_caller!(RF, M_xy, M_z,
p_x,p_y,p_z,p_ΔBz,p_T1,p_T2,p_ρ,
s_Gx,s_Gy,s_Gz,s_Δt,s_f,
N_Spins32,N_Δt32, backend)
KernelAbstractions.synchronize(backend)
return sum(abs.(M_xy .- target).^2)
end

# Magnetization
M_xy = zeros(ComplexF32, Nspins)
M_z  = zeros(Float32, Nspins)

# Phantom
T1 = 1.0f0; T2 = 0.5f0; M0 = 1.0f0
obj = (;
    p_x  = zeros(Float32, Nspins),
    p_y  = zeros(Float32, Nspins),
    p_z  = zeros(Float32, Nspins),
    p_ΔBz = zeros(Float32, Nspins),
    p_T1 = fill(T1, Nspins),
    p_T2 = fill(T2, Nspins),
    p_ρ  = fill(M0, Nspins)
)

seq = (;
    s_Gx = zeros(Float32, Nt),
    s_Gy = zeros(Float32, Nt),
    s_Gz = zeros(Float32, Nt),
    s_Δt = fill(dt, Nt),
    s_f  = zeros(Float32, Nt),
    s_B1 = 1f-6 .* ones(ComplexF32, Nt)  # real + imag
)

target = rand(Float32, 2Nspins)

# Move to GPU
M_xy  = adapt(backend, M_xy)
M_z   = adapt(backend, M_z)
obj   = adapt(backend, obj)
seq   = adapt(backend, seq)
target = adapt(backend, target)

# Loop bounds


# AD setup
# enzyme_mode = Enzyme.set_runtime_activity(Enzyme.Reverse)
ad_backend  = AutoEnzyme(; mode = Enzyme.Reverse)

# println("Reverse-mode AD w.r.t. RF")
# loss_RF, ∇RF = value_and_gradient(
#     loss_fn_RF, ad_backend,
#     seq.s_B1,                     
#     Constant(M_xy), Constant(M_z),
#     Constant(obj.p_x), Constant(obj.p_y), Constant(obj.p_z),
#     Constant(obj.p_ΔBz), Constant(obj.p_T1), Constant(obj.p_T2), Constant(obj.p_ρ),
#     Constant(seq.s_Gx), Constant(seq.s_Gy), Constant(seq.s_Gz),
#     Constant(seq.s_Δt), Constant(seq.s_f),
#     Constant(N_Spins32), Constant(N_Δt32),
#     Constant(target), Constant(backend)
# )
# @show loss_RF norm(Array(∇RF))

# println("Testing kernel no AD")
# loss1 = loss_fn(seq.s_B1, M_xy, M_z, obj, seq, target, backend)
# @show M_xy
# @show loss1

M_xy = zeros(Float32, 2*Nspins)
M_xy = M_xy |> adapt(backend)

println("Running Enzyme AD w.r.t M_xy")
loss2, ∇X1 = value_and_gradient(loss_fn, ad_backend, M_xy, Constant(M_z), Constant(obj), Constant(seq),
                              Constant(target), Constant(backend))

