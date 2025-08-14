using CUDA
using KernelAbstractions
using Adapt
using DifferentiationInterface
import Enzyme
import ChainRulesCore: rrule, NoTangent, ProjectTo
using LinearAlgebra: norm

# backend = CPU()
# CUDA.allowscalar(true)

backend = CUDABackend()
CUDA.allowscalar(false)

include("miniKomaCore.jl") 

function excitation_caller!(mag, obj, seq, backend)
    nthreads = cld(length(mag.M_xy), GROUP_SIZE) * GROUP_SIZE
    excitation_kernel!(backend, GROUP_SIZE)(
        mag.M_xy, mag.M_z,
        obj.p_x, obj.p_y, obj.p_z, obj.p_ΔBz, obj.p_T1, obj.p_T2, obj.p_ρ, UInt32(length(obj.p_x)),
        seq.s_Gx, seq.s_Gy, seq.s_Gz, seq.s_Δt, seq.s_f, seq.s_B1, UInt32(length(seq.s_Δt));
        ndrange = nthreads
    )
end

function f(X, mag, obj, seq, target, backend)
    @assert eltype(X) === Float32
    @assert X isa CuArray{Float32}

    N = length(X) ÷ 2
    Z = complex.(X[1:N], X[N+1:end])
    mag_new = (; deepcopy(mag)..., M_xy=Z)
    # mag_new = (M_xy=mag.M_xy, M_z=mag.M_z) |> adapt(backend)
    excitation_caller!(mag_new, obj, seq, backend)
    loss = sum(abs.(mag_new.M_xy .- target).^2)
    return loss
end

# Magnetization
Nspins = 100
mag = (;
    M_xy = zeros(Complex{Float32}, Nspins),
    M_z  = zeros(Float32, Nspins)
)
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
Nt = 100
dt = 1f-5
seq = (;
    s_Gx = zeros(Float32, Nt),
    s_Gy = zeros(Float32, Nt),
    s_Gz = zeros(Float32, Nt),
    s_Δt = fill(dt, Nt),
    s_f  = zeros(Float32, Nt),
    s_B1 = 1f-6 * ones(Complex{Float32}, Nt)
)
# Target
target = rand(Complex{Float32}, Nspins)

# Move everything to GPU backend
mag    = mag    |> adapt(backend)
obj    = obj    |> adapt(backend)
seq    = seq    |> adapt(backend)
target = target |> adapt(backend)

X = cu(zeros(Float32, Nspins * 2))

# Gradient via Enzyme
enzyme_mode = Enzyme.set_runtime_activity(Enzyme.Reverse)
ad_backend  = AutoEnzyme(; mode = enzyme_mode)

loss, ∇X = value_and_gradient(f, ad_backend, X,
                              Constant(mag), Constant(obj), Constant(seq),
                              Constant(target), Constant(backend))

@info "loss" loss
@info "‖grad‖" norm(Array(∇X))