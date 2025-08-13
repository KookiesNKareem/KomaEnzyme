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

# -----------------------
# AD-safe tiny kernels
# -----------------------

# Pack X = [Mx; My] (Float32) -> ComplexF32 M_xy on device
@kernel function pack_complex!(M::AbstractVector{ComplexF32}, X::AbstractVector{Float32}, N::UInt32)
    i = @index(Global)
    if i <= N
        @inbounds M[i] = ComplexF32(X[i], X[i + Int(N)])
    end
end

# Per-element squared L2 difference on device: out[i] = |A[i]-B[i]|^2
@kernel function l2loss_elem!(out::AbstractVector{Float32},
                              A::AbstractVector{ComplexF32},
                              B::AbstractVector{ComplexF32})
    i = @index(Global)
    if i <= length(A)
        @inbounds d = A[i] - B[i]
        @inbounds out[i] = real(d)*real(d) + imag(d)*imag(d)
    end
end

# -----------------------
# Differentiable GPU reduction wrapper
# -----------------------
gpu_sum(x) = CUDA.sum(x)

function rrule(::typeof(gpu_sum), x::CuArray{T}) where {T}
    y = gpu_sum(x)                 # scalar (computed on device, read on host)
    proj = ProjectTo(x)
    function pullback(ȳ)
        # d/dx sum(x) = 1  =>  ∂L/∂x = ȳ * ones_like(x)
        c = convert(T, ȳ)
        gx = CUDA.fill(c, size(x))
        return (NoTangent(), proj(gx))
    end
    return y, pullback
end

# -----------------------
# Loss function (differentiable)
# -----------------------
# function f(X, mag, obj, seq, target, backend)
#     @assert eltype(X) === Float32
#     @assert X isa CuArray{Float32}

#     N = length(X) ÷ 2
#     @views mag.M_xy .= X[1:N] .+ (1f0im) .* X[N+1:end]

#     # 2) Physics step (in-place)
#     excitation_caller!(mag, obj, seq, backend)

#     # 3) Elementwise loss on device
#     tmp = similar(mag.M_xy, Float32)
#     nthreads2 = cld(length(tmp), GROUP_SIZE) * GROUP_SIZE
#     l2loss_elem!(backend, GROUP_SIZE)(tmp, mag.M_xy, target; ndrange = nthreads2)

#     # 4) Differentiable reduction
#     return gpu_sum(tmp)
# end

function f(X, mag, obj, seq, target, backend)
    @assert eltype(X) === Float32
    @assert X isa CuArray{Float32}

    N = length(X) ÷ 2
    Z = complex.(X[1:N], X[N+1:end])
    mag_new = (; mag..., M_xy=Z)

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