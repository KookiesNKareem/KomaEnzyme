using CUDA
using KernelAbstractions
using Adapt
# using DifferentiationInterface
import Enzyme
import ChainRulesCore: rrule, NoTangent, ProjectTo
using LinearAlgebra: norm
using EnzymeCore
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
    KernelAbstractions.synchronize(backend)
end

# function f(X, mag, obj, seq, target, backend)
#     @assert X isa CuArray{Float32}
#     N = length(X) ÷ 2

#     r = @view X[1:N]
#     i = @view X[N+1:end]

#     im32 = ComplexF32(0, 1)
#     zminus = complex.(r, i) .- target
#     return sum(abs2.(zminus))
# end

@inline _val(x) = x
@inline _val(x::EnzymeCore.Duplicated) = x.val
@inline _val(x::EnzymeCore.Const)      = x.val   # some installs expose Const; yours shows Constant{…}

function f(X, mag, obj, seq, target, backend)
    Xp   = _val(X)
    magp = _val(mag)
    objp = _val(obj)
    seqp = _val(seq)
    tgt  = _val(target)
    be   = _val(backend)   # <- unwrap Constant{CUDABackend} to CUDABackend()

    N = length(Xp) ÷ 2
    r = @view Xp[1:N]
    i = @view Xp[N+1:end]

    @. magp.M_xy = complex(r, i)  # ComplexF32 since r,i are Float32

    excitation_caller!(magp, objp, seqp, be)  # pass the real backend here
    return sum(abs.(magp.M_xy .- tgt).^2)
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

mag_bar = (;
    M_xy = similar(mag.M_xy),
    M_z  = similar(mag.M_z)
)
fill!(mag_bar.M_xy, zero(eltype(mag_bar.M_xy)))
fill!(mag_bar.M_z,  zero(eltype(mag_bar.M_z)))

X = cu(zeros(Float32, Nspins * 2))
δx = cu(zeros(Float32, Nspins * 2))
# Gradient via Enzyme
enzyme_mode = Enzyme.set_runtime_activity(Enzyme.Reverse)
# ad_backend  = AutoEnzyme(; mode = enzyme_mode)

# loss, ∇X = value_and_gradient(f, ad_backend, X,
# Duplicated(mag, mag_bar), Constant(obj), Constant(seq),
#                               Constant(target), Constant(backend))

δX, _ = Enzyme.gradient(
    Enzyme.Reverse,
    f,
    Enzyme.Duplicated(X, δx),
    Enzyme.Duplicated(mag, mag_bar),
    Const(obj), Const(seq), Const(target), Const(backend)
)

@info "‖grad‖" norm(Array(δX))