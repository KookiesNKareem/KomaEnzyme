using CUDA
using KernelAbstractions
using Adapt
using DifferentiationInterface
import Enzyme

backend = CPU() # <-- CPU(), or CUDABackend() for GPU execution

## Kernel definition
include("miniKomaCore.jl")
function excitation_caller!(mag, obj, seq, backend)
    excitation_kernel!(backend, GROUP_SIZE)(
        mag..., 
        obj..., UInt32(length(obj.p_x)), 
        seq..., UInt32(length(seq.s_Δt)), 
        ndrange=cld(length(mag.M_xy), GROUP_SIZE) * GROUP_SIZE
    )
end

## Initializing inputs
# Magnetization
Nspins = 100
mag = (;
    M_xy = zeros(Complex{Float32}, Nspins),
    M_z = zeros(Float32, Nspins)
)
# Phantom
T1=1.0f0
T2=0.5f0
M0=1.0f0
obj = (;
    p_x = zeros(Float32, Nspins),
    p_y = zeros(Float32, Nspins),
    p_z = zeros(Float32, Nspins),
    p_ΔBz = zeros(Float32, Nspins),
    p_T1 = fill(T1, Nspins),
    p_T2 = fill(T2, Nspins),
    p_ρ = fill(M0, Nspins)
)
# Sequence
Nt = 100
dt = 1f-5
seq = (;
    s_Gx = zeros(Float32, Nt),
    s_Gy = zeros(Float32, Nt),
    s_Gz = zeros(Float32, Nt),
    s_Δt = fill(dt, Nt),
    s_f = zeros(Float32, Nt),
    s_B1 = 1f-6 * ones(Complex{Float32}, Nt)
)
# Target magnetization
target = rand(Complex{Float32}, Nspins)

## Optimization-related
# Loss function f
function f(X, mag, obj, seq, target, backend)
    # Unpack X. Note that X is a real vector [Mx; My] to not have problems with complex vectors in Enzyme.
    N = length(X) ÷ 2
    Mx = @view X[1:N]
    My = @view X[N+1:2N]
    # Replaces value in struct mag (initial xy-magnetization = X). For optimizing Bx it should replace seq.s_B1 ...
    mag = (; mag..., M_xy=complex.(Mx, My))
    excitation_caller!(mag, obj, seq, backend) # This modifies mag in-place
    # Calculate the loss
    loss = sum(abs.(mag.M_xy .- target) .^ 2) # sum(abs2, mag.M_xy .- target), doesn't work with Enzyme yet
    return loss
end

## Adapt the structs to the backend
mag = mag |> adapt(backend)
obj = obj |> adapt(backend)
seq = seq |> adapt(backend)
target = target |> adapt(backend)
X = [real.(mag.M_xy); imag.(mag.M_xy)]

# Sanity check, does f work?
f(X, mag, obj, seq, target, backend)

## Gradient calculation, why do I need runtime activity???
# Enzyme
# ad_backend = Enzyme.set_runtime_activity(Enzyme.Reverse)
# ∇X = Enzyme.gradient(ad_backend, f, X, Enzyme.Const(mag), Enzyme.Const(obj), Enzyme.Const(seq), Enzyme.Const(target), Enzyme.Const(backend))

# DifferentiationInterface
ad_backend = AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse)) # Note sure why set_runtime_activity is needed
loss, ∇X = value_and_gradient(f, ad_backend, X, Constant(mag), Constant(obj), Constant(seq), Constant(target), Constant(backend))

## Gradient descent
# lr = 1f-20
# for iter in 1:100
#     loss, ∇X = value_and_gradient(f, AutoEnzyme(; mode=Enzyme.Reverse), X, Constant(mag), Constant(obj), Constant(seq), Constant(target), Constant(backend))
#     X .-= lr .* ∇X
#     println("iter $iter — loss=$(loss)  ∥grad∥=$(norm(∇X))")
# end
## Result
# Mxy_optimal = complex.(X[1:N], X[N+1:2N]) |> adapt(CPU())