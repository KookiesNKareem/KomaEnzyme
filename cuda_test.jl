# using Enzyme, KernelAbstractions, DifferentiationInterface, CUDA

# @kernel function square!(A)
#     I = @index(Global, Linear)
#     @inbounds A[I] *= A[I]
# end

# function square_caller(A, backend)
#     kernel = square!(backend)
#     kernel(A, ndrange = size(A))
#     KernelAbstractions.synchronize(backend)
#     return
# end

# a = cu(Array{Float64}(undef, 64))
# a .= (1:1:64)
# ∂a = cu(Array{Float64}(undef, 64))
# backend = get_backend(a)
# Enzyme.Compiler.VERBOSE_ERRORS[] = true
# autodiff(Reverse, square_caller, Duplicated(a, ∂a), Const(backend))

using CUDA
using Enzyme           # Enzyme.jl
using DifferentiationInterface  # DI.jl

# -- CUDA kernel definition --
function square_kernel!(A)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(A)
        @inbounds A[i] = A[i] * A[i]
    end
    return
end

# -- Host‐side caller that launches the kernel --
function square_caller(A, ∂A, backend)
    # compute launch configuration
    N = length(A)
    threads = 256
    blocks = cld(N, threads)
    # launch
    @cuda threads=threads blocks=blocks square_kernel!(A)
    # ensure completion
    CUDA.synchronize()
    return
end

# -- Prepare data on GPU --
N = 64
a  = CUDA.CuArray{Float64}(undef, N)
a .= 1.0:N
∂a = CUDA.CuArray{Float64}(undef, N)  # to hold gradients

# Verbose Enzyme errors if you need them
Enzyme.Compiler.VERBOSE_ERRORS[] = true

# -- Run reverse‐mode AD --
# We wrap the caller so that Enzyme sees the two arrays: input/output and its seed.
autodiff(Reverse, square_caller, Duplicated(a, ∂a), Const(nothing))