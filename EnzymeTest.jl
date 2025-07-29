# using Test
# using Enzyme
# using KernelAbstractions

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


# @kernel function mul!(A, B)
#     I = @index(Global, Linear)
#     @inbounds A[I] *= B
# end

# function mul_caller(A, B, backend)
#     kernel = mul!(backend)
#     kernel(A, B, ndrange = size(A))
#     KernelAbstractions.synchronize(backend)
#     return
# end

# A = cu(Array{Float64}(undef, 64))
# dA = cu(Array{Float64}(undef, 64))
# supports_reverse = true
# if supports_reverse

#     A .= (1:1:64)
#     dA .= 1

#     Enzyme.autodiff(Reverse, square_caller, Duplicated(A, dA), Const(backend))
#     @test all(dA .≈ (2:2:128))

#     # active arguments not support for GPU kernels
#     if backend == CPU
#         A .= (1:1:64)
#         dA .= 1

#         _, dB, _ = Enzyme.autodiff(Reverse, mul_caller, Duplicated(A, dA), Active(1.2), Const(backend))[1]

#         @test all(dA .≈ 1.2)
#         @test dB ≈ sum(1:1:64)
#     end
# end

# A .= (1:1:64)
# dA .= 1

# Enzyme.autodiff(Forward, square_caller, Duplicated(A, dA), Const(backend))
# KernelAbstractions.synchronize(backend)
# @test all(dA .≈ 2:2:128)


# using CUDA, Enzyme, Test

# function mul_kernel(A, shared)
#     i = threadIdx().x
#     if i <= length(A)
#         shared[i] = A[i] * A[i]
#         A[i] = shared[i]
#     end
#     return nothing
# end

# function grad_mul_kernel(A, dA,sh,d_sh)
#     Enzyme.autodiff_deferred(Enzyme.Const(mul_kernel), Const, Duplicated(A, dA), Duplicated(sh, d_sh))
#     return nothing
# end

# A = CUDA.ones(64,)
# @cuda threads=length(A) shmem=64*4 mul_kernel(A,CuDynamicSharedArray(Float32, 64) )
# A = CUDA.ones(64,)
# dA = similar(A)
# dA .= 1
# @cuda threads=length(A) shmem=64*4 grad_mul_kernel(A, dA,CuDynamicSharedArray(Float32, 64),CuDynamicSharedArray(Float32, 64))
# @test all(dA .== 2)

using CUDA
using Enzyme
using Test

function mul_kernel(A)
    i = threadIdx().x
    if i <= length(A)
        A[i] *= A[i]
    end
    return nothing
end

function grad_mul_kernel(A, dA)
    autodiff_deferred(Reverse, Const(mul_kernel), Const, Duplicated(A, dA))
    return nothing
end

@testset "mul_kernel" begin
    A = CUDA.ones(64,)
    @cuda threads=length(A) mul_kernel(A)
    A = CUDA.ones(64,)
    dA = similar(A)
    dA .= 1
    @cuda threads=length(A) grad_mul_kernel(A, dA)
    @test all(dA .== 2)
end

function exp_kernel(A)
    i = threadIdx().x
    if i <= length(A)
        A[i] = exp(A[i])
    end
    return nothing
end

function grad_exp_kernel(A, dA)
    autodiff_deferred(Reverse, Const(exp_kernel), Const, Duplicated(A, dA))
    return nothing
end

@testset "exp_kernel" begin
    A = CUDA.ones(64,)
    @cuda threads=length(A) exp_kernel(A)
    A = CUDA.ones(64,)
    dA = similar(A)
    dA .= 1
    @cuda threads=length(A) grad_exp_kernel(A, dA)
    @test all(dA .== exp(1.f0))
end

function cos_kernel(A)
    i = threadIdx().x
    if i <= length(A)
        A[i] = cos(A[i])
    end
    return nothing
end

function grad_cos_kernel(A, dA)
    autodiff_deferred(Reverse, Const(cos_kernel), Const, Duplicated(A, dA))
    return nothing
end

@testset "cos_kernel" begin
    A = CUDA.ones(64,)
    @cuda threads=length(A) cos_kernel(A)
    A = CUDA.ones(64,)
    dA = similar(A)
    dA .= 1
    @cuda threads=length(A) grad_cos_kernel(A, dA)
    @test all(dA .≈ -sin(1.f0))
end

function val_kernel!(_, ::Val{N}) where N
    return nothing
end

function dval_kernel!(du, ::Val{N}) where {N}
    autodiff_deferred(Reverse, Const(val_kernel!), Const, du, Const(Val(N)))
    return nothing
end

# Test for https://github.com/EnzymeAD/Enzyme.jl/issues/358
@testset "Test val kernel" begin
    n = 10
    u = CUDA.rand(n)
    dzdu = CUDA.rand(n)
    @cuda threads=4 dval_kernel!(Duplicated(u, dzdu), Val(n))
end

# https://github.com/EnzymeAD/Enzyme.jl/issues/367

relu(x) = ifelse(x<0, zero(x), x)
## Define CUDA kernel for dense layer
function dense!(
  feats_out, feats_in, W, b,
  ::Val{nfeat_out}, ::Val{nfeat_in}, ::Val{ndof}
) where {nfeat_out, nfeat_in, ndof}

  ## Each thread will update features for a single sample
  idof = (blockIdx().x - 1) * blockDim().x + threadIdx().x

  ## Prevent out-of-bounds array access
  (idof > ndof) && return nothing

  ## Compute `feats_out`
  for i = 1:nfeat_out
    for k = 1:nfeat_in
      feats_out[i, idof] += W[i, k] * feats_in[k, idof]
    end

    feats_out[i, idof] = relu(feats_out[i, idof] + b[i])
  end

  return nothing
end # dense!

## Wrapper for Enzyme to differentiate `dense!`
function ddense!(
  dfeats_out, dfeats_in, dW, db,
  ::Val{nfeat_out}, ::Val{nfeat_in}, ::Val{ndof}
) where {nfeat_out, nfeat_in, ndof}

  autodiff_deferred(
    Reverse,
    Const(dense!),
    Const,
    dfeats_out, dfeats_in, dW, db,
    Const(Val(nfeat_out)), Const(Val(nfeat_in)), Const(Val(ndof))
  )
  return nothing

end # ddense!

function call_ddense()
  nthread      = 32
  ndof         = 32
  nblock       = ceil(Int, ndof / nthread)
  nfeat_out    = 32
  nfeat_in     = 16
  feats_out    = zeros(nfeat_out, ndof)    |> cu
  feats_in     = rand(nfeat_in, ndof)      |> cu
  W            = rand(nfeat_out, nfeat_in) |> cu
  b            = rand(nfeat_out)           |> cu
  dzdfeats_out = rand(nfeat_out, ndof)     |> cu
  dfeats_out   = Duplicated(feats_out, dzdfeats_out)
  dzdfeats_in  = zero(feats_in)
  dfeats_in    = Duplicated(feats_in, dzdfeats_in)
  dzdW         = zero(W)
  dW           = Duplicated(W, dzdW)
  dzdb         = zero(b)
  db           = Duplicated(b, dzdb)

  @cuda threads=nthread blocks=nblock ddense!(
    dfeats_out, dfeats_in, dW, db,
    Val(nfeat_out), Val(nfeat_in), Val(ndof)
  )

end # call_ddense

@testset "DDense" begin
    call_ddense()
end

function square_kernel!(x)
    i = threadIdx().x
    x[i] *= x[i]
    sync_threads()
    return nothing
end

# basic squaring on GPU
function square!(x)
    @cuda blocks = 1 threads = length(x) square_kernel!(x)
    return nothing
end

@testset "Reverse Kernel" begin
    A = CUDA.rand(64)
    dA = CUDA.ones(64)
    A .= (1:1:64)
    dA .= 1
    Enzyme.autodiff(Reverse, square!, Duplicated(A, dA))
    @test all(dA .≈ (2:2:128))

    A = CUDA.rand(32)
    dA = CUDA.ones(32)
    dA2 = CUDA.ones(32)
    A .= (1:1:32)
    dA .= 1
    dA2 .= 3
    Enzyme.autodiff(Reverse, square!, BatchDuplicated(A, (dA, dA2)))
    @test all(dA .≈ (2:2:64))
    @test all(dA2 .≈ 3*(2:2:64))
end