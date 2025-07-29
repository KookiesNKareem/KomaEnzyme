using CUDA, KernelAbstractions, Enzyme, Adapt


const params = (
    γ = 42.58f6 * 2f0π,
    B = 1f-6,
    T₁ = 1.0f0,
    T₂ = 0.5f0,
    M₀ = 1.0f0
)

@inline function cross(a, b)
    return (a[2] * b[3] - a[3] * b[2],
            a[3] * b[1] - a[1] * b[3],
            a[1] * b[2] - a[2] * b[1])
end

# Inputs: (dm, m)
# Output: dm (mutates)
@kernel function bloch!(dm, m)
    # Unpack params
    γ, B, T₁, T₂, M₀ = params
    i = @index(Global)
    mᵢ  = @views(m[:, i])
    dmᵢ = @views(dm[:, i])
    mx, my, mz = mᵢ
    # Calculate right hand side
    Beff = (0.0f0, 0.0f0, B)
    dmᵢ .+= γ .* cross(mᵢ, Beff)
    dmᵢ .-= (mx / T₂, my / T₂, (mz - M₀) / T₁)
end

function bloch_caller(dm, m, backend)
    kernel = bloch!(backend)
    kernel(dm, m, ndrange = size(dm, 2))
    KernelAbstractions.synchronize(backend)
    return
end

backend = CUDABackend()
m  = zeros(3, 1000) |> adapt(backend)
dm = zeros(3, 1000) |> adapt(backend)
m[1, :] .= 1.0f0

bloch_caller(dm, m, backend)

## Enzyme Autodiff

∂dm = ones(3, 1000) |> adapt(backend)
∂m = zeros(3, 1000) |> adapt(backend)

autodiff(Reverse, bloch_caller, Duplicated(dm, ∂dm), Duplicated(m, ∂m), Const(backend))