using CUDA
using KernelAbstractions
using KernelAbstractions.Extras: @unroll
using Adapt
using Enzyme

# @inline function get_spin_coordinates(x::AbstractVector{T}, y::AbstractVector{T},
#                                       z::AbstractVector{T}, i::Integer, t::Integer) where {T}
#   @inbounds (x[i], y[i], z[i])
# end

# @inline function get_spin_coordinates(x::AbstractMatrix{T}, y::AbstractMatrix{T},
#                                       z::AbstractMatrix{T}, i::Integer, t::Integer) where {T}
#   @inbounds (x[i, t], y[i, t], z[i, t])
# end
struct Literal{T} end
Literal(T) = Literal{T}()
Base.:*(x::Number, ::Literal{T}) where T = T(x)
const u32 = Literal(UInt32)

@kernel unsafe_indices=true inbounds=true function excitation!(
    M_xy::AbstractVector{Complex{T}}, M_z::AbstractVector{T},
    @Const(p_x),  @Const(p_y),  @Const(p_z),
    @Const(p_ΔBz),@Const(p_T1), @Const(p_T2), @Const(p_ρ),
    N_Spins::UInt32,
    @Const(s_Gx), @Const(s_Gy), @Const(s_Gz),
    @Const(s_Δt), @Const(s_Δf), @Const(s_B1),
    N_Δt::UInt32
) where {T}
    γ = 42.58f6 * 2π

    # thread/block indices
    @uniform N = @groupsize()[1]
    i_l = @index(Local,  Linear)
    i_g = @index(Group,  Linear)
    i   = (i_g - 1u32)*UInt32(N) + i_l

    if i <= N_Spins
      # load per‐spin state
    #   x, y, z      = get_spin_coordinates(p_x, p_y, p_z, i, 1)
        x = p_x[i]
        y = p_y[i]
        z= p_z[i]
        ΔBz          = p_ΔBz[i]
        Mxy_r, Mxy_i = reim(M_xy[i])
        Mz           = M_z[i]
        ρ            = p_ρ[i]
        T1           = p_T1[i]
        T2           = p_T2[i]

      for s_idx in UInt32(1):N_Δt
        # x, y, z = get_spin_coordinates(p_x, p_y, p_z, i, s_idx)

        # effective field
        Bz = (x * s_Gx[s_idx] + y * s_Gy[s_idx] + z * s_Gz[s_idx]) + ΔBz - s_Δf[s_idx] / γ
        B1_r, B1_i = reim(s_B1[s_idx])
        B = sqrt(B1_r^2 + B1_i^2 + Bz^2)

        Δt = s_Δt[s_idx]
        φ  = -π * γ * B * Δt
        sin_φ, cos_φ = sincos(φ)

        # rotation coefficients
        α_r =  cos_φ
        α_i = -(iszero(B) ? Bz/(B + eps(T)) : Bz/B) * sin_φ
        β_r =  (iszero(B) ?  B1_i/(B + eps(T)) : B1_i/B) * sin_φ
        β_i = -(iszero(B) ?  B1_r/(B + eps(T)) : B1_r/B) * sin_φ

        # apply rotation + relaxation
        Mxy_new_r = 2*(Mxy_i*(α_r*α_i - β_r*β_i) + Mz*(α_i*β_i + α_r*β_r)) +
                    Mxy_r*(α_r^2 - α_i^2 - β_r^2 + β_i^2)
        Mxy_new_i = -2*(Mxy_r*(α_r*α_i + β_r*β_i) - Mz*(α_r*β_i - α_i*β_r)) +
                     Mxy_i*(α_r^2 - α_i^2 + β_r^2 - β_i^2)
        Mz_new     =  Mz*(α_r^2 + α_i^2 - β_r^2 - β_i^2) -
                     2*(Mxy_r*(α_r*β_r - α_i*β_i) + Mxy_i*(α_r*β_i + α_i*β_r))

        ΔT1 = exp(-Δt/T1)
        ΔT2 = exp(-Δt/T2)

        Mxy_r = Mxy_new_r * ΔT2
        Mxy_i = Mxy_new_i * ΔT2
        Mz    = Mz_new    * ΔT1 + ρ*(1 - ΔT1)
      end

      M_xy[i] = Complex(Mxy_r, Mxy_i)
      M_z[i]  = Mz
    end
end

const backend = CUDABackend() 
const groupsize = 256
# 4) Thin wrapper to move data → GPU, launch, sync:
function excite!(M_xy, M_z,
                 p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
                 s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1, NΔt)

  kern = excitation!(backend, groupsize)
  Nsp = UInt32(length(M_xy))

  kern(M_xy, M_z,
       p_x, p_y, p_z,
       p_ΔBz, p_T1, p_T2, p_ρ,
       Nsp,
       s_Gx, s_Gy, s_Gz,
       s_Δt, s_Δf, s_B1,
       NΔt;
       ndrange=(cld(length(M_xy), groupsize) * groupsize))

  KernelAbstractions.synchronize(backend)
  return
end

Nsp = 1000
Nt  = 128

# initial magnetizations (Float64)
M_xy = fill(Complex(0.0f0, 0.0f0), Nsp) |> adapt(backend)
M_z  = ones(Float32, Nsp) |> adapt(backend)

# dummy spin coords & parameters
p_x   = rand(Float32, Nsp)|> adapt(backend)
p_y   = rand(Float32, Nsp) |> adapt(backend)
p_z   = rand(Float32, Nsp) |> adapt(backend)
p_ΔBz = zeros(Float32, Nsp) |> adapt(backend)
p_T1  = fill(1.0f0, Nsp) |> adapt(backend)
p_T2  = fill(0.5f0, Nsp) |> adapt(backend)
p_ρ   = fill(1.0f0, Nsp) |> adapt(backend)

s_Gx  = rand(Float32, Nt) |> adapt(backend)
s_Gy  = rand(Float32, Nt) |> adapt(backend)
s_Gz  = rand(Float32, Nt) |> adapt(backend)
s_Δt  = fill(1f-3, Nt) |> adapt(backend)
s_Δf  = zeros(Float32, Nt) |> adapt(backend)
s_B1  = fill(Complex(0.1f0,0.0f0), Nt) |> adapt(backend)

Nsp = UInt32(length(M_xy))
NΔt = UInt32(length(s_Δt))    
# forward run
excite!(M_xy, M_z, p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1, NΔt)

# prepare adjoints
# ∂Mxy = ones(eltype(M_xy), size(M_xy)...) |> adapt(CUDABackend())
# ∂Mz  = zeros(eltype(M_z),  size(M_z) )  |> adapt(CUDABackend())

# # reverse-mode AD
# autodiff(Reverse,
#         excite!,
#         Duplicated(M_xy, ∂Mxy),
#         Duplicated(M_z,  ∂Mz),
#         Const(CUDABackend()))

# println("Reverse-mode AD complete.")
