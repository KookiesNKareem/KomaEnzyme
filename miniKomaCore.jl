using CUDA
using KernelAbstractions

@inline function get_spin_coordinates(x::AbstractVector{T}, y::AbstractVector{T}, z::AbstractVector{T}, i::Integer, t::Integer) where {T<:Real}
    @inbounds (x[i], y[i], z[i])
end

@inline function get_spin_coordinates(x::AbstractMatrix{T}, y::AbstractMatrix{T}, z::AbstractMatrix{T}, i::Integer, t::Integer) where {T<:Real}
    @inbounds (x[i, t], y[i, t], z[i, t])
end

@kernel unsafe_indices=true inbounds=true function excitation_simple!(
    M_xy::AbstractVector{Complex{T}},
    @Const(p_x), @Const(p_y), @Const(p_z),
    N_spins::UInt32,
) where {T}
    gs = @groupsize()[1]
    li = @index(Local, Linear)
    gi = @index(Group, Linear)
    idx = (gi - UInt32(1)) * UInt32(gs) + li
    if idx <= N_spins
        x, y, _ = get_spin_coordinates(p_x, p_y, p_z, idx, 1)
        r, i = reim(M_xy[idx])
        M_xy[idx] = Complex(r + x, i + y)
    end
end

@kernel unsafe_indices=true inbounds=true function excitation!(
    M_xy::AbstractVector{Complex{T}},
    M_z::AbstractVector{T},
    @Const(p_x), @Const(p_y), @Const(p_z),
    @Const(p_ΔBz), @Const(p_T1), @Const(p_T2), @Const(p_ρ),
    N_spins::UInt32,
    @Const(s_Gx), @Const(s_Gy), @Const(s_Gz),
    @Const(s_Δt), @Const(s_Δf), @Const(s_B1),
    N_steps::UInt32,
) where {T}
    gs = @groupsize()[1]
    li = @index(Local, Linear)
    gi = @index(Group, Linear)
    idx = (gi - UInt32(1)) * UInt32(gs) + li
    if idx <= N_spins
        # load initial state
        r, i = reim(M_xy[idx]); mz = M_z[idx]
        ΔBz = p_ΔBz[idx]; ρ = p_ρ[idx]
        T1 = p_T1[idx];    T2 = p_T2[idx]

        # loop over time steps
        for step in UInt32(1):N_steps
            x, y, z = get_spin_coordinates(p_x, p_y, p_z, idx, step)
            Δt = s_Δt[step]
            # effective field
            Bz = x*s_Gx[step] + y*s_Gy[step] + z*s_Gz[step] + ΔBz - s_Δf[step]/γ
            B1r, B1i = reim(s_B1[step])
            B  = sqrt(B1r^2 + B1i^2 + Bz^2)
            φ  = -π * γ * B * Δt
            sinφ, cosφ = sincos(φ)
            # rotation coefficients
            αr =  cosφ
            αi = -(Bz/(B + eps(T))) * sinφ
            βr =  (B1i/(B + eps(T))) * sinφ
            βi = -(B1r/(B + eps(T))) * sinφ
            # apply rotation + relaxation
            new_r = 2*(i*(αr*αi - βr*βi) + mz*(αi*βi + αr*βr)) + r*(αr^2 - αi^2 - βr^2 + βi^2)
            new_i = -2*(r*(αr*αi + βr*βi) - mz*(αr*βi - αi*βr)) + i*(αr^2 - αi^2 + βr^2 - βi^2)
            new_mz = mz*(αr^2 + αi^2 - βr^2 - βi^2) - 2*(r*(αr*βr - αi*βi) + i*(αr*βi + αi*βr))
            expT1 = exp(-Δt/T1); expT2 = exp(-Δt/T2)
            r = new_r * expT2; i = new_i * expT2
            mz = new_mz*expT1 + ρ*(1 - expT1)
        end
        M_xy[idx] = Complex(r, i)
        M_z[idx]  = mz
    end
end