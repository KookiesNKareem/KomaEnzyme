using Reactant, Enzyme, Test

Reactant.set_default_backend("gpu")

function cube(x)
    return x * 3
end

julia_data = ones(10)
reactant_data = Reactant.ConcreteRArray(julia_data)

fwd(Mode, RT, x, y) = Enzyme.autodiff(Mode, cube, RT, Duplicated(x, y))

@testset "Forward" begin
    func = fwd(Forward, Duplicated, ones(3, 2), 3.1 * ones(3, 2))
    @test func[1] ≈ 9.3 * ones(3, 2)
end

# function run_spin_excitation!(
#     p::Phantom{T},
#     seq::DiscreteSequence{T},
#     M::Mag{T}) where {T<:Real}
#     for s in seq
#         x, y, z = get_spin_coords(p.motion, p.x, p.y, p.z, s.t)
#         ΔBz = p.Δw ./ T(2π .* γ) .- s.Δf ./ T(γ)
#         Bz = (s.Gx .* x .+ s.Gy .* y .+ s.Gz .* z) .+ ΔBz
#         B = sqrt.(abs.(s.B1) .^ 2 .+ abs.(Bz) .^ 2)
#         B .+= (B .== 0) .* eps(T)
#         φ = T(-2π .* γ) .* (B .* s.Δt)
#         mul!(Q(φ, s.B1 ./ B, Bz ./ B), M)
#         M.xy .= M.xy .* exp.(-s.Δt ./ p.T2)
#         M.z .= M.z .* exp.(-s.Δt ./ p.T1) .+ p.ρ .* (1 .- exp.(-s.Δt ./ p.T1))
#         outflow_spin_reset!(M, s.t, p.motion; replace_by=p.ρ)
#     end
#     return nothing
# end

# fwd(mode, RT, p, seq, M) =
#     Enzyme.autodiff(
#         mode,
#         run_spin_excitation!,
#         RT,
#         Duplicated(p, seq, M))