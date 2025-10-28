#!/usr/bin/env julia
using KomaMRICore, Suppressor
using CairoMakie, LaTeXStrings, KomaMRIPlots
using JLD2, FileIO
using FFTW
import Images
using Statistics: mean

# Rebuild the base sequence (must match optimization setup)
function build_sequence(; FOV=1000e-3, N=60)
    sys = Scanner()
    sys.Smax = 150.0
    sys.Gmax = 100e-3
    seq = PulseDesigner.spiral_base(FOV, N, sys; Nint=1)(0)
    # Reverse gradients for RF excitation
    x, y = 1, 2
    seq.GR[x].A = reverse(seq.GR[x].A)
    seq.GR[y].A = reverse(seq.GR[y].A)
    seq.GR[x].rise, seq.GR[x].fall = seq.GR[x].fall, seq.GR[x].rise
    seq.GR[y].rise, seq.GR[y].fall = seq.GR[y].fall, seq.GR[y].rise
    seq.GR[x].delay = max(dur(seq.GR[x]), dur(seq.GR[y])) - dur(seq.GR[x])
    seq.GR[y].delay = max(dur(seq.GR[x]), dur(seq.GR[y])) - dur(seq.GR[y])
    return seq, sys
end

# Build phantom grid and mask (must match optimization)
function build_grid(; Nspins_x=80, Nspins_y=80, FOV_sim=200e-3)
    xs = range(-FOV_sim/2, FOV_sim/2, Nspins_x)
    ys = range(-FOV_sim/2, FOV_sim/2, Nspins_y)
    x = [x for (x, y) in Iterators.product(xs, ys)][:]
    y = [y for (x, y) in Iterators.product(xs, ys)][:]
    obj = Phantom(; x, y)
    mask = [sqrt(x^2 + y^2) .<= FOV_sim/2.2 for (x, y) in Iterators.product(xs, ys)][:]
    return (; obj, xs, ys, Nspins_x, Nspins_y, mask)
end

# Simulation parameters aligned with optimization
function build_sim_params()
    sim_params = KomaMRICore.default_sim_params()
    sim_params["Δt_rf"] = Inf
    sim_params["Δt"] = Inf
    sim_params["return_type"] = "state"
    sim_params["precision"] = "f64"
    sim_params["Nthreads"] = 32
    sim_params["sim_method"] = KomaMRICore.Bloch()
    return sim_params
end

# Load and process target image matching the subdir name
function load_target(stem::AbstractString, Nspins_x::Int, Nspins_y::Int)
    img_path = joinpath("target_images", string(stem, ".png"))
    img = load(img_path)
    img_bw = reverse(getproperty.(img', :b) .* 1.0, dims=2)
    img_bw .= img_bw ./ maximum(img_bw)
    cx, cy = size(img_bw) .÷ 2 .+ 1
    radius_px = 20
    ft_mask = [(sqrt((i - cx)^2 + (j - cy)^2) <= radius_px) *
               exp(-π * ((i - cx)^2 + (j - cy)^2) / (2 * radius_px^2))
              for i in 1:size(img_bw, 1), j in 1:size(img_bw, 2)]
    img_bw_lowpass = abs.(fftshift(ifft(fftshift(fft(fftshift(img_bw))) .* ft_mask)))
    img_resized = Images.imresize(img_bw_lowpass, (Nspins_x, Nspins_y))
    target_profile = 0.5im .* img_resized[:]
    return target_profile
end

# Simulate with Koma and return achieved complex Mxy vector
function simulate_with_koma(obj, seq, sys, sim_params)
    mag = @suppress simulate(obj, seq, sys; sim_params)
    return mag.xy
end

# Plot and save comparison
function save_validation_plots(outdir::AbstractString, xs, ys, target_profile, achieved_xy, t, x)
    Nspins_x = length(xs)
    Nspins_y = length(ys)
    target2d = reshape(target_profile, Nspins_x, Nspins_y)
    achieved2d = reshape(achieved_xy, Nspins_x, Nspins_y)

    # Target vs Achieved
    fig = Figure(size=(1200, 800))
    ax1 = Axis(fig[1, 1], xlabel=L"$x$ (cm)", ylabel=L"$y$ (cm)", title="Target |Mₓᵧ|")
    hm1 = heatmap!(ax1, xs .* 1e2, ys .* 1e2, abs.(target2d); colormap=:grays)
    Colorbar(fig[1, 2], hm1, ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
    ax2 = Axis(fig[1, 3], xlabel=L"$x$ (cm)", ylabel=L"$y$ (cm)", title="Achieved |Mₓᵧ| (Koma)")
    hm2 = heatmap!(ax2, xs .* 1e2, ys .* 1e2, abs.(achieved2d); colormap=:grays)
    Colorbar(fig[1, 4], hm2, ticks=[0.0, 0.25, 0.5, 1.0])
    colsize!(fig.layout, 1, Auto(1)); colsize!(fig.layout, 2, Fixed(18))
    colsize!(fig.layout, 3, Auto(1)); colsize!(fig.layout, 4, Fixed(18))
    colgap!(fig.layout, 8)
    cl_hi = max(maximum(abs.(target2d)), maximum(abs.(achieved2d)))
    hm1.colorrange[] = (0.0, cl_hi)
    hm2.colorrange[] = (0.0, cl_hi)
    save(joinpath(outdir, "koma_validation.png"), fig, px_per_unit=4)

    # RF profile (real/imag)
    fig_rf = Figure(size=(1000, 350))
    axrf = Axis(fig_rf[1, 1], xlabel="Time (ms)", ylabel="B1 (µT)", title="Optimized RF (Koma sim input)")
    lines!(axrf, t .* 1e3, real(x) .* 1e6, color=:blue, label="Real")
    lines!(axrf, t .* 1e3, imag(x) .* 1e6, color=:red, label="Imag")
    axislegend(axrf, position=:rt)
    save(joinpath(outdir, "koma_rf.png"), fig_rf, px_per_unit=4)
end

# Discover all saved RFs (only in per-target subdirectories under Results)
function find_rf_results(root::AbstractString="Results")
    paths = String[]
    isdir(root) || return paths
    for entry in readdir(root; join=true)
        if isdir(entry)
            rf = joinpath(entry, "RF_2D_image_results.jld2")
            isfile(rf) && push!(paths, rf)
        end
    end
    return sort(paths)
end

results = find_rf_results()
isempty(results) && (@warn "No JLD2 RF results found under Results/"; return)

# Common grid and sim params
grid = build_grid()
sim_params = build_sim_params()

for rfpath in results
    outdir = dirname(rfpath)
    stem = splitext(basename(outdir))[1]  # directory name matches image stem
    @info "Validating" rfpath stem
    # Load RF
    @load rfpath B1_r B1_i
    x = complex.(B1_r, B1_i)
    Nrf = length(x)

    # Build sequence matching optimization and set RF
    seq, sys = build_sequence()
    # Compute RF timing consistent with optimization
    Trf = dur(seq.ADC[1]) - seq.ADC[1].delay
    rf_delay = max(dur(seq.GR[1]), dur(seq.GR[2])) - dur(seq.ADC[1])
    # Assign the optimized RF into the sequence
    seq.RF[1] = RF(x, Trf, 0.0, rf_delay)
    # Ensure ADC matches RF for plotting/sampling consistency
    seq.ADC[1].N = Nrf
    seq.ADC[1].T = Trf
    seq.ADC[1].delay = seq.RF[1].delay

    # Target for this stem
    target_profile = load_target(stem, grid.Nspins_x, grid.Nspins_y)

    # Simulate
    seq_aux = copy(seq); seq_aux.RF[1].A .= x
    achieved_xy = simulate_with_koma(grid.obj, seq_aux, sys, sim_params)
    @info "Achieved magnitude check" stem max_abs = maximum(abs.(achieved_xy))

    # Save plots
    save_validation_plots(outdir, grid.xs, grid.ys, target_profile, achieved_xy, collect(range(0, Trf, Nrf)), x)
end