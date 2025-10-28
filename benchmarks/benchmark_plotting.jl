# ad_benchmarks.jl
using CairoMakie
using LaTeXStrings
using Makie: LineElement

set_theme!(theme_latexfonts())

# --------------------------
# Data
# --------------------------
spins = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000]

gpu_reverse = [0.0823, 0.1789, 0.2428, 0.8139, 1.7605, 13.5493, 51.2897]
cpu_reverse = [0.2041, 0.9834, 2.0752, 10.8788, 21.6761, 111.5503, 197.1594]

cpu_forward = [28.5666, 135.2504, 267.4058]
gpu_forward = [11.5884, 14.9918, 14.9999, 15.4541, 16.3199, 34.7988, 62.0309]

finite_x_cpu = [100, 500]
finite_y_cpu = [41.4283, 190.1514]

finite_x_gpu = [100, 500]
finite_y_gpu = [50.9882, 115.4154]

# 🎨 Wong colorblind-safe palette
palette   = Makie.wong_colors()
color_fd  = palette[3]  # reddish-orange
color_fwd = palette[2]  # blue
color_rev = palette[1]  # green

# --------------------------
# Figure / Axis
# --------------------------
fig = Figure(resolution = (1000, 800), fontsize = 28, padding = 30)

ax = Axis(fig[1, 1];
    xlabel = L"\text{Number of spins}",
    ylabel = L"\text{Time for 20 iterations (s)}",
    xscale = log10,
    yscale = log10,
    aspect = AxisAspect(1),
)

lw = 4.0  # consistent thicker line width

# Helper: match x to the available y-length using the leading spins values
match_xy(y) = begin
    n = min(length(spins), length(y))
    return spins[1:n], y[1:n]
end

# --------------------------
# Plot series
# Convention: Color = algorithm, Line style = device (GPU solid, CPU dotted)
# --------------------------
# Reverse-mode
lines!(ax, match_xy(gpu_reverse)...; color = color_rev, linestyle = :solid, linewidth = lw, label = "Reverse-mode AD (GPU)")
lines!(ax, match_xy(cpu_reverse)...; color = color_rev, linestyle = :dot,   linewidth = lw, label = "Reverse-mode AD (CPU)")

# Forward-mode
lines!(ax, match_xy(gpu_forward)...; color = color_fwd, linestyle = :solid, linewidth = lw, label = "Forward-mode AD (GPU)")
lines!(ax, match_xy(cpu_forward)...; color = color_fwd, linestyle = :dot,   linewidth = lw, label = "Forward-mode AD (CPU)")

# Finite differences
lines!(ax, finite_x_gpu, finite_y_gpu; color = color_fd, linestyle = :solid, linewidth = lw, label = "Finite Diff (GPU)")
lines!(ax, finite_x_cpu, finite_y_cpu; color = color_fd, linestyle = :dot,   linewidth = lw, label = "Finite Diff (CPU)")

# --------------------------
# Reference line & label
# --------------------------
hlines!(ax, 5.0; color = (:black, 0.6), linestyle = :dot, linewidth = 2)
text!(ax, "5 seconds"; position = (spins[1] * 1.2, 9.0), align = (:left, :top), color = (:black, 0.7), fontsize = 24)

# --------------------------
# Legends
# --------------------------
# Main legend for the plotted series (shows color + label text)
main_legend = Legend(fig, ax;
    nbanks = 1,
    framevisible = true,
    bgcolor = (:white, 0.8),
    padding = (10, 10, 10, 10),
)

# Place both legends in a right-side column (aux legend under the main one)
right = GridLayout()
right[1, 1] = main_legend
fig[1, 2] = right
colsize!(fig.layout, 2, Auto())

# --------------------------
# Ticks & Limits
# --------------------------
ax.xticks = ([1e2, 1e3, 1e4, 1e5],
             [L"10^2", L"10^3", L"10^4", L"10^5"])
ax.yticks = ([1e-2, 1e-1, 1, 10, 100, 1000],
             [L"10^{-2}", L"10^{-1}", L"10^{0}", L"10^{1}", L"10^{2}", L"10^{3}"])

# Tight Y limits
all_y = vcat(gpu_reverse, cpu_reverse, cpu_forward, gpu_forward, finite_y_cpu, finite_y_gpu)
ylims!(ax, minimum(all_y) * 0.7, maximum(all_y))

# --------------------------
# Render & Save
# --------------------------
fig
save("ad_benchmarks.png", fig; px_per_unit = 2)
save("ad_benchmarks.pdf", fig)
println("Saved plots: ad_benchmarks.png, ad_benchmarks.pdf")