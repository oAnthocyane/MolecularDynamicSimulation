import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from tqdm import tqdm

x, y = 5.0, 20.0
m = 0.5
sigma = 0.2
epsilon = 0.8
N_global = 64
k_B = 1.0
T_target = 1.5
dt = 0.0001
num_steps = 10000
sampling_interval = 100

rcut = 3.0 * sigma
rmin = 1.0 * sigma
shift = 4 * epsilon * ((sigma / rcut) ** 12 - (sigma / rcut) ** 6)

cell_size = rcut
n_cells_x = max(1, int(x / cell_size))
n_cells_y = max(1, int(y / cell_size))

def initialize_positions(N):
    n_side_x = int(np.ceil(np.sqrt(N * x / y)))
    n_side_y = int(np.ceil(np.sqrt(N * y / x)))
    spacing_x = x / n_side_x
    spacing_y = y / n_side_y

    positions = []
    for i in range(n_side_x):
        for j in range(n_side_y):
            if len(positions) < N:
                positions.append([i * spacing_x + spacing_x / 2, j * spacing_y + spacing_y / 2])
    positions = np.array(positions)
    if positions.shape[0] < N:
        raise ValueError(f"Could not initialize {N} particles, only got {positions.shape[0]}")
    return positions

def initialize_velocities(N, T):
    velocities = np.random.normal(0, np.sqrt(k_B * T / m), (N, 2))
    velocities -= np.mean(velocities, axis=0)
    return velocities

def apply_boundary_conditions(positions, velocities, N):
    for i in range(N):
        if positions[i, 0] < 0 or positions[i, 0] > x:
            positions[i, 0] = np.clip(positions[i, 0], 0, x)
            velocities[i, 0] *= -1
        positions[i, 1] %= y

def build_cell_lists(positions, N):
    cell_lists = [[] for _ in range(n_cells_x * n_cells_y)]
    for i in range(N):
        cx = int(np.floor(positions[i, 0] / cell_size)) % n_cells_x
        cy = int(np.floor(positions[i, 1] / cell_size)) % n_cells_y
        cell_lists[cx + cy * n_cells_x].append(i)
    return cell_lists

def get_neighbors(i, positions, cell_lists):
    cx = int(np.floor(positions[i, 0] / cell_size)) % n_cells_x
    cy = int(np.floor(positions[i, 1] / cell_size)) % n_cells_y
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            nx = (cx + dx) % n_cells_x
            ny = (cy + dy) % n_cells_y
            neighbors.extend(cell_lists[nx + ny * n_cells_x])
    return neighbors

def compute_forces(positions, N, use_cell_lists):
    forces = np.zeros((N, 2))
    potential_energy = 0.0
    cell_lists = build_cell_lists(positions, N) if use_cell_lists else None

    for i in range(N):
        neighbors = get_neighbors(i, positions, cell_lists) if use_cell_lists else range(i+1, N)
        for j in neighbors:
            if i >= j:
                continue

            r_vec = positions[j] - positions[i]
            r_vec -= np.round(r_vec / [x, y]) * [x, y]
            r = np.linalg.norm(r_vec)

            if r < rmin:
                f_mag = 24 * epsilon * (2 * (sigma / rmin) ** 12 - (sigma / rmin) ** 6) / rmin
                potential_energy += 4 * epsilon * ((sigma / rmin) ** 12 - (sigma / rmin) ** 6) - shift
            elif r < rcut:
                sr6 = (sigma / r) ** 6
                f_mag = 24 * epsilon * (2 * sr6 ** 2 - sr6) / r
                potential_energy += 4 * epsilon * (sr6 ** 2 - sr6) - shift
            else:
                continue

            f_vec = f_mag * r_vec / r
            forces[i] += f_vec
            forces[j] -= f_vec
    return forces, potential_energy

def simulate(T=None, N_particles=None, visualize=False, use_cell_lists=True):
    T = T if T is not None else T_target
    N_particles = N_particles if N_particles is not None else N_global

    positions = initialize_positions(N_particles)
    velocities = initialize_velocities(N_particles, T)
    forces, potential_energy = compute_forces(positions, N_particles, use_cell_lists)

    pressures = []
    trajectory = [] if visualize else None

    for step in range(num_steps):
        positions += velocities * dt + 0.5 * forces / m * dt ** 2
        apply_boundary_conditions(positions, velocities, N_particles)
        new_forces, potential_energy = compute_forces(positions, N_particles, use_cell_lists)
        velocities += 0.5 * (forces + new_forces) / m * dt
        forces = new_forces

        if (step + 1) % sampling_interval == 0:
            ke = 0.5 * m * np.sum(velocities ** 2)
            temp = ke / (N_particles * k_B)
            pressure = (2 * N_particles * k_B * temp + np.sum(forces * positions)) / (x * y)
            pressures.append(pressure)
            if visualize:
                trajectory.append(positions.copy())

    return np.array(pressures), trajectory

def analyze_sensitivity_and_uncertainty():
    base_pressures, _ = simulate()
    avg_base = np.mean(base_pressures[-int(0.3 * len(base_pressures)):])

    delta_T = 0.1 * T_target
    delta_N = int(0.05 * N_global)

    p_T_plus, _ = simulate(T=T_target + delta_T)
    p_T_minus, _ = simulate(T=T_target - delta_T)
    p_N_plus, _ = simulate(N_particles=N_global + delta_N)
    p_N_minus, _ = simulate(N_particles=N_global - delta_N)

    avg_T_plus = np.mean(p_T_plus[-int(0.3 * len(p_T_plus)):])
    avg_T_minus = np.mean(p_T_minus[-int(0.3 * len(p_T_minus)):])
    avg_N_plus = np.mean(p_N_plus[-int(0.3 * len(p_N_plus)):])
    avg_N_minus = np.mean(p_N_minus[-int(0.3 * len(p_N_minus)):])

    dP_dT = (avg_T_plus - avg_T_minus) / (2 * delta_T)
    dP_dN = (avg_N_plus - avg_N_minus) / (2 * delta_N)

    uncertainty = np.sqrt((dP_dT * delta_T) ** 2 + (dP_dN * delta_N) ** 2)
    percent_uncertainty = 100 * uncertainty / avg_base

    print(f"Base pressure: {avg_base:.4f}")
    print(f"dP/dT ≈ {dP_dT:.4f}")
    print(f"dP/dN ≈ {dP_dN:.4f}")
    print(f"Uncertainty in pressure ≈ ±{uncertainty:.4f} ({percent_uncertainty:.2f}%)")

    plot_pressure_distributions(
        base_pressures, p_T_plus, p_T_minus, p_N_plus, p_N_minus,
        avg_base, dP_dT, dP_dN, uncertainty, percent_uncertainty
    )

def plot_pressure_distributions(p0, p1, p2, p3, p4, avg_base, dPdT, dPdN, unc, unc_pct):
    labels = ["Base", "T +10%", "T -10%", "N +5%", "N -5%"]
    data = [p0, p1, p2, p3, p4]
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    axs[0].boxplot([d[-int(0.3*len(d)):] for d in data], labels=labels)
    axs[0].set_title("Pressure distributions (last 30% steps)")
    axs[0].set_ylabel("Pressure")

    axs[1].bar(["dP/dT", "dP/dN"], [dPdT, dPdN], color=['skyblue', 'salmon'])
    axs[1].set_title(
        f"Sensitivity Analysis\n"
        f"Base Pressure = {avg_base:.2f} ± {unc:.2f} ({unc_pct:.2f}%)\n"
        f"dP/dT = {dPdT:.4f}, dP/dN = {dPdN:.4f}"
    )
    axs[1].set_ylabel("Sensitivity")

    plt.tight_layout()
    plt.savefig("sensitivity_analysis.png")
    plt.close()
    print("Saved: sensitivity_analysis.png")

def main():
    analyze_sensitivity_and_uncertainty()

if __name__ == "__main__":
    main()
