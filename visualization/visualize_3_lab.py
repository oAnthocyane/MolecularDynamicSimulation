import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def visualize_sensitivity_analysis():
    # Path to the results folder
    results_dir = "../result_simulation"
    
    # Load summary data
    summary_file = os.path.join(results_dir, "sensitivity_summary.csv")
    summary_data = {}
    
    with open(summary_file, 'r') as f:
        for line in f:
            if line.strip() and ',' in line and not line.startswith('parameter'):
                param, value = line.strip().split(',')
                summary_data[param] = float(value)
    
    # Load pressure data
    pressures_file = os.path.join(results_dir, "sensitivity_pressures.csv")
    pressures_df = pd.read_csv(pressures_file)
    
    # Calculate the index for the last 30% of data
    start_idx = int(len(pressures_df) * 0.7)
    last_30_percent = pressures_df.iloc[start_idx:]
    
    # Extract pressure data for box plots
    pressure_data = [
        last_30_percent['base'],
        last_30_percent['t_plus'],
        last_30_percent['t_minus'],
        last_30_percent['n_plus'],
        last_30_percent['n_minus']
    ]
    
    # Create figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    
    # Box plot of pressure distributions
    axs[0].boxplot(pressure_data, 
                   labels=["Base", "T +10%", "T -10%", "N +5%", "N -5%"])
    axs[0].set_title("Pressure distributions (last 30% steps)")
    axs[0].set_ylabel("Pressure")
    
    # Bar chart of sensitivities
    sensitivities = [summary_data['dp_dt'], summary_data['dp_dn']]
    axs[1].bar(["dP/dT", "dP/dN"], sensitivities, color=['skyblue', 'salmon'])
    axs[1].set_title(
        f"Sensitivity Analysis\n"
        f"Base Pressure = {summary_data['base_pressure']:.2f} ± {summary_data['uncertainty']:.2f} "
        f"({summary_data['percent_uncertainty']:.2f}%)\n"
        f"dP/dT = {summary_data['dp_dt']:.4f}, dP/dN = {summary_data['dp_dn']:.4f}"
    )
    axs[1].set_ylabel("Sensitivity")
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "sensitivity_analysis.png"))
    print(f"Saved: {os.path.join(results_dir, 'sensitivity_analysis.png')}")

def plot_simulation_results():
    # Path to the results folder
    results_dir = "../result_simulation"
    
    # Load and plot results for each boundary condition
    boundary_types = ["periodic", "reflective", "mixed"]
    
    # Create a figure with subplots for energy, pressure, and temperature
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    for boundary in boundary_types:
        # Load results
        results_file = os.path.join(results_dir, f"results_{boundary}.csv")
        data = pd.read_csv(results_file)
        
        # Plot energy
        axs[0].plot(data['time'], data['energy'], label=boundary)
        
        # Plot pressure
        axs[1].plot(data['time'], data['pressure'], label=boundary)
        
        # Plot temperature
        axs[2].plot(data['time'], data['temperature'], label=boundary)
    
    # Set labels and titles
    axs[0].set_title("Energy vs Time")
    axs[0].set_ylabel("Energy")
    axs[0].legend()
    
    axs[1].set_title("Pressure vs Time")
    axs[1].set_ylabel("Pressure")
    axs[1].legend()
    
    axs[2].set_title("Temperature vs Time")
    axs[2].set_xlabel("Time")
    axs[2].set_ylabel("Temperature")
    axs[2].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "simulation_comparison.png"))
    print(f"Saved: {os.path.join(results_dir, 'simulation_comparison.png')}")

def main():
    print("Visualizing sensitivity analysis...")
    visualize_sensitivity_analysis()
    
    print("Plotting simulation results...")
    plot_simulation_results()

if __name__ == "__main__":
    main()