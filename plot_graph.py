import numpy as np
import matplotlib.pyplot as plt
import os

def plot_run(filename):
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return

    print(f"Plotting {filename}...")
    z = np.load(filename)
    
    # Extract training metrics
    valid_err = z["valid_error"]
    rho = z["rho_Whh"]
    
    # Extract diagnostics (NaN padded)
    grad_time = z["grad_time"]
    sat_time = z["sat_time"]
    
    # Check for GRU gate diagnostics
    has_gates = "gate_z_sat_time" in z and "gate_r_sat_time" in z
    
    # Choose the last valid checkpoint (filter out completely NaN rows)
    # We find the last index where grad_time is not entirely NaNs
    valid_indices = np.where(~np.isnan(grad_time).all(axis=1))[0]
    if len(valid_indices) == 0:
        print(f"No valid diagnostic data found in {filename}.")
        return
    last_idx = valid_indices[-1]

    # Extract the actual data at the final checkpoint
    g = grad_time[last_idx]
    s = sat_time[last_idx]
    g = g[np.isfinite(g)]
    s = s[np.isfinite(s)]
    
    # Determine grid size based on whether we have GRU gates
    num_plots = 6 if has_gates else 4
    cols = 2
    rows = (num_plots + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    fig.suptitle(f"Diagnostics for {filename}", fontsize=14, fontweight='bold')
    axes = axes.flatten()

    # 1. Validation Error
    axes[0].plot(valid_err, color='blue', linewidth=2)
    axes[0].set_title("Validation Error (%)")
    axes[0].set_xlabel("Checkpoint")
    axes[0].set_ylabel("Error")
    axes[0].grid(True, linestyle='--', alpha=0.7)

    # 2. Spectral Radius (rho)
    axes[1].plot(rho, color='purple', linewidth=2)
    axes[1].set_title("Spectral Radius $\\rho(W_{hh})$")
    axes[1].set_xlabel("Checkpoint")
    axes[1].set_ylabel("$\\rho$")
    axes[1].grid(True, linestyle='--', alpha=0.7)

    # 3. Gradient-through-time Histogram
    axes[2].hist(np.log10(g + 1e-12), bins=60, color='orange', edgecolor='black', alpha=0.7)
    axes[2].set_title("Gradient Signal: $\\log_{10}||\\partial\\mathcal{L}/\\partial h_t||_2$")
    axes[2].set_xlabel("$\\log_{10}(g_t)$")
    axes[2].set_ylabel("Frequency")

    # 4. Hidden Saturation Histogram
    axes[3].hist(s, bins=60, range=(0, 1), color='green', edgecolor='black', alpha=0.7)
    axes[3].set_title("Hidden Saturation Distance")
    axes[3].set_xlabel("Distance $d(h)$ (0 = Saturated)")
    axes[3].set_ylabel("Frequency")

    # 5 & 6. GRU Gate Saturations (if applicable)
    if has_gates:
        z_gate = z["gate_z_sat_time"][last_idx]
        r_gate = z["gate_r_sat_time"][last_idx]
        z_gate = z_gate[np.isfinite(z_gate)]
        r_gate = r_gate[np.isfinite(r_gate)]

        axes[4].hist(z_gate, bins=60, range=(0, 0.5), color='cyan', edgecolor='black', alpha=0.7)
        axes[4].set_title("Update Gate ($z_t$) Saturation")
        axes[4].set_xlabel("Distance $d(z)$ (0 = Saturated)")
        axes[4].set_ylabel("Frequency")

        axes[5].hist(r_gate, bins=60, range=(0, 0.5), color='magenta', edgecolor='black', alpha=0.7)
        axes[5].set_title("Reset Gate ($r_t$) Saturation")
        axes[5].set_xlabel("Distance $d(r)$ (0 = Saturated)")
        axes[5].set_ylabel("Frequency")
    
    # Hide any unused subplots
    for i in range(num_plots, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save the plot
    save_name = filename.replace(".npz", "_plots.png")
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {save_name}")
    plt.close()

if __name__ == "__main__":
    # List of files you generated from the terminal commands
    files_to_plot = [
        "A1_mem_rnn_tanh_noclip_final_state.npz",
        "A2_mem_rnn_tanh_clip005_final_state.npz",
        "A3_mem_rnn_tanh_clip001_final_state.npz",
        "A4_mem_gru_noclip_final_state.npz",
        "A5_mem_gru_clip005_final_state.npz",
        "B1_mul_rnn_tanh_noclip_final_state.npz",
        "B2_mul_gru_noclip_final_state.npz"
    ]
    
    for f in files_to_plot:
        plot_run(f)