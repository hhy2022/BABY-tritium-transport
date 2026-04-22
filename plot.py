import pandas as pd
import matplotlib.pyplot as plt
import os

results_folder = "results_old/baby_2d/sweep_He"

files = {
    "flux_liquid_surface.csv": "Liquid surface",
    "flux_inconel_top_cap.csv": "Inconel top cap",
    "flux_gap_sidewall.csv": "Inconel gap sidewall",
    "flux_inconel_outer_bottom.csv": "Inconel outer bottom",
    "flux_inconel_outer_side.csv": "Inconel outer side",
    "flux_inconel_outer_top.csv": "Inconel outer top",
    "inventory_cllif.csv": "Inventory CLLiF",
    "inventory_inconel.csv": "Inventory Inconel",
}

# --- Plot 1: Surface fluxes ---
fig, ax = plt.subplots(figsize=(10, 6))
for filename, label in files.items():
    if "flux" not in filename:
        continue
    path = os.path.join(results_folder, filename)
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    t = df.iloc[:, 0]
    flux = df.iloc[:, 1]
    ax.plot(t, flux, label=label)

ax.set_xlabel("Time (s)")
ax.set_ylabel("Flux (H/s)")
ax.set_title("Surface fluxes")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig(f"{results_folder}/fluxes.png", dpi=150)
plt.show()

# --- Plot 2: Tritium inventory ---
fig, ax = plt.subplots(figsize=(10, 6))
for filename, label in files.items():
    if "inventory" not in filename:
        continue
    path = os.path.join(results_folder, filename)
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    t = df.iloc[:, 0]
    inv = df.iloc[:, 1]
    ax.plot(t, inv, label=label)

ax.set_xlabel("Time (s)")
ax.set_ylabel("Inventory (H)")
ax.set_title("Tritium inventory")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig(f"{results_folder}/inventory.png", dpi=150)
plt.show()
