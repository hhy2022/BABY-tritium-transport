import pandas as pd
import matplotlib.pyplot as plt
import os

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
BASE_FOLDER = "results/baby_2d"

SWEEPS = {
    "H2": os.path.join(BASE_FOLDER, "sweep_H2"),
    "He": os.path.join(BASE_FOLDER, "sweep_He"),
}

# Add / uncomment the files you want to plot
FILES = {
    "flux_liquid_surface.csv": "Liquid surface",
    # "flux_inconel_top_cap.csv":       "Inconel top cap",
    # "flux_gap_sidewall.csv":          "Inconel gap sidewall",
    # "flux_inconel_outer_bottom.csv":  "Inconel outer bottom",
    "flux_inconel_outer_side.csv": "Inconel outer side",
    # "flux_inconel_outer_top.csv":     "Inconel outer top",
    # "inventory_cllif.csv":            "Inventory CLLiF",
    # "inventory_inconel.csv":          "Inventory Inconel",
}

SECONDS_PER_DAY = 86400  # conversion factor
OUTPUT_FOLDER = BASE_FOLDER  # where .png files are saved


# ─────────────────────────────────────────────
# Helper: plot a single sweep, single keyword
# ─────────────────────────────────────────────
def plot_single_case(
    sweep_name: str,
    keyword: str,
    ylabel: str,
    title: str,
    out_name: str,
):
    """
    Draw one figure for *one* sweep (e.g. "H2") containing every
    CSV file whose name includes `keyword`.

    Time column (seconds) is divided by 86 400 → plotted in days.
    """
    folder = SWEEPS[sweep_name]

    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = False

    for filename, label in FILES.items():
        if keyword not in filename:
            continue
        path = os.path.join(folder, filename)
        if not os.path.exists(path):
            print(f"  [skip] not found: {path}")
            continue

        df = pd.read_csv(path)
        t = df.iloc[:, 0] / SECONDS_PER_DAY  # s → days
        val = df.iloc[:, 1]

        ax.plot(t, val, label=label)
        plotted = True

    if not plotted:
        print(f"  [warn] no data found for sweep={sweep_name}, keyword={keyword}")
        plt.close(fig)
        return

    ax.set_xlabel("Time (days)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}  —  {sweep_name}")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_FOLDER, out_name)
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────
# Helper: plot all sweeps on one combined figure
# ─────────────────────────────────────────────
SWEEP_STYLES = {
    "H2": {"linestyle": "-", "alpha": 1.00},
    "He": {"linestyle": "--", "alpha": 0.85},
}


def plot_combined(
    keyword: str,
    ylabel: str,
    title: str,
    out_name: str,
):
    """
    Draw one figure that overlays **all** sweeps (H2 solid, He dashed).
    Useful for direct comparison.  Time in days.
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    plotted = False

    for sweep_name, folder in SWEEPS.items():
        style = SWEEP_STYLES[sweep_name]
        for filename, label in FILES.items():
            if keyword not in filename:
                continue
            path = os.path.join(folder, filename)
            if not os.path.exists(path):
                continue

            df = pd.read_csv(path)
            t = df.iloc[:, 0] / SECONDS_PER_DAY
            val = df.iloc[:, 1]

            ax.plot(
                t,
                val,
                label=f"{label} [{sweep_name}]",
                linestyle=style["linestyle"],
                alpha=style["alpha"],
            )
            plotted = True

    if not plotted:
        print(f"  [warn] no data found for keyword={keyword}")
        plt.close(fig)
        return

    ax.set_xlabel("Time (days)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}  —  H2 (solid) vs He (dashed)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_FOLDER, out_name)
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # ── Option A: one figure per sweep case ──────────────────────────────
    for sweep_name in SWEEPS:
        plot_single_case(
            sweep_name=sweep_name,
            keyword="flux",
            ylabel="Flux (H/s)",
            title="Surface fluxes",
            out_name=f"fluxes_{sweep_name}.png",
        )

    # Uncomment to also plot inventory per case:
    # for sweep_name in SWEEPS:
    #     plot_single_case(
    #         sweep_name=sweep_name,
    #         keyword="inventory",
    #         ylabel="Inventory (H)",
    #         title="Tritium inventory",
    #         out_name=f"inventory_{sweep_name}.png",
    #     )

    # ── Option B: combined comparison figure (H2 vs He) ──────────────────
    plot_combined(
        keyword="flux",
        ylabel="Flux (H/s)",
        title="Surface fluxes",
        out_name="fluxes_compare.png",
    )

    # plot_combined(
    #     keyword="inventory",
    #     ylabel="Inventory (H)",
    #     title="Tritium inventory",
    #     out_name="inventory_compare.png",
    # )
