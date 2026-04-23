"""
1D Hydrogen Transport Model: CLLiF (liquid) | Inconel (solid)
==============================================================

Geometry (x-axis):
    [0, L_liquid]                : CLLiF melt  (x = 0 → liquid top surface)
    [L_liquid, L_liquid+L_solid] : Inconel 625 (x = L_liquid+L_solid → solid bottom)

Boundary conditions:
    x = 0          : c = 0  (tritium released to gap / free surface)
    x = L_liquid   : discontinuous interface (CLLiF <-> Inconel)
                     henry law (CLLiF) | sievert law (Inconel)
    x = L_liquid + L_solid : selectable via right_bc argument:
                     "zero_flux"     — default Neumann (no action)
                     "dirichlet"     — fixed c = 0
                     "recomb_he"     — J = -Kr * c²          (He sweep gas)
                     "recomb_h2"     — J = -Kr * c² - Kr * c_H2 * c  (H2 sweep gas)
                     "recomb_large"  — Kr x 1e48 (Kr → ∞ limit test)
"""

import numpy as np
import festim as F
import h_transport_materials as htm
import requests
import ufl
import os
import gc
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
L_liquid = 0.07  # m  CLLiF melt thickness
L_solid = 0.003  # m  Inconel 625 wall thickness

ID_LIQUID = 1
ID_SOLID = 2
ID_LEFT = 1
ID_IFACE = 2
ID_RIGHT = 3

# ---------------------------------------------------------------------------
# Material properties
# ---------------------------------------------------------------------------
Na = 6.022e23

htm_D_flibe = htm.diffusivities.filter(material="flibe").filter(author="calderoni")
htm_S_flibe = htm.solubilities.filter(material="flibe").filter(author="calderoni")
htm_D_inconel = htm.diffusivities.filter(material="inconel_625")
htm_S_inconel = htm.solubilities.filter(material="inconel_625")
htm_Kr_inconel = htm.recombination_coeffs.filter(material="inconel_625")

flibe_D_0 = htm_D_flibe[0].pre_exp.magnitude
flibe_E_D = htm_D_flibe[0].act_energy.magnitude
flibe_S_0 = htm_S_flibe[0].pre_exp.magnitude
flibe_E_S = htm_S_flibe[0].act_energy.magnitude

inconel_D_0 = htm_D_inconel[0].pre_exp.magnitude
inconel_E_D = htm_D_inconel[0].act_energy.magnitude
inconel_S_0 = htm_S_inconel[0].pre_exp.magnitude
inconel_E_S = htm_S_inconel[0].act_energy.magnitude
inconel_Kr_0 = htm_Kr_inconel[1].pre_exp.magnitude
inconel_E_Kr = htm_Kr_inconel[1].act_energy.magnitude

# ---------------------------------------------------------------------------
# Temperature
# ---------------------------------------------------------------------------
temperature_K = 650 + 273.15  # K

D_fl = flibe_D_0 * np.exp(-flibe_E_D / (F.k_B * temperature_K))
K_fl = flibe_S_0 * np.exp(-flibe_E_S / (F.k_B * temperature_K))
D_in = inconel_D_0 * np.exp(-inconel_E_D / (F.k_B * temperature_K))
K_in = inconel_S_0 * np.exp(-inconel_E_S / (F.k_B * temperature_K))
Kr_val = inconel_Kr_0 * np.exp(-inconel_E_Kr / (F.k_B * temperature_K))

# ---------------------------------------------------------------------------
# Penalty
# ---------------------------------------------------------------------------
penalty = 1e35
atol = 1e4
rtol = 1e-10

# ---------------------------------------------------------------------------
# H2 sweep gas concentration
# ---------------------------------------------------------------------------
h2_P_gauge = 3  # psi gauge
h2_conc_ppm = 1000  # ppm H2
P_atm = 14.7  # psi
P_h2 = (h2_conc_ppm / 1e6) * (h2_P_gauge + P_atm) * 6894.76  # Pa
h2_conc = (P_h2 / (8.314 * 298)) * Na  # H/m³


# ---------------------------------------------------------------------------
# Irradiation time
# ---------------------------------------------------------------------------
def get_total_irradiation_time() -> float:
    from libra_toolbox.tritium.model import ureg

    url = "https://raw.githubusercontent.com/LIBRA-project/BABY-1L-run-1/refs/tags/v0.5/data/processed_data.json"
    data = requests.get(url).json()
    duration = 0.0
    for irr in data["irradiations"]:
        start = irr["start_time"]["value"] * ureg(irr["start_time"]["unit"])
        end = irr["stop_time"]["value"] * ureg(irr["stop_time"]["unit"])
        duration += (end - start).to(ureg("s")).magnitude
    return duration


irradiation_time = get_total_irradiation_time()


def tritium_source(t):
    """Tritium production rate [H/m³/s]: constant during irradiation, zero afterwards."""
    return 2.19e8 if t < irradiation_time else 0.0


# ---------------------------------------------------------------------------
# Recombination flux functions
# ---------------------------------------------------------------------------
def recomb_he(c, T):
    """Pure He sweep gas: J = -Kr * c²"""
    Kr = inconel_Kr_0 * ufl.exp(-inconel_E_Kr / (F.k_B * T))
    return -Kr * c**2


def recomb_h2(c, T):
    """H2 sweep gas: J = -Kr * c² - Kr * c_H2 * c"""
    Kr = inconel_Kr_0 * ufl.exp(-inconel_E_Kr / (F.k_B * T))
    return -Kr * c**2 - Kr * h2_conc * c


def recomb_large(c, T):
    """Kr → ∞ limit test: Kr x 1e48"""
    Kr = inconel_Kr_0 * 1e48 * ufl.exp(-inconel_E_Kr / (F.k_B * T))
    return -Kr * c**2


# ---------------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------------
def build_model(right_bc: str = "recomb_he", final_time: float = 1e5):
    """
    Build and return the FESTIM model.

    Parameters
    ----------
    right_bc : str
        Right boundary condition type. One of:
        "zero_flux", "dirichlet", "recomb_he", "recomb_h2", "recomb_large"
    final_time : float
        Simulation end time [s].
    """
    model = F.HydrogenTransportProblemDiscontinuous()

    # Mesh
    vertices = np.unique(
        np.concatenate(
            [
                np.linspace(0, L_liquid, 200),
                np.linspace(L_liquid, L_liquid + L_solid, 50),
            ]
        )
    )
    model.mesh = F.Mesh1D(vertices)

    # Materials
    mat_cllif = F.Material(
        D_0=flibe_D_0,
        E_D=flibe_E_D,
        K_S_0=flibe_S_0,
        E_K_S=flibe_E_S,
        solubility_law="henry",
    )
    mat_inconel = F.Material(
        D_0=inconel_D_0,
        E_D=inconel_E_D,
        K_S_0=inconel_S_0,
        E_K_S=inconel_E_S,
        solubility_law="sievert",
    )

    # Subdomains
    vol_cllif = F.VolumeSubdomain1D(
        id=ID_LIQUID, borders=[0, L_liquid], material=mat_cllif
    )
    vol_inconel = F.VolumeSubdomain1D(
        id=ID_SOLID, borders=[L_liquid, L_liquid + L_solid], material=mat_inconel
    )
    surf_left = F.SurfaceSubdomain1D(id=ID_LEFT, x=0.0)
    surf_right = F.SurfaceSubdomain1D(id=ID_RIGHT, x=L_liquid + L_solid)

    iface = F.Interface(
        id=ID_IFACE, subdomains=[vol_cllif, vol_inconel], penalty_term=penalty
    )

    model.subdomains = [vol_cllif, vol_inconel, surf_left, surf_right]
    model.interfaces = [iface]
    model.surface_to_volume = {surf_left: vol_cllif, surf_right: vol_inconel}

    # Species
    T_sp = F.Species("T", mobile=True, subdomains=[vol_cllif, vol_inconel])
    model.species = [T_sp]

    # Source
    model.sources = [
        F.ParticleSource(value=tritium_source, volume=vol_cllif, species=T_sp)
    ]

    # Boundary conditions
    bc_list = [F.FixedConcentrationBC(subdomain=surf_left, species=T_sp, value=0.0)]

    flux_fn_map = {
        "recomb_he": recomb_he,
        "recomb_h2": recomb_h2,
        "recomb_large": recomb_large,
    }

    if right_bc == "zero_flux":
        pass  # default Neumann: no action needed
    elif right_bc == "dirichlet":
        bc_list.append(
            F.FixedConcentrationBC(subdomain=surf_right, species=T_sp, value=0.0)
        )
    elif right_bc in flux_fn_map:
        bc_list.append(
            F.ParticleFluxBC(
                value=flux_fn_map[right_bc],
                subdomain=surf_right,
                species_dependent_value={"c": T_sp},
                species=T_sp,
            )
        )
    else:
        raise ValueError(
            f"Unknown right_bc='{right_bc}'. Choose from: zero_flux, dirichlet, recomb_he, recomb_h2, recomb_large"
        )

    model.boundary_conditions = bc_list
    model.temperature = temperature_K

    # Time stepping
    model.settings = F.Settings(
        transient=True,
        atol=atol,
        rtol=rtol,
        final_time=final_time,
        stepsize=F.Stepsize(
            initial_value=10,
            growth_factor=1.1,
            cutback_factor=0.9,
            target_nb_iterations=4,
            milestones=[irradiation_time],
        ),
    )

    # Exports
    os.makedirs(f"results_1d/{right_bc}", exist_ok=True)
    model.exports = [
        F.VTXSpeciesExport(
            filename=f"results_1d/{right_bc}/T_cllif.bp",
            field=T_sp,
            subdomain=vol_cllif,
        ),
        F.VTXSpeciesExport(
            filename=f"results_1d/{right_bc}/T_inconel.bp",
            field=T_sp,
            subdomain=vol_inconel,
        ),
        F.TotalVolume(
            field=T_sp,
            volume=vol_cllif,
            filename=f"results_1d/{right_bc}/inventory_cllif.csv",
        ),
        F.TotalVolume(
            field=T_sp,
            volume=vol_inconel,
            filename=f"results_1d/{right_bc}/inventory_inconel.csv",
        ),
        F.SurfaceFlux(
            field=T_sp,
            surface=surf_right,
            filename=f"results_1d/{right_bc}/flux_outer_wall.csv",
        ),
    ]

    return model, T_sp, vol_cllif, vol_inconel


# ---------------------------------------------------------------------------
# Post-process
# ---------------------------------------------------------------------------
def post_process(model, T_sp, vol_cllif, vol_inconel, right_bc):
    from dolfinx import geometry

    u_fl = T_sp.subdomain_to_post_processing_solution[vol_cllif]
    u_in = T_sp.subdomain_to_post_processing_solution[vol_inconel]

    def eval_1d(u, subdomain, x_eval):
        mesh = subdomain.submesh
        bb = geometry.bb_tree(mesh, mesh.topology.dim)
        pt = np.array([[x_eval, 0.0, 0.0]])
        cands = geometry.compute_collisions_points(bb, pt)
        cells = geometry.compute_colliding_cells(mesh, cands, pt)
        return u.eval(pt, np.array([cells.links(0)[0]]))[0]

    c_liq = eval_1d(u_fl, vol_cllif, L_liquid)
    c_sol = eval_1d(u_in, vol_inconel, L_liquid)
    c_wall = eval_1d(u_in, vol_inconel, L_liquid + L_solid)

    # Interface check
    print(f"\n=== Interface check ({right_bc}) ===")
    print(f"  c_CLLiF   (x→L⁻) = {c_liq:.4e}  [H/m³]")
    print(f"  c_Inconel (x→L⁺) = {c_sol:.4e}  [H/m³]")
    print(f"  p_CLLiF   = c/K_H        = {c_liq / K_fl:.4e}  [Pa]")
    print(f"  p_Inconel = (c/K_S)²     = {(c_sol / K_in) ** 2:.4e}  [Pa]")
    print(
        f"  henry check              = {(c_liq / K_fl) / (c_sol / K_in) ** 2:.4e}  (should be ~1)"
    )

    # Right boundary check
    J_expected = -Kr_val * c_wall**2
    flux_data = pd.read_csv(f"results_1d/{right_bc}/flux_outer_wall.csv")
    J_festim = flux_data.iloc[-1, 1]

    print(f"\n=== Right BC check ({right_bc}) ===")
    print(f"  c_right_wall        = {c_wall:.4e}  [H/m³]")
    print(f"  Kr(T)               = {Kr_val:.4e}")
    print(f"  J_expected = -Kr*c² = {J_expected:.4e}  [H/m²/s]")
    print(f"  J_festim            = {J_festim:.4e}  [H/m²/s]")
    print(f"  J_festim        = {J_festim:.4e}  [H/m²/s]")
    print(
        f"  ratio (converted)   = {(J_festim * Na) / J_expected:.4e}  (should be ~-1 for recomb)"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Run all cases and compare
    cases = ["zero_flux", "dirichlet", "recomb_he", "recomb_h2", "recomb_large"]

    for case in cases:
        print(f"\n{'=' * 60}")
        print(f"  Running case: {case}")
        print(f"{'=' * 60}")

        model, T_sp, vol_cllif, vol_inconel = build_model(right_bc=case, final_time=1e5)
        model.initialise()
        model.run()

        post_process(model, T_sp, vol_cllif, vol_inconel, right_bc=case)

        del model
        gc.collect()


# ---------------------------------------------------------------------------
# Plot: inventory CLLiF, inventory Inconel, flux outer wall
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

titles = ["Inventory CLLiF", "Inventory Inconel", "Flux outer wall"]
files = ["inventory_cllif.csv", "inventory_inconel.csv", "flux_outer_wall.csv"]
ylabels = ["Inventory [H/m²]", "Inventory [H/m²]", "Flux [mol/m²/s]"]

for ax, fname, title, ylabel in zip(axes, files, titles, ylabels):
    for case in cases:
        fpath = f"results_1d/{case}/{fname}"
        if not os.path.exists(fpath):
            continue
        df = pd.read_csv(fpath)
        t = df.iloc[:, 0]
        y = df.iloc[:, 1]
        ax.plot(t, y, label=case)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig("results_1d/comparison.png", dpi=150)
plt.show()
print("Plot saved to results_1d/comparison.png")
