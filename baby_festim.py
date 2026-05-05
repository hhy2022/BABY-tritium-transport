import numpy as np
import festim as F
import h_transport_materials as htm
import requests
import ufl
from dolfinx import fem
from dolfinx.io import gmsh as gmshio
from dolfinx.log import LogLevel, set_log_level
from mpi4py import MPI
import os
import gc

# ---------------------------------------------------------------------------
# Physical-group ID constants
# ---------------------------------------------------------------------------

# --- Volume regions  ---
ID_INCONEL = 1
ID_CLLIF = 3

# --- Boundary surfaces  ---
ID_LEFT_SYM_LIQUID = 21  # symmetry axis through the liquid region
ID_LEFT_SYM_INCONEL = 22  # symmetry axis through the Inconel bottom
ID_TOP_CAP = 11  # top face of the Inconel cap (inner, facing gap)
ID_GAP_SIDEWALL = 12  # inner side of Inconel wall facing the gap
ID_LIQUID_SURFACE = 13  # top free surface of the CLLiF melt
ID_LIQUID_INCONEL_IFACE = 99  # auto-computed CLLiF <-> Inconel interface

ID_HEATER_CAP = 14  # top face of heater void (CLLiF upper region)
ID_LIQUID_HEATER_IFACE = 16  # side face of heater void (CLLiF <-> heater interface)
# Inconel outer wall split into three separate groups (mirroring the 3D model)
ID_INCONEL_OUTER_BOTTOM = 31  # bottom face of Inconel vessel (z=y_he_top)
ID_INCONEL_OUTER_SIDE = 32  # outer cylindrical side wall (r=r_inconel)
ID_INCONEL_OUTER_TOP = 33  # top face of Inconel cap (z=y_IV_top)

# ---------------------------------------------------------------------------
# Cylindrical surface flux: J = int(-D grad(c) . n * r dS)
# ---------------------------------------------------------------------------


class CylindricalSurfaceFlux(F.SurfaceFlux):
    """
    Surface flux for a 2D axisymmetric (r, z) mesh.

    Computes:
        J = integral(-D * grad(c) . n * r dS) * 2π
    """

    azimuth_range: tuple = (0.0, 2 * np.pi)

    def __init__(self, field, surface, filename, name=None):
        super().__init__(field=field, surface=surface, filename=filename)
        self._name = name

    @property
    def title(self):
        label = self._name if self._name else f"surface {self.surface.id}"
        return f"{self.field.name} flux {label}"

    def compute(self, u, ds, entity_maps=None):
        from scifem import assemble_scalar

        if isinstance(u, ufl.indexed.Indexed):
            mesh = self.field.sub_function_space.mesh
        else:
            mesh = u.function_space.mesh

        n = ufl.FacetNormal(mesh)
        x = ufl.SpatialCoordinate(mesh)
        r = x[0]

        flux = assemble_scalar(
            fem.form(
                -self.D * r * ufl.dot(ufl.grad(u), n) * ds(self.surface.id),
                entity_maps=entity_maps,
            )
        )
        flux *= self.azimuth_range[1] - self.azimuth_range[0]

        self.value = flux
        self.data.append(self.value)


# ---------------------------------------------------------------------------
# Surface flux computed from the recombination equation.
# This is the PHYSICAL release rate (always >= 0) and is the right quantity
# to integrate for cumulative tritium release.
#
# He case:  J = Kr * c^2
# H2 case:  J = Kr * c^2 + Kr * c_H2 * c
# ---------------------------------------------------------------------------


class CylindricalSurfaceFluxFromEquation(F.SurfaceFlux):
    """
    Cylindrical surface flux computed from the recombination equation.

    For sweep_gas == "He":
        J = integral(Kr * c^2 * r  dS) * 2*pi

    For sweep_gas == "H2":
        J = integral( (Kr * c^2 + Kr * c_H2 * c) * r  dS) * 2*pi
    """

    azimuth_range: tuple = (0.0, 2 * np.pi)

    def __init__(
        self,
        field,
        surface,
        filename,
        volume_subdomain,
        inconel_Kr_0,
        inconel_E_Kr,
        temperature,
        h2_conc=0.0,  # H2 concentration [m^-3]; 0.0 for pure He case
        name=None,
    ):
        super().__init__(field=field, surface=surface, filename=filename)
        self.volume_subdomain = volume_subdomain
        self.inconel_Kr_0 = inconel_Kr_0
        self.inconel_E_Kr = inconel_E_Kr
        self.temperature = temperature
        self.h2_conc = h2_conc
        self._name = name

    @property
    def title(self):
        label = self._name if self._name else f"surface {self.surface.id}"
        return f"{self.field.name} recomb-eq flux {label}"

    def compute(self, u, ds, entity_maps):
        from scifem import assemble_scalar

        if isinstance(u, ufl.indexed.Indexed):
            mesh = self.field.sub_function_space.mesh
        else:
            mesh = u.function_space.mesh

        x = ufl.SpatialCoordinate(mesh)
        r = x[0]

        Kr = self.inconel_Kr_0 * ufl.exp(
            -self.inconel_E_Kr / (F.k_B * self.temperature)
        )

        # Physical release: T+T -> T2  plus  T+H -> HT (if H2 present)
        # Both terms are non-negative since c >= 0 and h2_conc >= 0
        integrand = Kr * u**2 + Kr * self.h2_conc * u

        flux = assemble_scalar(
            fem.form(
                integrand * r * ds(self.surface.id),
                entity_maps=entity_maps,
            )
        )
        flux *= self.azimuth_range[1] - self.azimuth_range[0]

        self.value = flux
        self.data.append(self.value)


# ---------------------------------------------------------------------------
# Fetch irradiation time
# ---------------------------------------------------------------------------
RUN_URLS = {
    1: "https://raw.githubusercontent.com/LIBRA-project/BABY-1L-run-1/refs/tags/v0.6/data/processed_data.json",
    2: "https://raw.githubusercontent.com/LIBRA-project/BABY-1L-run-2/refs/tags/v0.5/data/processed_data.json",
    3: "https://raw.githubusercontent.com/LIBRA-project/BABY-1L-run-3/refs/tags/v0.2/data/processed_data.json",
    4: "https://raw.githubusercontent.com/LIBRA-project/BABY-1L-run-4/refs/tags/v0.1/data/processed_data.json",
}


def get_total_irradiation_time(run_id: int) -> float:
    from libra_toolbox.tritium.model import ureg

    url = RUN_URLS[run_id]
    data = requests.get(url).json()
    duration = 0
    for irr in data["irradiations"]:
        start = irr["start_time"]["value"] * ureg(irr["start_time"]["unit"])
        end = irr["stop_time"]["value"] * ureg(irr["stop_time"]["unit"])
        duration += (end - start).to(ureg("s")).magnitude
    return duration


# Total tritium production per run [particles] — Table 2 from BABY-1L paper, last column
TRITIUM_PRODUCTION = {
    1: 9.45e9,
    2: 5.36e10,
    3: 1.66e10,
    4: 1.81e10,
}

# CLLiF volume [m^3] — nominal 1 L
V_CLLIF = 1.0e-3

# ---------------------------------------------------------------------------
# Material properties
# ---------------------------------------------------------------------------

htm_D_flibe = htm.diffusivities.filter(material="flibe").filter(author="calderoni")
htm_S_flibe = htm.solubilities.filter(material="flibe").filter(author="calderoni")

flibe_D_0 = htm_D_flibe[0].pre_exp.magnitude * 0.2
flibe_E_D = htm_D_flibe[0].act_energy.magnitude
flibe_S_0 = htm_S_flibe[0].pre_exp.magnitude
flibe_E_S = htm_S_flibe[0].act_energy.magnitude

htm_D_inconel = htm.diffusivities.filter(material="inconel_625")
htm_S_inconel = htm.solubilities.filter(material="inconel_625")
htm_recomb_inconel = htm.recombination_coeffs.filter(material="inconel_625")

inconel_D_0 = htm_D_inconel[0].pre_exp.magnitude * 10
inconel_E_D = htm_D_inconel[0].act_energy.magnitude
inconel_S_0 = htm_S_inconel[0].pre_exp.magnitude
inconel_E_S = htm_S_inconel[0].act_energy.magnitude

inconel_Kr_0 = htm_recomb_inconel[1].pre_exp.magnitude
inconel_E_Kr = htm_recomb_inconel[1].act_energy.magnitude

# -------------------------------------------------------------------
# Penalty term + solver tolerances
# -------------------------------------------------------------------

temperature_K = 650 + 273.15
# h = 0.001

# D_flibe = flibe_D_0 * np.exp(-flibe_E_D / (8.617e-5 * temperature_K))
# K_flibe = flibe_S_0 * np.exp(-flibe_E_S / (8.617e-5 * temperature_K))
# D_inconel = inconel_D_0 * np.exp(-inconel_E_D / (8.617e-5 * temperature_K))
# K_inconel = inconel_S_0 * np.exp(-inconel_E_S / (8.617e-5 * temperature_K))

penalty = 1e31
atol = 1e-6
rtol = 1e-6


# ---------------------------------------------------------------------------
# H2 sweep gas concentration (used by both BC and recomb-eq flux)
# ---------------------------------------------------------------------------


def compute_h2_conc():
    """Compute H2 number density in the sweep gas [m^-3]."""
    h2_P_gauge = 3  # psi (gauge)
    h2_conc_ppm = 1000  # ppm H2 in sweep gas
    mole_frac_h2 = h2_conc_ppm / 1e6
    P_atm = 14.7
    P_abs = h2_P_gauge + P_atm
    P_h2 = mole_frac_h2 * P_abs
    P_h2 *= 6894.76  # psi -> Pa
    gas_constant = 8.314
    T_room = 298
    h2_conc_mol = P_h2 / (gas_constant * T_room)  # mol/m^3
    h2_conc = h2_conc_mol * 6.022e23  # m^-3
    return h2_conc


# ---------------------------------------------------------------------------
# Build the FESTIM model
# ---------------------------------------------------------------------------


def build_model(sweep_gas: str, run_id: int, results_folder: str = "results/baby_2d"):
    """
    Build the 2D axisymmetric FESTIM model.

    sweep_gas : "He" or "H2"
    """

    irradiation_time = get_total_irradiation_time(run_id)

    # Average volumetric production rate during irradiation [T / m^3 / s]
    source_strength = TRITIUM_PRODUCTION[run_id] / (irradiation_time * V_CLLIF)
    # print(f"  Run {run_id}: source = {source_strength:.3e} T/m^3/s")
    # exit(0)

    def tritium_source(t):
        return source_strength if t < irradiation_time else 0.0

    model = F.HydrogenTransportProblemDiscontinuous()

    # --- Mesh ---
    _read = gmshio.read_from_msh("baby_2d.msh", MPI.COMM_WORLD, 0, gdim=2)
    mesh = _read.mesh
    cell_tags = _read.cell_tags
    facet_tags = _read.facet_tags
    model.mesh = F.Mesh(mesh=mesh, coordinate_system="cylindrical")
    model.facet_meshtags = facet_tags
    model.volume_meshtags = cell_tags

    # --- Materials ---
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

    # --- Volume subdomains ---
    vol_inconel = F.VolumeSubdomain(id=ID_INCONEL, material=mat_inconel)
    vol_cllif = F.VolumeSubdomain(id=ID_CLLIF, material=mat_cllif)

    # --- Surface subdomains ---
    left_sym_liquid = F.SurfaceSubdomain(id=ID_LEFT_SYM_LIQUID)
    left_sym_inconel = F.SurfaceSubdomain(id=ID_LEFT_SYM_INCONEL)
    top_cap = F.SurfaceSubdomain(id=ID_TOP_CAP)
    gap_sidewall = F.SurfaceSubdomain(id=ID_GAP_SIDEWALL)
    liquid_surface = F.SurfaceSubdomain(id=ID_LIQUID_SURFACE)
    heater_cap_bc = F.SurfaceSubdomain(id=ID_HEATER_CAP)
    liquid_heater_interface = F.SurfaceSubdomain(id=ID_LIQUID_HEATER_IFACE)
    inconel_outer_bottom = F.SurfaceSubdomain(id=ID_INCONEL_OUTER_BOTTOM)
    inconel_outer_side = F.SurfaceSubdomain(id=ID_INCONEL_OUTER_SIDE)
    inconel_outer_top = F.SurfaceSubdomain(id=ID_INCONEL_OUTER_TOP)

    # --- Discontinuous interface ---
    iface_liquid_inconel = F.Interface(
        id=ID_LIQUID_INCONEL_IFACE,
        subdomains=[vol_cllif, vol_inconel],
        penalty_term=penalty,
    )

    model.subdomains = [
        vol_inconel,
        vol_cllif,
        left_sym_liquid,
        left_sym_inconel,
        top_cap,
        gap_sidewall,
        liquid_surface,
        heater_cap_bc,
        liquid_heater_interface,
        inconel_outer_bottom,
        inconel_outer_side,
        inconel_outer_top,
    ]
    model.interfaces = [iface_liquid_inconel]

    model.surface_to_volume = {
        left_sym_liquid: vol_cllif,
        left_sym_inconel: vol_inconel,
        top_cap: vol_inconel,
        gap_sidewall: vol_inconel,
        liquid_surface: vol_cllif,
        heater_cap_bc: vol_cllif,
        liquid_heater_interface: vol_cllif,
        inconel_outer_bottom: vol_inconel,
        inconel_outer_side: vol_inconel,
        inconel_outer_top: vol_inconel,
    }

    # --- Species ---
    T = F.Species("T", mobile=True, subdomains=[vol_cllif, vol_inconel])
    model.species = [T]

    # --- Source ---
    model.sources = [
        F.ParticleSource(value=tritium_source, volume=vol_cllif, species=T),
    ]

    # -----------------------------------------------------------------------
    # Recombination boundary conditions on outer Inconel surfaces
    # -----------------------------------------------------------------------

    if sweep_gas == "H2":
        h2_conc = compute_h2_conc()

        def recombination_flux(c, T):
            Kr = inconel_Kr_0 * ufl.exp(-inconel_E_Kr / (F.k_B * T))
            return -Kr * c**2 - Kr * h2_conc * c

    elif sweep_gas == "He":
        h2_conc = 0.0

        def recombination_flux(c, T):
            Kr = inconel_Kr_0 * ufl.exp(-inconel_E_Kr / (F.k_B * T))
            return -Kr * c**2

    else:
        raise ValueError(f"Unknown sweep_gas '{sweep_gas}'. Use 'He' or 'H2'.")

    subfolder = f"{results_folder}/run_{run_id}/sweep_{sweep_gas}"

    # outer_inconel_surfaces = [
    #     inconel_outer_bottom,
    #     inconel_outer_side,
    #     inconel_outer_top,
    # ]

    recomb_bcs = []
    # for surf in outer_inconel_surfaces:
    #     bc = F.ParticleFluxBC(
    #         value=recombination_flux,
    #         subdomain=surf,
    #         species_dependent_value={"c": T},
    #         species=T,
    #     )
    #     bc._volume_subdomain = vol_inconel
    #     recomb_bcs.append(bc)

    inner_vessel_surfaces = [liquid_surface, gap_sidewall, top_cap]

    model.boundary_conditions = [
        *[
            F.FixedConcentrationBC(subdomain=surf, species=T, value=0.0)
            for surf in inner_vessel_surfaces
        ],
        *recomb_bcs,
    ]

    model.temperature = temperature_K

    dt = F.Stepsize(
        initial_value=10,
        growth_factor=1.1,
        cutback_factor=0.9,
        target_nb_iterations=4,
        milestones=[irradiation_time],
    )

    model.settings = F.Settings(
        transient=True,
        atol=atol,
        rtol=rtol,
        final_time=60 * 24 * 3600,
        stepsize=dt,
    )

    # -----------------------------------------------------------------------
    # Exports
    # -----------------------------------------------------------------------

    # Helper to build a recomb-eq flux export for a given outer surface
    def make_recomb_eq_export(surface, name, filename):
        return CylindricalSurfaceFluxFromEquation(
            field=T,
            surface=surface,
            filename=filename,
            volume_subdomain=vol_inconel,
            inconel_Kr_0=inconel_Kr_0,
            inconel_E_Kr=inconel_E_Kr,
            temperature=temperature_K,
            h2_conc=h2_conc,  # 0.0 for He, real value for H2
            name=name,
        )

    model.exports = [
        # Concentration fields
        F.VTXSpeciesExport(
            filename=f"{subfolder}/T_cllif.bp", field=T, subdomain=vol_cllif
        ),
        F.VTXSpeciesExport(
            filename=f"{subfolder}/T_inconel.bp", field=T, subdomain=vol_inconel
        ),
        # ---- Gradient-based fluxes (-D grad c . n) on every surface ----
        CylindricalSurfaceFlux(
            field=T,
            surface=liquid_surface,
            filename=f"{subfolder}/flux_liquid_surface.csv",
            name="liquid surface",
        ),
        CylindricalSurfaceFlux(
            field=T,
            surface=top_cap,
            filename=f"{subfolder}/flux_inconel_top_cap.csv",
            name="Inconel top cap",
        ),
        CylindricalSurfaceFlux(
            field=T,
            surface=gap_sidewall,
            filename=f"{subfolder}/flux_gap_sidewall.csv",
            name="Inconel gap sidewall",
        ),
        # CylindricalSurfaceFlux(
        #     field=T,
        #     surface=inconel_outer_bottom,
        #     filename=f"{subfolder}/flux_inconel_outer_bottom.csv",
        #     name="Inconel outer bottom",
        # ),
        # CylindricalSurfaceFlux(
        #     field=T,
        #     surface=inconel_outer_side,
        #     filename=f"{subfolder}/flux_inconel_outer_side.csv",
        #     name="Inconel outer side",
        # ),
        # CylindricalSurfaceFlux(
        #     field=T,
        #     surface=inconel_outer_top,
        #     filename=f"{subfolder}/flux_inconel_outer_top.csv",
        #     name="Inconel outer top",
        # ),
        # # ---- Recomb-eq (physical release) fluxes on all 3 outer surfaces ----
        # make_recomb_eq_export(
        #     inconel_outer_bottom,
        #     "Inconel outer bottom",
        #     f"{subfolder}/flux_inconel_outer_bottom_recomb_eq.csv",
        # ),
        # make_recomb_eq_export(
        #     inconel_outer_side,
        #     "Inconel outer side",
        #     f"{subfolder}/flux_inconel_outer_side_recomb_eq.csv",
        # ),
        # make_recomb_eq_export(
        #     inconel_outer_top,
        #     "Inconel outer top",
        #     f"{subfolder}/flux_inconel_outer_top_recomb_eq.csv",
        # ),
        # ---- Tritium inventory per region ----
        F.TotalVolume(
            field=T, volume=vol_cllif, filename=f"{subfolder}/inventory_cllif.csv"
        ),
        F.TotalVolume(
            field=T, volume=vol_inconel, filename=f"{subfolder}/inventory_inconel.csv"
        ),
    ]

    return model
    # return model, T, vol_cllif, vol_inconel


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    for run_id in [1, 2]:
        # for run_id in [1, 2, 4]:
        print(f"\n=== Run {run_id} / He sweep ===")
        model = build_model(sweep_gas="He", run_id=run_id)
        # model, T, vol_cllif, vol_inconel = build_model(sweep_gas="He", run_id=run_id)
        model.initialise()
        model.run()

        # from dolfinx import geometry
        # import numpy as np

        # u_flibe = T.subdomain_to_post_processing_solution[vol_cllif]
        # u_inconel = T.subdomain_to_post_processing_solution[vol_inconel]
        # r_iface = 0.07
        # z_test = 0.0558

        # mesh_flibe = vol_cllif.submesh
        # mesh_inconel = vol_inconel.submesh

        # bb_tree_flibe = geometry.bb_tree(mesh_flibe, mesh_flibe.topology.dim)
        # bb_tree_inconel = geometry.bb_tree(mesh_inconel, mesh_inconel.topology.dim)

        # for export in model.exports:
        #     if hasattr(export, "data") and len(export.data) > 0:
        #         print(f"{export.title}: {export.data[-1]:.3e}")

        # def eval_at(u, bb_tree, mesh, r, z):
        #     pt = np.array([[r, z, 0.0]])
        #     candidates = geometry.compute_collisions_points(bb_tree, pt)
        #     cells = geometry.compute_colliding_cells(mesh, candidates, pt)
        #     return u.eval(pt, np.array([cells.links(0)[0]]))[0]

        # c_l = eval_at(u_flibe, bb_tree_flibe, mesh_flibe, r_iface, z_test)
        # c_r = eval_at(u_inconel, bb_tree_inconel, mesh_inconel, r_iface, z_test)

        # print(f"c_flibe          = {c_l}")
        # print(f"c_inconel        = {c_r}")
        # print(f"c_henry/K_H      = {c_l / K_flibe}")
        # print(f"(c_sievert/K_S)^2 = {(c_r / K_inconel) ** 2}")
        # print(
        #     f"henry ratio            = {(c_l / K_flibe) / (c_r / K_inconel) ** 2:.4e}"
        # )

        del model
        gc.collect()
    # # ---- He sweep ----
    # model = build_model(sweep_gas="He")
    # model.initialise()
    # model.run()
    # del model
    # gc.collect()

    # # ---- H2 sweep ----
    # model = build_model(sweep_gas="H2")
    # model.initialise()
    # model.run()
    # del model
    # gc.collect()
