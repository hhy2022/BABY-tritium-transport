import numpy as np
import festim as F
import h_transport_materials as htm
import requests
import ufl
from dolfinx import fem
from dolfinx.io import gmsh as gmshio
from dolfinx.log import LogLevel, set_log_level
from mpi4py import MPI

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
# The 2D mesh is an (r, z) cross-section; rotating it 2*pi recovers the full 3D flux.
# ---------------------------------------------------------------------------


class CylindricalSurfaceFlux(F.SurfaceFlux):
    """
    Surface flux for a 2D axisymmetric (r, z) mesh.

    Computes:
        J = integral(-D * grad(c) . n * r  dS)

    where r = x[0] is the radial coordinate and azimuth_range covers the full
    revolution by default (0 to 2*pi).
    """

    azimuth_range: tuple = (0.0, 2 * np.pi)

    def __init__(self, field, surface, filename, volume_subdomain):
        super().__init__(field=field, surface=surface, filename=filename)
        self.volume_subdomain = volume_subdomain

    @property
    def title(self):
        return f"{self.field.name} cylindrical flux surface {self.surface.id}"

    def compute(self, u, ds, entity_maps):
        from scifem import assemble_scalar

        if isinstance(u, ufl.indexed.Indexed):
            mesh = self.field.sub_function_space.mesh
        else:
            mesh = u.function_space.mesh

        n = ufl.FacetNormal(mesh)
        x = ufl.SpatialCoordinate(mesh)
        r = x[0]  # radial coordinate

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
# Fetch irradiation time from experimental data
# ---------------------------------------------------------------------------


def get_total_irradiation_time() -> float:
    """Fetch total irradiation duration (seconds) from the experimental data repository."""
    from libra_toolbox.tritium.model import ureg

    url = "https://raw.githubusercontent.com/LIBRA-project/BABY-1L-run-1/refs/tags/v0.5/data/processed_data.json"
    data = requests.get(url).json()
    duration = 0
    for irr in data["irradiations"]:
        start = irr["start_time"]["value"] * ureg(irr["start_time"]["unit"])
        end = irr["stop_time"]["value"] * ureg(irr["stop_time"]["unit"])
        duration += (end - start).to(ureg("s")).magnitude
    return duration


irradiation_time = get_total_irradiation_time()

# ---------------------------------------------------------------------------
# Material properties from h_transport_materials
# ---------------------------------------------------------------------------

# --- FLiBe (CLLiF) ---
htm_D_flibe = htm.diffusivities.filter(material="flibe").filter(author="calderoni")
htm_S_flibe = htm.solubilities.filter(material="flibe").filter(author="calderoni")

flibe_D_0 = htm_D_flibe[0].pre_exp.magnitude
flibe_E_D = htm_D_flibe[0].act_energy.magnitude
flibe_S_0 = htm_S_flibe[0].pre_exp.magnitude
flibe_E_S = htm_S_flibe[0].act_energy.magnitude

# --- Inconel 625 ---
htm_D_inconel = htm.diffusivities.filter(material="inconel_625")
htm_S_inconel = htm.solubilities.filter(material="inconel_625")
htm_recomb_inconel = htm.recombination_coeffs.filter(material="inconel_625")

inconel_D_0 = htm_D_inconel[0].pre_exp.magnitude
inconel_E_D = htm_D_inconel[0].act_energy.magnitude
inconel_S_0 = htm_S_inconel[0].pre_exp.magnitude
inconel_E_S = htm_S_inconel[0].act_energy.magnitude

inconel_Kr_0 = htm_recomb_inconel[1].pre_exp.magnitude
inconel_E_Kr = htm_recomb_inconel[1].act_energy.magnitude

# -------------------------------------------------------------------
# estimate the penalty term for the CLLiF-Inconel interface based on the diffusivity and mesh size
# -------------------------------------------------------------------

T = 650 + 273.15  # K
h = 0.001  # mesh size at interface (your mesh_size=0.001)

# FLiBe
D_flibe = flibe_D_0 * np.exp(-flibe_E_D / (8.617e-5 * T))
K_flibe = flibe_S_0 * np.exp(-flibe_E_S / (8.617e-5 * T))

# Inconel
D_inconel = inconel_D_0 * np.exp(-inconel_E_D / (8.617e-5 * T))
K_inconel = inconel_S_0 * np.exp(-inconel_E_S / (8.617e-5 * T))

print(
    f"Estimated diffusivity at interface: D_flibe={D_flibe:.2e} m^2/s, D_inconel={D_inconel:.2e} m^2/s"
)
print(
    f"Estimated solubility at interface: K_flibe={K_flibe:.2e} H/m^3/Pa^0.5, K_inconel={K_inconel:.2e} H/m^3/Pa^0.5"
)
exit()

# # penalty_factor = 100  # safety factor to ensure stability
# # penalty = penalty_factor * max(D_flibe * K_flibe / h, D_inconel * K_inconel / h)

penalty = 1e21


# ---------------------------------------------------------------------------
# Tritium source term: constant during irradiation, zero afterwards
# ---------------------------------------------------------------------------
def tritium_source(t):
    """Tritium production rate [H/m^3/s]: constant during irradiation, zero afterwards."""
    return 2.19e8 if t < irradiation_time else 0.0


# ---------------------------------------------------------------------------
# Build the FESTIM model
# ---------------------------------------------------------------------------


def build_model(results_folder: str = "results/baby_2d"):

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
        solubility_law="sievert",
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
    # Heater void boundaries on the CLLiF side: explicitly set to zero flux.
    # Replace with actual heater BC when the heater is re-enabled.
    heater_cap_bc = F.SurfaceSubdomain(id=ID_HEATER_CAP)
    liquid_heater_interface = F.SurfaceSubdomain(id=ID_LIQUID_HEATER_IFACE)
    # Inconel outer wall split into three surfaces (matching the 3D model structure)
    inconel_outer_bottom = F.SurfaceSubdomain(id=ID_INCONEL_OUTER_BOTTOM)
    inconel_outer_side = F.SurfaceSubdomain(id=ID_INCONEL_OUTER_SIDE)
    inconel_outer_top = F.SurfaceSubdomain(id=ID_INCONEL_OUTER_TOP)

    # --- Discontinuous interface (CLLiF <-> Inconel) ---
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
        heater_cap_bc,  # top face of heater void (currently zero flux)
        liquid_heater_interface,  # side face of heater void (currently zero flux)
        inconel_outer_bottom,
        inconel_outer_side,
        inconel_outer_top,
    ]
    model.interfaces = [iface_liquid_inconel]

    # Map every outer surface to the volume it belongs to
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

    # --- Boundary conditions ---
    model.boundary_conditions = [
        # CLLiF free surface: fixed zero concentration (tritium released to atmosphere)
        F.FixedConcentrationBC(
            subdomain=liquid_surface,
            species=T,
            value=0.0,
        ),
        # # heater cap is assumed to be zero flux for now
        # F.ParticleFluxBC(
        #     subdomain=heater_cap_bc,
        #     species=T,
        #     value=0,
        # ),
        # # heater-liquid interface is assumed to be zero flux for now
        # F.ParticleFluxBC(
        #     subdomain=liquid_heater_interface,
        #     species=T,
        #     value=0,
        # ),
    ]

    # --- Temperature (uniform, 650 degC) ---
    model.temperature = 650 + 273.15  # K

    # --- Time stepping ---
    dt = F.Stepsize(
        initial_value=10,  # seconds
        growth_factor=1.05,
        cutback_factor=0.9,
        target_nb_iterations=4,
        milestones=[irradiation_time],
    )

    # --- Solver settings ---
    model.settings = F.Settings(
        transient=True,
        atol=1e-8,
        rtol=1e-8,
        final_time=60 * 24 * 3600,  # 60 days in seconds
        stepsize=dt,
    )

    # --- Exports ---
    model.exports = [
        # Concentration fields (VTX format for ParaView)
        F.VTXSpeciesExport(
            filename=f"{results_folder}/T_cllif.bp",
            field=T,
            subdomain=vol_cllif,
        ),
        F.VTXSpeciesExport(
            filename=f"{results_folder}/T_inconel.bp",
            field=T,
            subdomain=vol_inconel,
        ),
        # Surface fluxes computed from -D grad(c) . n * r (cylindrical)
        CylindricalSurfaceFlux(
            field=T,
            surface=liquid_surface,
            filename=f"{results_folder}/flux_liquid_surface.csv",
            volume_subdomain=vol_cllif,
        ),
        CylindricalSurfaceFlux(
            field=T,
            surface=top_cap,
            filename=f"{results_folder}/flux_inconel_top_cap.csv",
            volume_subdomain=vol_inconel,
        ),
        CylindricalSurfaceFlux(
            field=T,
            surface=gap_sidewall,
            filename=f"{results_folder}/flux_gap_sidewall.csv",
            volume_subdomain=vol_inconel,
        ),
        CylindricalSurfaceFlux(
            field=T,
            surface=inconel_outer_bottom,
            filename=f"{results_folder}/flux_inconel_outer_bottom.csv",
            volume_subdomain=vol_inconel,
        ),
        CylindricalSurfaceFlux(
            field=T,
            surface=inconel_outer_side,
            filename=f"{results_folder}/flux_inconel_outer_side.csv",
            volume_subdomain=vol_inconel,
        ),
        CylindricalSurfaceFlux(
            field=T,
            surface=inconel_outer_top,
            filename=f"{results_folder}/flux_inconel_outer_top.csv",
            volume_subdomain=vol_inconel,
        ),
        # Integrated tritium inventory per volume region
        F.TotalVolume(
            field=T,
            volume=vol_cllif,
            filename=f"{results_folder}/inventory_cllif.csv",
        ),
        F.TotalVolume(
            field=T,
            volume=vol_inconel,
            filename=f"{results_folder}/inventory_inconel.csv",
        ),
    ]

    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    os.makedirs("results/baby_2d", exist_ok=True)

    # set_log_level(LogLevel.INFO)

    model = build_model(results_folder="results/baby_2d")  # change folder as needed
    model.initialise()
    model.run()
