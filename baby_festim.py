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
        J = integral(-D * grad(c) . n * r dS) * 2*pi
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
# H2 case:  J = Kr * c^2 + Kr * c_H2(t) * c
#
# The export reads the current simulation time from the model time constant
# bound after model.initialise(). This allows c_H2 to be a time-dependent
# Python function while keeping the exported quantity in one CSV file.
# ---------------------------------------------------------------------------


class CylindricalSurfaceFluxFromEquation(F.SurfaceFlux):
    """
    Cylindrical surface flux computed from the recombination equation.

    For sweep_gas == "He":
        J = integral(Kr * c^2 * r  dS) * 2*pi

    For sweep_gas == "H2":
        J = integral( (Kr * c^2 + Kr * c_H2(t) * c) * r  dS) * 2*pi
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
        h2_conc=0.0,  # H2 concentration [m^-3], or a function h2_conc(t)
        name=None,
    ):
        super().__init__(field=field, surface=surface, filename=filename)
        self.volume_subdomain = volume_subdomain
        self.inconel_Kr_0 = inconel_Kr_0
        self.inconel_E_Kr = inconel_E_Kr
        self.temperature = temperature
        self.h2_conc = h2_conc
        self.time_constant = None
        self._name = name

    @property
    def title(self):
        label = self._name if self._name else f"surface {self.surface.id}"
        return f"{self.field.name} recomb-eq flux {label}"

    def bind_time_constant(self, time_constant):
        self.time_constant = time_constant

    def _current_time(self):
        if self.time_constant is None:
            return None
        value = self.time_constant.value
        return float(value[0] if np.ndim(value) > 0 else value)

    def _current_h2_conc(self):
        if callable(self.h2_conc):
            t = self._current_time()
            if t is None:
                raise RuntimeError(
                    "A time-dependent h2_conc requires bind_time_constant()."
                )
            return float(self.h2_conc(t))
        return float(self.h2_conc)

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

        h2_conc = self._current_h2_conc()

        # Physical release: T+T -> T2  plus  T+H -> HT (if H2 present)
        # Both terms are non-negative since c >= 0 and h2_conc >= 0
        integrand = Kr * u**2 + Kr * h2_conc * u

        flux = assemble_scalar(
            fem.form(
                integrand * r * ds(self.surface.id),
                entity_maps=entity_maps,
            )
        )
        flux *= self.azimuth_range[1] - self.azimuth_range[0]

        self.value = flux
        self.data.append(self.value)


class CylindricalSurfaceFluxMassTransfer(F.SurfaceFlux):
    """
    Cylindrical release flux from a first-order mass-transfer law.

    Boundary law:
        J_release = k * c

    Integrated release:
        R = integral(k * c * 2*pi*r dS)
    """

    azimuth_range: tuple = (0.0, 2 * np.pi)

    def __init__(self, field, surface, filename, k, name=None):
        super().__init__(field=field, surface=surface, filename=filename)
        self.k = k
        self._name = name

    @property
    def title(self):
        label = self._name if self._name else f"surface {self.surface.id}"
        return f"{self.field.name} mass-transfer release {label}"

    def compute(self, u, ds, entity_maps=None):
        from scifem import assemble_scalar

        if isinstance(u, ufl.indexed.Indexed):
            mesh = self.field.sub_function_space.mesh
        else:
            mesh = u.function_space.mesh

        x = ufl.SpatialCoordinate(mesh)
        r = x[0]

        flux = assemble_scalar(
            fem.form(
                self.k * u * r * ds(self.surface.id),
                entity_maps=entity_maps,
            )
        )
        flux *= self.azimuth_range[1] - self.azimuth_range[0]

        self.value = flux
        self.data.append(self.value)


class CylindricalTotalVolume(F.TotalVolume):
    """
    TotalVolume for 2D axisymmetric (r, z) mesh.
    Computes: N = integral(u * 2*pi r dr dz) = real 3D inventory [particles]
    """

    azimuth_range: tuple = (0.0, 2 * np.pi)

    def compute(self, u, dx, entity_maps=None):
        from scifem import assemble_scalar

        mesh = u.function_space.mesh
        x = ufl.SpatialCoordinate(mesh)
        r = x[0]

        total = assemble_scalar(
            fem.form(
                u * r * dx(self.volume.id),
                entity_maps=entity_maps,
            )
        )
        total *= self.azimuth_range[1] - self.azimuth_range[0]

        self.value = total
        self.data.append(self.value)


# ---------------------------------------------------------------------------
# Run metadata and irradiation handling
# ---------------------------------------------------------------------------
RUN_URLS = {
    1: "https://raw.githubusercontent.com/LIBRA-project/BABY-1L-run-1/refs/tags/v0.6/data/processed_data.json",
    2: "https://raw.githubusercontent.com/LIBRA-project/BABY-1L-run-2/refs/tags/v0.5/data/processed_data.json",
    3: "https://raw.githubusercontent.com/LIBRA-project/BABY-1L-run-3/refs/tags/v0.2/data/processed_data.json",
    4: "https://raw.githubusercontent.com/LIBRA-project/BABY-1L-run-4/refs/tags/v0.1/data/processed_data.json",
}

# H2 switch time for the He_then_H2 runs [seconds]: sweep gas changes at day 19.
T_SWITCH_H2 = 19 * 86400


def get_total_measurement_time(run_id: int) -> float:
    """
    Return the time of the last tritium release measurement for this run [seconds].
    This is typically much longer than the irradiation time (post-irradiation
    sampling continues for days/weeks).
    """
    from libra_toolbox.tritium.model import ureg

    url = RUN_URLS[run_id]
    data = requests.get(url).json()
    cr = data["cumulative_tritium_release"]

    # Sampling times from both IV and OV; take the maximum to cover full measurement window
    iv_times = cr["IV"]["sampling_times"]
    ov_times = cr["OV"]["sampling_times"]

    # Get unit (typically "day")
    unit = iv_times.get("unit", "day")
    iv_max = max(iv_times["value"]) * ureg(unit)
    ov_max = max(ov_times["value"]) * ureg(unit)

    t_max = max(iv_max, ov_max).to(ureg("s")).magnitude
    return t_max


def get_irradiation_segments(run_id: int):
    """
    Return list of (t_start, t_end) tuples in SECONDS for each
    irradiation segment of this run. Handles multi-segment irradiations
    (e.g. run 4 has two irradiation campaigns).
    """
    from libra_toolbox.tritium.model import ureg

    url = RUN_URLS[run_id]
    data = requests.get(url).json()
    segments = []
    for irr in data["irradiations"]:
        start = (
            (irr["start_time"]["value"] * ureg(irr["start_time"]["unit"]))
            .to(ureg("s"))
            .magnitude
        )
        stop = (
            (irr["stop_time"]["value"] * ureg(irr["stop_time"]["unit"]))
            .to(ureg("s"))
            .magnitude
        )
        segments.append((start, stop))
    return segments


def make_tritium_source(run_id):
    """
    Build a time-dependent tritium source that is active only during the
    actual irradiation segments of this run.

    The source strength is normalised by the TOTAL irradiation duration so that
    the time-integral of the source over all segments equals
    TRITIUM_PRODUCTION[run_id], regardless of how many segments there are.

    Note: ParticleSource passes a NUMERICAL time `t` to the value function,
    so a plain Python `if` over the segments is valid here.
    """
    segments = get_irradiation_segments(run_id)

    total_irr_duration = sum(end - start for start, end in segments)
    source_strength = TRITIUM_PRODUCTION[run_id] / (total_irr_duration * V_CLLIF)

    def tritium_source(t):
        # Strict left inequality: with implicit Euler the step ending exactly
        # at t_start represents the interval (t_prev, t_start], which is
        # entirely BEFORE the irradiation. Including it would over-count by
        # source_strength * dt_before * V_CLLIF particles per segment start.
        for t_start, t_end in segments:
            if t_start < t <= t_end:
                return source_strength
        return 0.0

    return tritium_source, source_strength, segments


# Total tritium production per run [particles] - Table 2 from BABY-1L paper, last column
TRITIUM_PRODUCTION = {
    1: 9.45e9,
    2: 5.36e10,
    3: 1.66e10,
    4: 1.81e10,  # tuned per-run (paper Table 2 = 1.81e10 too low to match exp 55 Bq release)
}

# CLLiF volume [m^3] - nominal 1 L
V_CLLIF = 1.0e-3

# ---------------------------------------------------------------------------
# Material properties
# ---------------------------------------------------------------------------

htm_D_flibe = htm.diffusivities.filter(material="flibe").filter(author="calderoni")
htm_S_flibe = htm.solubilities.filter(material="flibe").filter(author="calderoni")

flibe_D_0 = htm_D_flibe[0].pre_exp.magnitude
flibe_D_0 *= 0.35
flibe_E_D = htm_D_flibe[0].act_energy.magnitude
flibe_S_0 = htm_S_flibe[0].pre_exp.magnitude * 1e15
flibe_E_S = htm_S_flibe[0].act_energy.magnitude

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
# Penalty term + solver tolerances
# -------------------------------------------------------------------

temperature_K = 650 + 273.15
# h = 0.001

D_flibe = flibe_D_0 * np.exp(-flibe_E_D / (8.617e-5 * temperature_K))
K_flibe = flibe_S_0 * np.exp(-flibe_E_S / (8.617e-5 * temperature_K))
D_inconel = inconel_D_0 * np.exp(-inconel_E_D / (8.617e-5 * temperature_K))
K_inconel = inconel_S_0 * np.exp(-inconel_E_S / (8.617e-5 * temperature_K))

# print(D_flibe, K_flibe)
# print(D_inconel, K_inconel)
# print("+++++++")
# flibe_D_0 = flibe_D_0 * 0.2
# inconel_D_0 = inconel_D_0 * 2

# D_flibe = flibe_D_0 * np.exp(-flibe_E_D / (8.617e-5 * temperature_K))
# K_flibe = flibe_S_0 * np.exp(-flibe_E_S / (8.617e-5 * temperature_K))
# D_inconel = inconel_D_0 * np.exp(-inconel_E_D / (8.617e-5 * temperature_K))
# K_inconel = inconel_S_0 * np.exp(-inconel_E_S / (8.617e-5 * temperature_K))
# Kr = inconel_Kr_0 * np.exp(-inconel_E_Kr / (8.617e-5 * temperature_K))
# print(f"D_flibe   = {D_flibe:.3e}  m^2/s")
# print(f"K_flibe   = {K_flibe:.3e}  (Henry units)")
# print(f"D_inconel = {D_inconel:.3e}  m^2/s")
# print(f"K_inconel = {K_inconel:.3e}  (Sievert units)")
# print(f"Kr        = {Kr:.3e}  m^4/s/atom")
# exit(0)

# penalty = 1e32  # this is used for the run 1 & 2 with the factor of 1e12 in the salt solubility.
# penalty = 1e35  # this is used for the run 4 with the factor of 1e12 in the salt solubility.
penalty = 1e34
# penalty = 1e10
atol = 1e-6
rtol = 1e-6


# ---------------------------------------------------------------------------
# H2 sweep gas concentration (used by both BC and recomb-eq flux)
# ---------------------------------------------------------------------------


def compute_h2_conc():
    """Compute H2 number density in the sweep gas [m^-3]."""
    h2_P_gauge = 10  # psi (gauge)
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

    sweep_gas : "He", "H2", or "He_then_H2"
        "He"          : pure helium for the whole run (Kr*c^2 recombination).
        "H2"          : constant H2 in the sweep gas (Kr*c^2 + Kr*c_H2*c).
        "He_then_H2"  : pure He until T_SWITCH_H2, then H2 turns on
                        (handled with a UFL conditional in the BC, and with the
                         two-version "Method A" stitching in the exports).
    """

    measurement_time = get_total_measurement_time(run_id)

    # --- Time-dependent source over the real irradiation segments ---
    tritium_source, source_strength, irr_segments = make_tritium_source(run_id)

    # Force the time-stepper to land exactly on every irradiation boundary
    # (and on the H2 switch time for the He_then_H2 runs) so the solver does not
    # step across a discontinuity in a single large step.
    source_milestones = []
    for t_start, t_end in irr_segments:
        source_milestones.append(t_start)
        source_milestones.append(t_end)

    milestones = sorted(set(source_milestones))
    if sweep_gas in ("He_then_H2", "H2"):
        milestones.append(T_SWITCH_H2)
        milestones = sorted(set(milestones))

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
    # model.method_interface = "nitsche"
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
    # Recombination boundary conditions on the Inconel-gas surfaces.
    #
    # `h2_late` is the H2 number density that is "on" during the H2 phase.
    # It is reused below to build the post-switch export version (Method A).
    # -----------------------------------------------------------------------

    if sweep_gas == "H2":
        # Run 4 physical schedule: 1000 ppm H2 from t=0, switched to 3.5%
        # (35000 ppm = 35 x base) at T_SWITCH_H2. compute_h2_conc() returns the
        # 1000 ppm number density; multiplying by 35 gives the 3.5% value at
        # unchanged sweep-gas pressure.
        # TEST: lower h2_early to ~50 ppm (full/20) to match exp pre-day-19 OV
        # slope (~0.15 Bq/day). h2_late kept at 3.5% (35× nominal full).
        h2_early = compute_h2_conc() / 30  # was: compute_h2_conc()
        h2_late = compute_h2_conc() * 35  # was: h2_early * 35 — keep 3.5% target

        def recombination_flux(c, T, t):
            Kr = inconel_Kr_0 * ufl.exp(-inconel_E_Kr / (F.k_B * T))
            h2 = ufl.conditional(ufl.ge(t, T_SWITCH_H2), h2_late, h2_early)
            return -Kr * c**2 - Kr * h2 * c

    elif sweep_gas == "He":
        h2_late = 0.0

        def recombination_flux(c, T):
            Kr = inconel_Kr_0 * ufl.exp(-inconel_E_Kr / (F.k_B * T))
            return -Kr * c**2

    elif sweep_gas == "He_then_H2":
        h2_late = compute_h2_conc()

        def recombination_flux(c, T, t):
            Kr = inconel_Kr_0 * ufl.exp(-inconel_E_Kr / (F.k_B * T))
            # UFL symbolic conditional: H2 turns on at T_SWITCH_H2.
            # `t` here is the symbolic simulation time supplied by FESTIM.
            h2 = ufl.conditional(ufl.ge(t, T_SWITCH_H2), h2_late, 0.0)
            return -Kr * c**2 - Kr * h2 * c

    else:
        raise ValueError(
            f"Unknown sweep_gas '{sweep_gas}'. Use 'He', 'H2', or 'He_then_H2'."
        )

    if sweep_gas == "He_then_H2":

        def h2_conc_export(t):
            return h2_late if t >= T_SWITCH_H2 else 0.0

    elif sweep_gas == "H2":

        def h2_conc_export(t):
            return h2_late if t >= T_SWITCH_H2 else h2_early

    else:
        h2_conc_export = h2_late

    subfolder = f"{results_folder}/run_{run_id}/sweep_{sweep_gas}"

    # -----------------------------------------------------------------------
    # First-order release coefficients [m/s]
    # J_release = k * c
    # -----------------------------------------------------------------------

    k_release = {
        "liquid_surface": 1e5,
        # "gap_sidewall": 1e15,
        # "top_cap": 1e15,
    }

    outer_inconel_surfaces = [
        inconel_outer_bottom,
        inconel_outer_side,
        inconel_outer_top,
    ]

    solid_recombination_surface = [
        gap_sidewall,
        top_cap,
        *outer_inconel_surfaces,
    ]

    recomb_bcs = []
    for surf in solid_recombination_surface:
        bc = F.ParticleFluxBC(
            value=recombination_flux,
            subdomain=surf,
            species_dependent_value={"c": T},
            species=T,
        )
        bc._volume_subdomain = vol_inconel
        recomb_bcs.append(bc)

    # inner_vessel_surfaces = [liquid_surface, gap_sidewall, top_cap]

    # model.boundary_conditions = [
    #     *[
    #         F.FixedConcentrationBC(subdomain=surf, species=T, value=0.0)
    #         for surf in inner_vessel_surfaces
    #     ],
    #     *recomb_bcs,
    # ]
    def mass_transfer_flux(k):
        def flux(c, T):
            return -k * c

        return flux

    mass_transfer_bcs = []

    for surf, k_val in [
        (liquid_surface, k_release["liquid_surface"]),
        # (gap_sidewall, k_release["gap_sidewall"]),
        # (top_cap, k_release["top_cap"]),
    ]:
        bc = F.ParticleFluxBC(
            value=mass_transfer_flux(k_val),
            subdomain=surf,
            species_dependent_value={"c": T},
            species=T,
        )
        mass_transfer_bcs.append(bc)

    model.boundary_conditions = [
        *mass_transfer_bcs,
        *recomb_bcs,
    ]

    model.temperature = temperature_K

    dt = F.Stepsize(
        initial_value=10,
        growth_factor=1.1,
        cutback_factor=0.9,
        target_nb_iterations=4,
        milestones=milestones,
    )

    model.settings = F.Settings(
        transient=True,
        atol=atol,
        rtol=rtol,
        final_time=measurement_time,
        stepsize=dt,
    )

    # -----------------------------------------------------------------------
    # Exports
    # -----------------------------------------------------------------------

    # Helper to build a recomb-eq flux export for a given Inconel surface.
    def make_recomb_eq_export(surface, name, filename, h2_conc):
        return CylindricalSurfaceFluxFromEquation(
            field=T,
            surface=surface,
            filename=filename,
            volume_subdomain=vol_inconel,
            inconel_Kr_0=inconel_Kr_0,
            inconel_E_Kr=inconel_E_Kr,
            temperature=temperature_K,
            h2_conc=h2_conc,
            name=name,
        )

    # All Inconel-gas surfaces (inner walls feeding IV + outer walls feeding OV)
    # for which we report the physical (recombination) release rate.
    inconel_surfs = [
        (inconel_outer_bottom, "inconel_outer_bottom"),
        (inconel_outer_side, "inconel_outer_side"),
        (inconel_outer_top, "inconel_outer_top"),
        (gap_sidewall, "gap_sidewall"),
        (top_cap, "top_cap"),
    ]

    # -----------------------------------------------------------------------
    # Recombination-equation exports.
    #
    # He_then_H2 runs now use a time-dependent h2_conc function and write one
    # CSV per surface. The legacy two-version Method A is kept below for
    # reference but is not used.
    # -----------------------------------------------------------------------
    recomb_eq_exports = []
    legacy_flux_filenames = {
        "gap_sidewall": "flux_gap_sidewall.csv",
        "top_cap": "flux_inconel_top_cap.csv",
        "inconel_outer_bottom": "flux_inconel_outer_bottom_recomb_eq.csv",
        "inconel_outer_side": "flux_inconel_outer_side_recomb_eq.csv",
        "inconel_outer_top": "flux_inconel_outer_top_recomb_eq.csv",
    }
    for surf, sname in inconel_surfs:
        recomb_eq_exports.append(
            make_recomb_eq_export(
                surf,
                sname,
                f"{subfolder}/{legacy_flux_filenames[sname]}",
                h2_conc=h2_conc_export,
            )
        )

    # -----------------------------------------------------------------------
    # Legacy Method A (two-version export) for time-varying H2.
    #
    #   - He_then_H2 runs: export TWO files per surface,
    #         flux_<surface>_h0.csv  (computed with h2 = 0,      valid t < switch)
    #         flux_<surface>_h2.csv  (computed with h2 = h2_late, valid t >= switch)
    #     Post-processing stitches them at T_SWITCH_H2.
    #
    #   - He / H2 runs (constant sweep gas): export a SINGLE file per surface,
    #         flux_<surface>_recomb_eq.csv
    #     with the constant h2 value used by the BC (0.0 for He, h2_late for H2).
    # -----------------------------------------------------------------------
    # recomb_eq_exports = []
    # if sweep_gas == "He_then_H2":
    #     for surf, sname in inconel_surfs:
    #         recomb_eq_exports.append(
    #             make_recomb_eq_export(
    #                 surf,
    #                 f"{sname} (h2=0)",
    #                 f"{subfolder}/flux_{sname}_h0.csv",
    #                 h2_conc=0.0,
    #             )
    #         )
    #         recomb_eq_exports.append(
    #             make_recomb_eq_export(
    #                 surf,
    #                 f"{sname} (h2=on)",
    #                 f"{subfolder}/flux_{sname}_h2.csv",
    #                 h2_conc=h2_late,
    #             )
    #         )
    # else:
    #     # Constant sweep gas: a single fixed h2 value matches the BC exactly.
    #     for surf, sname in inconel_surfs:
    #         recomb_eq_exports.append(
    #             make_recomb_eq_export(
    #                 surf,
    #                 sname,
    #                 f"{subfolder}/flux_{sname}_recomb_eq.csv",
    #                 h2_conc=h2_late,
    #             )
    #         )

    model.exports = [
        # Concentration fields
        F.VTXSpeciesExport(
            filename=f"{subfolder}/T_cllif.bp", field=T, subdomain=vol_cllif
        ),
        F.VTXSpeciesExport(
            filename=f"{subfolder}/T_inconel.bp", field=T, subdomain=vol_inconel
        ),
        # ---- Gradient-based fluxes (-D grad c . n) on every surface ----
        # CylindricalSurfaceFlux(
        #     field=T,
        #     surface=liquid_surface,
        #     filename=f"{subfolder}/flux_liquid_surface.csv",
        #     name="liquid surface",
        # ),
        # CylindricalSurfaceFlux(
        #     field=T,
        #     surface=top_cap,
        #     filename=f"{subfolder}/flux_inconel_top_cap.csv",
        #     name="Inconel top cap",
        # ),
        # CylindricalSurfaceFlux(
        #     field=T,
        #     surface=gap_sidewall,
        #     filename=f"{subfolder}/flux_gap_sidewall.csv",
        #     name="Inconel gap sidewall",
        # ),
        # ---- Liquid free surface release (mass-transfer law J = k c, IV-bound) ----
        CylindricalSurfaceFluxMassTransfer(
            field=T,
            surface=liquid_surface,
            filename=f"{subfolder}/flux_liquid_surface.csv",
            k=k_release["liquid_surface"],
            name="liquid surface",
        ),
        # CylindricalSurfaceFluxMassTransfer(
        #     field=T,
        #     surface=gap_sidewall,
        #     filename=f"{subfolder}/flux_gap_sidewall.csv",
        #     k=k_release["gap_sidewall"],
        #     name="gap sidewall",
        # ),
        # CylindricalSurfaceFluxMassTransfer(
        #     field=T,
        #     surface=top_cap,
        #     filename=f"{subfolder}/flux_inconel_top_cap.csv",
        #     k=k_release["top_cap"],
        #     name="top cap",
        # ),
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
        # ---- Recomb-eq (physical release) fluxes on all Inconel surfaces ----
        # (single version for He/H2, two-version h0/h2 for He_then_H2)
        *recomb_eq_exports,
        # ---- Tritium inventory per region ----
        CylindricalTotalVolume(
            field=T, volume=vol_cllif, filename=f"{subfolder}/inventory_cllif.csv"
        ),
        CylindricalTotalVolume(
            field=T, volume=vol_inconel, filename=f"{subfolder}/inventory_inconel.csv"
        ),
    ]

    # return model
    return model, T, vol_cllif, vol_inconel


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # Sweep-gas schedule per run.
    #   run 1, 2 : pure He for the whole run.
    #   run 3    : He, then 1000 ppm H2 switched on at day 19.
    #   run 4    : 1000 ppm H2 from the start, then 3.5% H2 (35x) at day 19.
    RUN_SWEEP = {
        1: "He",
        2: "He",
        3: "He_then_H2",
        4: "H2",
    }

    for run_id in [1, 2, 3, 4]:
        # for run_id in [4]:  # iter 15: test h2_early /100 effect on run 4 OV shape
        sweep = RUN_SWEEP[run_id]
        print(f"\n=== Run {run_id} / {sweep} sweep ===")
        # set_log_level(LogLevel.INFO)
        # model = build_model(sweep_gas="He", run_id=run_id)
        # model, T, vol_cllif, vol_inconel = build_model(sweep_gas="He", run_id=run_id)
        model, T, vol_cllif, vol_inconel = build_model(sweep_gas=sweep, run_id=run_id)
        model.initialise()
        for export in model.exports:
            if hasattr(export, "bind_time_constant"):
                export.bind_time_constant(model.t)
        model.run()

        from dolfinx import geometry
        import numpy as np

        u_flibe = T.subdomain_to_post_processing_solution[vol_cllif]
        u_inconel = T.subdomain_to_post_processing_solution[vol_inconel]
        r_iface = 0.07
        z_test = 0.0558

        mesh_flibe = vol_cllif.submesh
        mesh_inconel = vol_inconel.submesh

        bb_tree_flibe = geometry.bb_tree(mesh_flibe, mesh_flibe.topology.dim)
        bb_tree_inconel = geometry.bb_tree(mesh_inconel, mesh_inconel.topology.dim)

        for export in model.exports:
            if hasattr(export, "data") and len(export.data) > 0:
                print(f"{export.title}: {export.data[-1]:.3e}")

        def eval_at(u, bb_tree, mesh, r, z):
            pt = np.array([[r, z, 0.0]])
            candidates = geometry.compute_collisions_points(bb_tree, pt)
            cells = geometry.compute_colliding_cells(mesh, candidates, pt)
            return u.eval(pt, np.array([cells.links(0)[0]]))[0]

        c_l = eval_at(u_flibe, bb_tree_flibe, mesh_flibe, r_iface, z_test)
        c_r = eval_at(u_inconel, bb_tree_inconel, mesh_inconel, r_iface, z_test)

        print(f"c_flibe          = {c_l}")
        print(f"c_inconel        = {c_r}")
        print(f"c_henry/K_H      = {c_l / K_flibe}")
        print(f"(c_sievert/K_S)^2 = {(c_r / K_inconel) ** 2}")
        print(
            f"henry ratio            = {(c_l / K_flibe) / (c_r / K_inconel) ** 2:.4e}"
        )

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
