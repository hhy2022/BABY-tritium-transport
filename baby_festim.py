"""BABY-1L tritium transport: FESTIM 2D axisymmetric hot-zone baseline.

Current model (2026-07-14). This file solves the hot-zone physics only:
  salt: well-mixed (Calderoni D x10), Henry solubility (x FLIBE_S_SCALE);
        production = nominal neutronics-anchored atoms/run;
  IV:   liquid free surface J = k_d*c (literature k_top 8.9e-8) + H2 sparge
        stripping (lambda) + inner-wall exchange (gap sidewall, top cap);
  OV:   Henry|Sievert interface -> diffusion through Inconel (literature) ->
        surface release J = 2*Kr*c^2 + Kr*c_H*c (Perujo oxidised, atoms flux).
Delayed collection on the cold hardware (deposition on tubing and the OV
enclosure, later released by H2) and the per-run production calibration are
applied in a separate 0-D post-processing layer (deposition fraction f,
OV split beta, first-order H2 scavenging per collection path; the fitting
and post-processing scripts will be released together with the paper).
Retired mechanisms (in-salt bound pool, locked floors, Langmuir c_sat,
external injections) default to 0 but remain reachable via BABY_* envs;
their history is kept in the comment blocks below.
"""

import numpy as np
import festim as F
import h_transport_materials as htm
import requests
import ufl
from dolfinx import fem
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI
import os
import gc
import json


def _env_float(name, default):
    """Fit-parameter override via environment variable (used by the fitting
    harness to sweep salt-side parameters without editing this file).
    Defaults reproduce the in-file values."""
    v = os.environ.get(name)
    return float(v) if v not in (None, "") else default


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
# He case:  J = 2 * Kr * c^2
# H2 case:  J = 2 * Kr * c^2 + Kr * c_H2(t) * c
# (factor 2 on the homonuclear term: each T2 removes TWO T atoms; the
# heteronuclear HT term removes ONE T per molecule, so coefficient 1 --
# FESTIM 2.x SurfaceReactionBC convention, atoms flux = 2*K when A=B.)
#
# The export reads the current simulation time from the model time constant
# bound after model.initialise(). This allows c_H2 to be a time-dependent
# Python function while keeping the exported quantity in one CSV file.
# ---------------------------------------------------------------------------


class CylindricalSurfaceFluxFromEquation(F.SurfaceFlux):
    """
    Cylindrical surface flux computed from the recombination equation.

    For sweep_gas == "He":
        J = integral(2 * Kr * c^2 * r  dS) * 2*pi

    For sweep_gas == "H2":
        J = integral( (2 * Kr * c^2 + Kr * c_H2(t) * c) * r  dS) * 2*pi
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
        kex_0=None,  # exchange coeff [m^4/s/atom]; None -> use Kr (Inconel case)
        name=None,
    ):
        super().__init__(field=field, surface=surface, filename=filename)
        self.volume_subdomain = volume_subdomain
        self.inconel_Kr_0 = inconel_Kr_0
        self.inconel_E_Kr = inconel_E_Kr
        self.temperature = temperature
        self.h2_conc = h2_conc
        self.kex_0 = kex_0
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

        # Exchange coeff: separate kex_0 if given (liquid), else Kr (Inconel).
        kex = self.kex_0 if self.kex_0 is not None else Kr

        # Physical release: T+T -> T2 (recomb)  plus  T+H -> HT (exchange, if H2)
        # Both terms are non-negative since c >= 0 and h2_conc >= 0
        integrand = 2 * Kr * u**2 + kex * h2_conc * u

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

    def __init__(self, field, surface, filename, k, name=None, c_sat=0.0):
        super().__init__(field=field, surface=surface, filename=filename)
        # k may be a constant [m/s] or a time-dependent function k(t) (used to
        # carry the H2 isotopic-exchange enhancement k_eff(t) = k_d + k_ex*c_H).
        self.k = k
        self._name = name
        # Langmuir saturation [1/m^3]; 0 = off. Must mirror the BC exactly.
        self.c_sat = c_sat
        self.time_constant = None

    @property
    def title(self):
        label = self._name if self._name else f"surface {self.surface.id}"
        return f"{self.field.name} mass-transfer release {label}"

    def bind_time_constant(self, time_constant):
        self.time_constant = time_constant

    def _current_k(self):
        if callable(self.k):
            if self.time_constant is None:
                raise RuntimeError(
                    "A time-dependent k requires bind_time_constant()."
                )
            value = self.time_constant.value
            t = float(value[0] if np.ndim(value) > 0 else value)
            return float(self.k(t))
        return float(self.k)

    def compute(self, u, ds, entity_maps=None):
        from scifem import assemble_scalar

        if isinstance(u, ufl.indexed.Indexed):
            mesh = self.field.sub_function_space.mesh
        else:
            mesh = u.function_space.mesh

        x = ufl.SpatialCoordinate(mesh)
        r = x[0]

        k = self._current_k()

        integrand = k * u
        if self.c_sat > 0:
            integrand = integrand / (1.0 + u / self.c_sat)

        flux = assemble_scalar(
            fem.form(
                integrand * r * ds(self.surface.id),
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
# Bound-tritium (TF) -> mobile conversion, rate prop. to p_H2(t).
#
# One-way volumetric reaction in the CLLiF:  T_bound --kappa(t)--> T
# kappa(t) follows the run's H2 schedule (0 under pure He). Implemented as a
# Reaction subclass whose rate is a UFL conditional on the model time, bound by
# TimeAwareProblem before the formulation is assembled (same pattern as the
# trap version, see baby_festim_trap.py).
# ---------------------------------------------------------------------------


class H2ConversionReaction(F.Reaction):
    """T_bound -> T at rate kappa(t)*c_bound, kappa stepping at t_switch."""

    def __init__(self, *args, kappa_early=0.0, kappa_late=0.0, t_switch=0.0,
                 c_floor_early=0.0, c_floor_late=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.kappa_early = kappa_early
        self.kappa_late = kappa_late
        self.t_switch = t_switch
        # locked residual [atoms/m^3]; release only above it. Steps at t_switch
        # like kappa: more H2 (late) frees deeper into the pool (lower floor).
        self.c_floor_early = c_floor_early
        self.c_floor_late = c_floor_late
        self.time_constant = None  # bound to model.t before formulation build

    def reaction_term(
        self, temperature, reactant_concentrations=None, product_concentrations=None
    ):
        if self.time_constant is None:
            # A silent kappa_early fallback would freeze the H2 switch forever;
            # fail loudly instead (TimeAwareProblem binds model.t before the
            # formulation is assembled).
            raise RuntimeError(
                "H2ConversionReaction.time_constant not bound -- use "
                "TimeAwareProblem (it binds model.t before assembling)."
            )
        # Strict > : with implicit Euler the step ending exactly at t_switch
        # covers (t_prev, t_switch], which is entirely BEFORE the switch.
        kappa = ufl.conditional(
            ufl.gt(self.time_constant, self.t_switch),
            self.kappa_late,
            self.kappa_early,
        )

        reactants = self.reactant
        if reactant_concentrations is not None:
            rc = [
                reactant_concentrations[i]
                if reactant_concentrations[i] is not None
                else r.concentration
                for i, r in enumerate(reactants)
            ]
        else:
            rc = [r.concentration for r in reactants]

        c_bound = rc[0]
        for x in rc[1:]:
            c_bound = c_bound * x

        # Release only the part of the pool ABOVE the locked floor; the floor
        # stays as permanent residual (-> freed excess plateaus once exhausted).
        if self.c_floor_early or self.c_floor_late:
            c_floor = ufl.conditional(
                ufl.gt(self.time_constant, self.t_switch),
                self.c_floor_late,
                self.c_floor_early,
            )
            return kappa * ufl.max_value(c_bound - c_floor, 0.0)
        return kappa * c_bound


class TimeAwareProblem(F.HydrogenTransportProblemDiscontinuous):
    """Binds the model time constant into any time-dependent reaction before
    the variational formulation is assembled (model.t exists by then)."""

    def create_subdomain_formulation(self, subdomain):
        for rxn in self.reactions:
            if hasattr(rxn, "time_constant"):
                rxn.time_constant = self.t
        # Upstream fetches source.species' test function on EVERY subdomain
        # before checking source.volume; a species absent from this subdomain
        # (e.g. T_bound, CLLiF-only) raises KeyError. Temporarily hide sources
        # whose species does not live on this subdomain.
        all_sources = self.sources
        self.sources = [
            s for s in all_sources if subdomain in s.species.subdomains
        ]
        try:
            return super().create_subdomain_formulation(subdomain)
        finally:
            self.sources = all_sources


# ---------------------------------------------------------------------------
# Run metadata and irradiation handling
# ---------------------------------------------------------------------------
RUN_URLS = {
    1: "https://raw.githubusercontent.com/LIBRA-project/BABY-1L-run-1/refs/tags/v0.6/data/processed_data.json",
    2: "https://raw.githubusercontent.com/LIBRA-project/BABY-1L-run-2/refs/tags/v0.5/data/processed_data.json",
    3: "https://raw.githubusercontent.com/LIBRA-project/BABY-1L-run-3/refs/tags/v0.2/data/processed_data.json",
    4: "https://raw.githubusercontent.com/LIBRA-project/BABY-1L-run-4/refs/tags/v0.1/data/processed_data.json",
}

# Local cache of each run's processed_data.json (fitting/cache/run_<id>.json):
# parallel fitting processes would otherwise re-download from GitHub every run.
_DATA_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fitting", "cache"
)


def _get_run_data(run_id: int) -> dict:
    os.makedirs(_DATA_CACHE_DIR, exist_ok=True)
    path = os.path.join(_DATA_CACHE_DIR, f"run_{run_id}.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            os.remove(path)  # poisoned partial write -> re-download
    resp = requests.get(RUN_URLS[run_id], timeout=30)
    resp.raise_for_status()
    data = resp.json()
    # Atomic publish so parallel processes never see a partial file.
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)
    return data


# H2 switch time for the He_then_H2 runs [seconds]: sweep gas changes at day 19.
# Exact per-run H2 switch times, computed from each run's general.json:
# gas_switch_time minus the first irradiation (generator) start.
#   run 3: 4/4/2025 15:06 - 3/17/2025 10:03 = day 18.210
#   run 4: 5/18/2025 19:58 - 5/1/2025 11:07 = day 17.369
# The old hardcoded value (19 d) was ~0.8-1.6 d late.
T_SWITCH_H2_BY_RUN = {3: 18.210 * 86400, 4: 17.369 * 86400}
T_SWITCH_H2 = 19 * 86400  # legacy default for runs not listed above


def get_total_measurement_time(run_id: int) -> float:
    """
    Return the time of the last tritium release measurement for this run [seconds].
    This is typically much longer than the irradiation time (post-irradiation
    sampling continues for days/weeks).
    """
    from libra_toolbox.tritium.model import ureg

    data = _get_run_data(run_id)
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

    data = _get_run_data(run_id)
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


# Tritium production = TWO EXPLICIT SOURCES per run (model finalized 2026-06-16):
#   (1) BABY breeding (TRITIUM_PRODUCTION): born DURING the irradiation, released
#       by diffusion + free-surface k_d. Sized to the FIRST plateau. NO trapping
#       (F_BOUND = 0 everywhere -- the bound pool is retired; the late plateaus
#       are an explicit external source, not trapped breeding).
#   (2) External cross-irradiation (EXT_INJ_* below): a SEPARATE source injected
#       MOBILE LATE in time -- run3 @ day28, run4 @ day18 -- making the last
#       plateau. run3 and run4 are INDEPENDENT external events.
# Mass conservation: BABY + external = total release.
#   run3: BABY 1.04e10 (18.5 Bq, 1st plateau) + ext 6.2e9 (11 Bq @ day28) = 1.66e10
#   run4: BABY 2.10e10 (37.5 Bq, 0.1% plateau) + ext 6.0e9 (10.7 Bq @ day18) = 2.70e10
# KNOWN COST (accepted by user 2026-06-16): with F=0 and run3 external only @day28,
#   run3's day18-28 2nd plateau sits flat ~19 (H2 strips BABY residual) while data
#   rises to ~24 -- a ~5 Bq gap (IVrmse 2.16). Filling it would need F back or a
#   2nd injection @day18. run1/2 (He cover): pure BABY, no external.
# 2026-06-22 LOCKED-FLOOR MODEL: production [atoms] per run. run2 is
# LOWERED below its fluence value -- the run2 data is contaminated/inflated
# (heater-freeze episode), so its true breeding is less. All other physical
# params (F, c_sat, kappa, lambda, k_d) are UNIFIED across runs; only production
# differs (each run a different fluence). The 2nd/3rd plateaus are made by the
# bound pool (F_BOUND) being freed by H2 down to a LOCKED FLOOR (KAPPA_FLOOR_*),
# not by an external source -- so EXT_INJ_* is OFF.
# 2026-07-14 (current line-holdup model): the values below are the nominal
# neutronics-anchored productions used by the hot-zone baseline.
# run4 = 2.75e10 (line-base recalibration of the suspect run4 rate).
# The per-run production CALIBRATION (run1 x0.8259, run2-4 x0.8786 -- the
# exact no-permanent-trap reparameterization) is applied in the
# post-processing layer, not here: the salt physics is linear, so scaling
# there is exact and the baseline CSVs stay reusable across calibrations.
TRITIUM_PROD_INTRINSIC = {1: 1.31e10, 2: 6.4e10, 3: 1.44e10, 4: 2.75e10}
TRITIUM_PROD_EXTERNAL = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}  # external retired (floor model)
TRITIUM_PRODUCTION = {
    r: _env_float(f"BABY_PROD_{r}", TRITIUM_PROD_INTRINSIC[r] + TRITIUM_PROD_EXTERNAL[r])
    for r in (1, 2, 3, 4)
}

# LATE external-source injection [atoms] = source (2) above, born MOBILE over the
# window [EXT_INJ_DAY, EXT_INJ_DAY + EXT_INJ_WINDOW_DAY]. Per-run (independent
# cross-irradiation events). run1/2 have none.
EXT_INJ_PROD = {r: _env_float(f"BABY_EXT_PROD_{r}", _d)
                for r, _d in {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}.items()}  # OFF (floor model)
EXT_INJ_DAY = {r: _env_float(f"BABY_EXT_DAY_{r}", _d)
               for r, _d in {1: 0.0, 2: 0.0, 3: 28.0, 4: 18.0}.items()}
EXT_INJ_WINDOW_DAY = {r: _env_float(f"BABY_EXT_WINDOW_{r}", _d)
                      for r, _d in {1: 1.0, 2: 1.0, 3: 17.0, 4: 11.0}.items()}

# CLLiF volume [m^3] - nominal 1 L
V_CLLIF = 1.0e-3

# Cross-run carry-over: initial BOUND (trapped) tritium [Bq] left in the REUSED
# salt by EARLIER runs (run3 seeds run1+run2's accumulated trap; run4 seeds
# run3's leftover). Set as a uniform initial T_bound concentration. Default 0
# (= independent runs). Used to model the history-accumulation picture.
#   Floor model (2026-06-22): run1/2 (He) trap F_BOUND and accumulate -- nothing
#   is released under He, so run2's trapped pool carries to run3 (history). run3
#   seeds run1+run2's accumulated trap (~41 Bq), run4 seeds run3's leftover (~41).
# 2026-07-14 (line-holdup model): carry-over set to 0 -- cross-run history is
# carried by the cold-hardware pools of the post-processing layer.
BOUND_INIT_BQ = {r: _env_float(f"BABY_BOUND_INIT_{r}", _d)
                 for r, _d in {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}.items()}
_LAMBDA_T_BQ = 3.57e14 * 3.016 / 6.02214076e23  # Bq per tritium atom

# "Locked floor" of the trapped pool [Bq]: H2 frees T_bound only DOWN TO this
# level, then the release stops -> the freed excess plateaus, the floor stays in
# the salt as permanent residual. This is the self-consistent realization of the
# "history accumulates (residual) + H2 frees a limited amount together" picture:
# releasing the WHOLE accumulated pool overshoots 3x and never plateaus (tested),
# so only the part ABOVE the floor is releasable. Default 0 (= release to empty).
# Floor stepping at the H2 switch: floor_late (more H2) <= floor_early (frees
# deeper). For single-switch runs (3) only the late floor matters (He has no
# release). For run4 the 0.1% floor is higher than the 3.5% floor. Default:
# early floor = late floor (= single constant floor).
# Floor model (2026-06-22): run3 0.1% floor 41 (= run4 0.1% floor, UNIFIED gas);
# run4 3.5% floor 35 (deeper -- more H2 frees more). run1/2 (He) never release.
# 2026-07-14 (line-holdup model): floors set to 0 (the floor mechanism is
# retired along with the bound pool; inert while F_BOUND = 0).
KAPPA_FLOOR_BQ = {r: _env_float(f"BABY_KAPPA_FLOOR_{r}", _d)
                  for r, _d in {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}.items()}
# Early (pre-switch) floor; only meaningful with F_BOUND > 0.
KAPPA_FLOOR_EARLY_BQ = {
    r: _env_float(f"BABY_KAPPA_FLOOR_EARLY_{r}", _d)
    for r, _d in {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}.items()
}

# ---------------------------------------------------------------------------
# Material properties
# ---------------------------------------------------------------------------

htm_D_flibe = htm.diffusivities.filter(material="flibe").filter(author="calderoni")
htm_S_flibe = htm.solubilities.filter(material="flibe").filter(author="calderoni")

flibe_D_0 = htm_D_flibe[0].pre_exp.magnitude
# Salt diffusivity multiplier (FITTED, salt side):
#   x10   = WELL-MIXED salt (molten salt convects): L^2/D << release time, the
#           release is SURFACE-limited, controlled by LIQUID_KD alone -- the
#           LIBRA k_top picture (k_d ~ 8.9e-8 reproduces run 1).
#   x0.35 = diffusion-influenced regime (step-4 test).
FLIBE_D_MULT = _env_float("BABY_D_MULT", 10.0)
flibe_D_0 *= FLIBE_D_MULT
flibe_E_D = htm_D_flibe[0].act_energy.magnitude
# Salt-side solubility scaling (FITTED; salt side is the legitimate fitting
# ground). Goes with the Henry interface jump and penalty=1e34.
# Partition knob: c_inconel = K_S,inc*sqrt(c_salt/K_H,salt), so LOWER scale
# pushes more tritium into the metal -> larger OV release.
# 8e13 calibrated 2026-06-11 against run-3 OV final (4.2 Bq) with the
# UNMODIFIED literature Inconel Kr (INCONEL_KEX_MULT = 1).
# (Sievert-CLLiF test 2026-06-10, CLOSED: with a fair x320 rescale the linear
# interface ratio dumps the post-switch inventory through the wall in ~3 d --
# OV 9.2 vs exp 4.2 -- because the constant ratio lacks Henry's 1/sqrt(c_l)
# throttling that produces the observed 27-d gradual OV rise. Henry kept.)
FLIBE_S_SCALE = _env_float("BABY_S_SCALE", 8e13)
flibe_S_0 = htm_S_flibe[0].pre_exp.magnitude * FLIBE_S_SCALE
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


# penalty = 1e32  # this is used for the run 1 & 2 with the factor of 1e12 in the salt solubility.
# penalty = 1e35  # this is used for the run 4 with the factor of 1e12 in the salt solubility.
penalty = _env_float("BABY_PENALTY", 1e34)  # goes with Henry CLLiF + FLIBE_S_SCALE = 1e15
# (sievert + FLIBE_S_SCALE=320 needed penalty ~1e22 -- interface c/K magnitudes
# scale the penalty term; retune ONLY if a parameter change breaks convergence)
# penalty = 1e10
atol = 1e-6
rtol = 1e-6


# ---------------------------------------------------------------------------
# H2 sweep gas concentration (used by both BC and recomb-eq flux)
# ---------------------------------------------------------------------------


def compute_p_h2(h2_conc_ppm=1000):
    """H2 partial pressure in the sweep gas [Pa] for a given H2 fraction [ppm]."""
    h2_P_gauge = 10  # psi (gauge)
    mole_frac_h2 = h2_conc_ppm / 1e6
    P_atm = 14.7
    P_abs = h2_P_gauge + P_atm
    P_h2 = mole_frac_h2 * P_abs
    P_h2 *= 6894.76  # psi -> Pa
    return P_h2


def compute_h2_conc(h2_conc_ppm=1000):
    """Dissolved atomic H concentration at the Inconel surface [m^-3].

    The isotopic-exchange recombination term is Kr * c_H * c, where Kr is the
    Inconel surface recombination coefficient and both c (dissolved T) and c_H
    must be DISSOLVED ATOMIC concentrations in the metal [atoms/m^3] for the
    units to match. The H partner in Sievert equilibrium with the sweep-gas
    H2 partial pressure is therefore

        c_H = K_s,inconel * sqrt(p_H2)

    NOT the gas-phase H2 molecular number density. Crucially this makes the
    exchange term scale as sqrt(p_H2) (so 1000 ppm -> 3.5% is x sqrt(35) ~ 5.9),
    not linearly in p_H2.
    """
    P_h2 = compute_p_h2(h2_conc_ppm)
    return K_inconel * np.sqrt(P_h2)


# Legacy (physically inconsistent) version: gas-phase H2 molecular number
# density [m^-3]. Kept for reference / quick A-B comparison.
# def compute_h2_conc(h2_conc_ppm=1000):
#     P_h2 = compute_p_h2(h2_conc_ppm)
#     gas_constant = 8.314
#     T_room = 298
#     h2_conc_mol = P_h2 / (gas_constant * T_room)  # mol/m^3
#     return h2_conc_mol * 6.022e23  # m^-3


# ---------------------------------------------------------------------------
# Liquid (CLLiF) free-surface release: LINEAR desorption + isotopic exchange.
#
#       J = (k_d + k_ex * c_H) * c
#
# Linear in c -> the release timescale is INDEPENDENT of inventory c0, so a
# single k_d fits runs of different total production consistently (unlike c^2,
# whose c0-dependence made run 3's higher-inventory He phase release too fast
# and overshoot). Run in the MIXED-control regime: D*0.35 (diffusion-influenced,
# Biot ~2.6) gives the fast early rise, while k_d is finite (not the old k=1e5
# diffusion sink) so the surface partially limits and the H2 exchange term
# k_ex*c_H actually boosts release -> a visible day-19 switch. c_H = K_H*p_H2
# (Henry, linear in p). k_ex is tuned for the switch strength.
# ---------------------------------------------------------------------------

# Fitted liquid free-surface coefficient (FINAL 2026-06-11):
#   LIQUID_KD = 8.5e-8 m/s ~ LIBRA k_top (8.9e-8) -- literature-adjacent.
#   Well-mixed melt: tau = V/(k*A) ~ 8.9 d reproduces run 1 (tau_exp 7.6 d)
#   and run 3's pre-switch FAST-but-PARTIAL first plateau (mobile pool only;
#   the rest is born bound, see F_BOUND below).
#   Run 2 (tau_exp ~ 14 d, same He cover) is the accepted compromise: no
#   unified linear model can give run 1 and run 2 different normalized shapes;
#   kd=8.5e-8 balances them (run1 RMSE 0.7, run2 RMSE 9.5, terminal both ok).
# 2026-06-16: raised 8.5e-8 -> 1.1e-7. Premise: run2 is contaminated by an
#   external source (inflated, can't cleanly remove), so it must NOT pin k_d.
#   run2's slow profile was pulling k_d DOWN (MCMC posterior 7.55e-8). With run2
#   excluded, the CLEAN runs prefer faster k_d: joint optimum of run1/3/4 is
#   1.1e-7 (run1 0.67->0.50, run3 IV 0.98->0.81 & OV 0.31->0.24, run4 ~same).
#   Cost: run2 fits worse, accepted (contaminated). Departs ~24% from LIBRA
#   k_top 8.9e-8 but is what the clean run1 free-surface data wants.
# 2026-07-14 (line-holdup model): back to the literature value 8.9e-8 (k_top).
#   The 1.1e-7 was fitted when the salt-side bound pool carried the H2 response;
#   with the line-holdup layer the release shaping happens in the cold hardware
#   and the free surface keeps its literature coefficient (Inconel-style rule:
#   fit in the post-processing layer, not on physical surfaces).
# H2 surface exchange on the liquid is OFF (the H2 response is carried by the
# sparge sink + line-holdup scavenging, not by surface kinetics).
LIQUID_KD = _env_float("BABY_KD", 8.9e-8)  # desorption coeff [m/s]
LIQUID_KEX = _env_float("BABY_KEX", 0.0)  # H2 surface-exchange coeff [m^4/s/atom]
# Langmuir-type saturation of the free-surface desorption [1/m^3]; 0 = off:
#   J = k_eff * c / (1 + c/LIQUID_CSAT)
# DROPPED in the final picture (default 0). It was an alternative explanation
# for run 2 being slower than run 1 (high inventory -> saturated surface), but
# the documented heater malfunction / salt freeze (paper Sec 3.3) explains
# run 2 more simply, so saturation is not invoked. Kept only as an env knob.
# 2026-06-22 floor model: c_sat = 2.3e13 ON (weak Langmuir). It barely touches
# the low-concentration runs but gives the natural curved (saturating) release
# shape; without it (or with a much stronger c_sat) the high-inventory run2 goes
# unphysically linear. This is the original calibrated Langmuir scale.
# 2026-07-14 (line-holdup model): retired (0). Tested during model cleanup:
# removing c_sat slightly improved the fit; run2's curvature is handled by the
# contamination hypothesis + production calibration, not surface saturation.
LIQUID_CSAT = _env_float("BABY_CSAT", 0.0)

# (c2 surface-control test, CLOSED by run 2: t_c ~ 1/c0 makes the high-production
# run release ~6x too fast; data demands first-order kinetics.)
# LIQUID_KR = 2e-20  # [m^4/s/atom]

# --- Sparge extraction (runs 3/4: gas bubbled through the melt) ---
# TRUE volumetric stripping sink: T --lambda(t)--> T_sparged (immobile tally
# species). The extracted tritium leaves with the sparge gas to the IV bubbler,
# so inventory_sparged.csv IS the cumulative sparge contribution to IV.
# lambda depends on the sparge-gas chemistry:
#   pure He bubbles  : lambda = LAMBDA_SPARGE_HE (dissolved T is mostly TF,
#                      non-volatile -> He strips little; ~0)
#   H2 in the bubbles: lambda = LAMBDA_SPARGE_1000 * (ppm/1000)**LAMBDA_EXP
#                      (isotope exchange at the bubble surface strips T)
# Runs 1/2 have NO sparger (cover gas only) -> no sink (sweep_gas "He" branch).
# FINAL 2026-06-11: lambda_1000 = 2e-5 (tau ~ 0.6 d) from run 4's sub-day
# post-campaign flushes; SATURATED in ppm (LAMBDA_EXP = 0: stripping is
# bubble-mass-transfer limited once H2 chemistry is available). The saturated
# lambda + sqrt(p_H2) metal exchange together reproduce the observed
# post-switch OV/IV flip between run 3 (1000 ppm, IV-dominant, ratio 0.41)
# and run 4 (3.5%, OV-dominant, ratio 1.9 ~ 0.41*sqrt(35)).
LAMBDA_SPARGE_HE = _env_float("BABY_LAMBDA_HE", 0.0)  # [1/s]
LAMBDA_SPARGE_1000 = _env_float("BABY_LAMBDA_1000", 2e-5)  # [1/s] at 1000 ppm
LAMBDA_EXP = _env_float("BABY_LAMBDA_EXP", 0.0)  # ppm-scaling exponent


def lambda_sparge(h2_ppm: float) -> float:
    """Volumetric sparge-stripping rate [1/s] for a given H2 fraction [ppm]."""
    if h2_ppm <= 0:
        return LAMBDA_SPARGE_HE
    return LAMBDA_SPARGE_1000 * (h2_ppm / 1000.0) ** LAMBDA_EXP


# Legacy surface-bump shortcut (well-mixed equivalence), no longer used:
# K_SPARGE = {1: 0.0, 2: 0.0, 3: 3.4e-8, 4: 3.4e-8}

# --- Sparging-induced trapped-tritium pool ---
# PHYSICAL PICTURE (revised 2026-06-13). The only hardware difference between
# the campaigns is the gas path: runs 1-2 use COVER GAS (swept over the melt),
# runs 3-4 SPARGE (gas bubbled THROUGH the melt). The bound pool is the
# tritium that the rising bubbles carry out of the salt but deposit on the
# vessel internals -- crucible walls, cold tubing, headspace -- where it
# adsorbs and is NOT collected by the inert sweep. Helium cannot release it;
# H2 does, by isotope exchange H2 + T(ad) -> HT + H(ad) (paper Sec 3.4,
# which independently attributes the H2 effect to T adsorbed on exactly these
# surfaces). So:
#   runs 1-2 (cover gas): no bubbling -> no trapping -> F_BOUND = 0, the
#            tritium desorbs straight off the free surface (release = breeding).
#            run 2's slow profile is the documented heater-malfunction / salt
#            freeze (paper Sec 3.3), an operational artefact, NOT a mechanism.
#   runs 3-4 (sparging): bubbles trap a fraction on the walls/tubing. This is
#            why ONLY 3-4 show a "storage" term -- it tracks the sparger, not
#            the salt's age. He phase: trapped fraction held; H2 phase:
#            released by exchange (the day-18 IV step + OV rise).
#   runs 3 AND 4: F = 0.35, the SAME trapped fraction, in the SAME fast pool
#          (kappa_conv). This is required by the experiment: runs 3-4 use the
#          identical sparger hardware, so the bubble-deposited fraction must be
#          the same parameter (same setup -> same value). Verified 2026-06-15:
#          unifying run 4 to F=0.35 (from an earlier free-fit 0.43) IMPROVES its
#          IV RMSE 2.15 -> 1.61 -- the 0.43 was a degenerate (F,P) artefact, not
#          physics. run 4's larger TOTAL release comes from its higher PRODUCTION
#          (cross-irradiation, P4 below), NOT a larger trapped fraction.
#          kappa scales with p_H2 (sqrt), so run 4's 0.1% start gives a slow
#          creep and its 3.5% switch a faster tail from the SAME kappa as run 3.
# The pool stays mass-conserving: total release = production (NO tritium
# created); it only DELAYS the trapped fraction from the He phase to the H2
# phase. (Langmuir surface saturation, an earlier candidate for run 2, is
# dropped -- run 2 is the heater-freeze artefact, not saturation.)
# 2026-06-16: bound pool RETIRED (F=0 everywhere). The 2nd/3rd plateaus are now
# made by the explicit LATE external source (EXT_INJ_*), not by trapped breeding.
# (The F-trapping defaults 0.35/0.5 are still reachable via BABY_FBOUND_* env for
# the alternative trapping model.)
# 2026-06-22 LOCKED-FLOOR MODEL: F_BOUND = 0.30 for ALL runs (unified). The
# trapped fraction is a property of the SALT (tritium retained in the melt), not
# the sparger -- so run1/2 (He) trap it too and accumulate it (history), and H2
# in run3/4 frees it down to a locked floor (KAPPA_FLOOR_*). This supersedes the
# earlier "F=0 for He / external-source" picture above.
# 2026-07-14 (current line-holdup model): F_BOUND = 0. The in-salt bound pool
# is retired; the delayed-release physics is attributed to the cold hardware
# (collection tubing / OV enclosure) and applied in the 0-D post-processing
# layer described in the module docstring. The FESTIM solve
# below is the hot-zone baseline. The earlier floor-model values remain
# reachable via BABY_FBOUND_* / BABY_KAPPA_FLOOR_* / BABY_BOUND_INIT_* envs.
F_BOUND = {
    1: _env_float("BABY_FBOUND_1", 0.0),
    2: _env_float("BABY_FBOUND_2", 0.0),
    3: _env_float("BABY_FBOUND_3", 0.0),
    4: _env_float("BABY_FBOUND_4", 0.0),
}
# Slow pool retired: with the unified fast-pool F=0.35 both sparged runs fit
# without a separate slow population (run 4 used 0.43 here before 2026-06-15).
F_BOUND_SLOW = {
    1: _env_float("BABY_FBOUND_SLOW_1", 0.0),
    2: _env_float("BABY_FBOUND_SLOW_2", 0.0),
    3: _env_float("BABY_FBOUND_SLOW_3", 0.0),
    4: _env_float("BABY_FBOUND_SLOW_4", 0.0),
}

# --- Dose-driven bound fraction (model F): the SAME reused 1.88 kg ClLiF is
# irradiated in run order (Nov-2024..May-2025); the bound fraction of newly
# produced T grows with the cumulative prior dose in the salt (oxidative /
# radiolytic ageing). A single global trend f(dose_before) then sets every
# run's bound fraction, and the irradiation atmosphere routes it to the fast
# (He: runs 1-3) or slow (H2: run 4) pool. Crucially run4's dose-before is the
# KNOWN run1-3 total, independent of the fitted P4 -> breaks the (P4,f_slow)
# degeneracy. Enabled by BABY_F_SAT > 0 (else the per-run F_BOUND above hold).
DOSE_BEFORE = {1: 0.0, 2: 9.45e9, 3: 9.45e9 + 5.36e10,
               4: 9.45e9 + 5.36e10 + 1.66e10}
HILL_N = 3
BABY_F_SAT = _env_float("BABY_F_SAT", 0.0)
BABY_D_HALF = _env_float("BABY_D_HALF", 5.5e10)


def dose_trend(dose, f_sat=None, d_half=None):
    f_sat = BABY_F_SAT if f_sat is None else f_sat
    d_half = BABY_D_HALF if d_half is None else d_half
    if dose <= 0 or f_sat <= 0:
        return 0.0
    return f_sat * dose**HILL_N / (d_half**HILL_N + dose**HILL_N)


if BABY_F_SAT > 0:
    for _r in (1, 2, 3, 4):
        _Fr = dose_trend(DOSE_BEFORE[_r])
        if _r in (1, 2, 3):      # He irradiation -> fast pool
            F_BOUND[_r] = _Fr
            F_BOUND_SLOW[_r] = 0.0
        else:                    # run 4: H2 present during irradiation -> slow
            F_BOUND[_r] = 0.0
            F_BOUND_SLOW[_r] = _Fr

# Conversion rate at 1000 ppm H2 [1/s]; ppm-scaling exponent
# (1.0 = Henry/linear, 0.5 = Sievert-like sqrt -- fitted 0.5).
# 3e-6 (floor model 2026-06-22): with the locked floor the freed excess must
# exhaust BY the 2nd-plateau time (run3 ~day28, run4 ~day13), so kappa is faster
# than the old 1.3e-6. Global (run3 & run4 fast pool).
KAPPA_CONV_1000 = _env_float("BABY_KAPPA_1000", 3e-6)
KAPPA_EXP = _env_float("BABY_KAPPA_EXP", 0.5)
KAPPA_SLOW_1000 = _env_float("BABY_KAPPA_SLOW_1000", 2.5e-7)
KAPPA_SLOW_EXP = _env_float("BABY_KAPPA_SLOW_EXP", 0.5)


def kappa_conv(h2_ppm: float) -> float:
    """Fast-pool conversion rate [1/s] at a given H2 fraction [ppm]."""
    if h2_ppm <= 0:
        return 0.0
    return KAPPA_CONV_1000 * (h2_ppm / 1000.0) ** KAPPA_EXP


def kappa_conv_slow(h2_ppm: float) -> float:
    """Slow-pool conversion rate [1/s] at a given H2 fraction [ppm]."""
    if h2_ppm <= 0:
        return 0.0
    return KAPPA_SLOW_1000 * (h2_ppm / 1000.0) ** KAPPA_SLOW_EXP

# Inconel H2 isotopic-exchange channel (T+H -> HT). LITERATURE ONLY: the user
# rule is that NO Inconel-side parameter may be fitted -- the exchange term uses
# the unmodified htm Kr (multiplier locked at 1.0; was fitted to 2.5 before,
# reverted 2026-06-11). All compensation happens on the salt side
# (FLIBE_S_SCALE partition).
INCONEL_KEX_MULT = 1.0

# Physical (un-fudged) CLLiF Henry solubility for the H-partner concentration.
# flibe_S_0 carries a x1e15 numerical-penalty scaling (see Material props); that
# scaling is for the interface jump, not a physical solubility, so divide it out
# here to get a meaningful dissolved-H concentration.
K_flibe_phys = (flibe_S_0 / FLIBE_S_SCALE) * np.exp(
    -flibe_E_S / (8.617e-5 * temperature_K)
)


def compute_h2_conc_henry(h2_conc_ppm=1000):
    """Dissolved H concentration in CLLiF [m^-3] via Henry's law:
        c_H = K_H * p_H2        (LINEAR in p_H2)
    Used as the isotopic-exchange partner for the liquid free-surface release.
    """
    return K_flibe_phys * compute_p_h2(h2_conc_ppm)


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

    # Smoke-test override: cap the simulated time (in days) to verify a new
    # code path quickly without burning a full run.
    _cap_days = os.environ.get("BABY_FINAL_DAYS")
    if _cap_days:
        measurement_time = min(measurement_time, float(_cap_days) * 86400.0)

    # Exact H2 switch time for this run (falls back to the legacy 19 d).
    t_switch_h2 = T_SWITCH_H2_BY_RUN.get(run_id, T_SWITCH_H2)

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
        milestones.append(t_switch_h2)
        milestones = sorted(set(milestones))
    milestones = [m for m in milestones if m <= measurement_time]

    model = TimeAwareProblem()

    # --- Mesh ---
    # BABY_MESH selects an alternative mesh file (e.g. baby_2d_coarse.msh for
    # fast fitting iterations; final results use the default fine mesh).
    mesh_file = os.environ.get("BABY_MESH", "baby_2d.msh")
    _read = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, 0, gdim=2)
    mesh = _read.mesh
    cell_tags = _read.cell_tags
    facet_tags = _read.facet_tags
    model.mesh = F.Mesh(mesh=mesh, coordinate_system="cylindrical")
    model.facet_meshtags = facet_tags
    model.volume_meshtags = cell_tags

    # --- Materials ---
    # BABY_SALT_LAW switches the salt solubility law for model-comparison
    # tests (default "henry"; "sievert" makes the interface ratio linear).
    mat_cllif = F.Material(
        D_0=flibe_D_0,
        E_D=flibe_E_D,
        K_S_0=flibe_S_0,
        E_K_S=flibe_E_S,
        solubility_law=os.environ.get("BABY_SALT_LAW", "henry"),
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

    # --- Bound (TF) pools: per-run fractions of production are born bound
    # (fast / slow populations, GLOBAL kinetics, per-run composition); H2
    # converts bound -> mobile T at kappa(t) ~ p_H2(t).
    f_bound = F_BOUND.get(run_id, 0.0)
    f_bound_slow = F_BOUND_SLOW.get(run_id, 0.0)
    model.species = [T]
    model.reactions = []

    # ppm schedule for this run's sweep gas (early phase, late phase).
    if sweep_gas == "He_then_H2":
        ppm_early, ppm_late = 0, 1000
    elif sweep_gas == "H2":  # run 4: 1000 ppm then 3.5%
        ppm_early, ppm_late = 1000, 35000
    else:  # pure He
        ppm_early, ppm_late = 0, 0

    T_bound = None
    if f_bound > 0:
        T_bound = F.Species("T_bound", mobile=False, subdomains=[vol_cllif])
        model.species.append(T_bound)
        model.reactions.append(
            H2ConversionReaction(
                reactant=[T_bound],
                product=[T],
                k_0=0.0,
                E_k=0.0,
                volume=vol_cllif,
                kappa_early=kappa_conv(ppm_early),
                kappa_late=kappa_conv(ppm_late),
                t_switch=t_switch_h2,
                c_floor_early=(KAPPA_FLOOR_EARLY_BQ.get(run_id, 0.0) / _LAMBDA_T_BQ) / V_CLLIF,
                c_floor_late=(KAPPA_FLOOR_BQ.get(run_id, 0.0) / _LAMBDA_T_BQ) / V_CLLIF,
            )
        )
        # Cross-run carry-over: seed T_bound with history from earlier runs.
        bound_init_bq = BOUND_INIT_BQ.get(run_id, 0.0)
        if bound_init_bq > 0:
            c_bound_init = (bound_init_bq / _LAMBDA_T_BQ) / V_CLLIF  # atoms/m^3
            if model.initial_conditions is None:
                model.initial_conditions = []
            model.initial_conditions.append(
                F.InitialConcentration(
                    value=c_bound_init, volume=vol_cllif, species=T_bound
                )
            )

    T_bound_slow = None
    if f_bound_slow > 0:
        T_bound_slow = F.Species("T_bound_slow", mobile=False, subdomains=[vol_cllif])
        model.species.append(T_bound_slow)
        model.reactions.append(
            H2ConversionReaction(
                reactant=[T_bound_slow],
                product=[T],
                k_0=0.0,
                E_k=0.0,
                volume=vol_cllif,
                kappa_early=kappa_conv_slow(ppm_early),
                kappa_late=kappa_conv_slow(ppm_late),
                t_switch=t_switch_h2,
            )
        )

    # --- Volumetric sparge sink (negative source): T --lambda(t)--> T_sparged.
    # Only the sparged runs (3: He bubbles then +H2; 4: H2 bubbles throughout)
    # have a sparger; runs 1/2 are cover-gas only ("He" branch -> no sink).
    # T_sparged is an immobile tally species: its inventory is the cumulative
    # number of particles carried out with the sparge gas (IV-bound).
    T_sparged = None
    if sweep_gas in ("He_then_H2", "H2"):  # runs with a sparger
        lam_early, lam_late = lambda_sparge(ppm_early), lambda_sparge(ppm_late)
    else:
        lam_early = lam_late = 0.0
    if lam_early > 0 or lam_late > 0:
        T_sparged = F.Species("T_sparged", mobile=False, subdomains=[vol_cllif])
        model.species.append(T_sparged)
        model.reactions.append(
            H2ConversionReaction(
                reactant=[T],
                product=[T_sparged],
                k_0=0.0,
                E_k=0.0,
                volume=vol_cllif,
                kappa_early=lam_early,
                kappa_late=lam_late,
                t_switch=t_switch_h2,
            )
        )

    # --- Source (split mobile / bound-fast / bound-slow) ---
    def source_mobile(t):
        return (1.0 - f_bound - f_bound_slow) * tritium_source(t)

    model.sources = [
        F.ParticleSource(value=source_mobile, volume=vol_cllif, species=T),
    ]
    if T_bound is not None:

        def source_bound(t):
            return f_bound * tritium_source(t)

        model.sources.append(
            F.ParticleSource(value=source_bound, volume=vol_cllif, species=T_bound)
        )
    if T_bound_slow is not None:

        def source_bound_slow(t):
            return f_bound_slow * tritium_source(t)

        model.sources.append(
            F.ParticleSource(
                value=source_bound_slow, volume=vol_cllif, species=T_bound_slow
            )
        )

    # Late external-source injection (born mobile over a short window).
    ext_prod = EXT_INJ_PROD.get(run_id, 0.0)
    ext_day = EXT_INJ_DAY.get(run_id, 0.0)
    ext_win = EXT_INJ_WINDOW_DAY.get(run_id, 1.0)
    if ext_prod > 0 and ext_day > 0:
        ext_a = ext_day * 86400.0
        ext_b = ext_a + ext_win * 86400.0
        ext_strength = ext_prod / (ext_win * 86400.0 * V_CLLIF)

        def source_external(t, _s=ext_strength, _a=ext_a, _b=ext_b):
            return _s if _a < t <= _b else 0.0

        model.sources.append(
            F.ParticleSource(value=source_external, volume=vol_cllif, species=T)
        )

    # -----------------------------------------------------------------------
    # Recombination boundary conditions on the Inconel-gas surfaces.
    #
    # `h2_late` is the dissolved atomic-H concentration that is "on" during
    # the H2 phase (Sievert: c_H = K_s,inconel*sqrt(p_H2)).
    # It is reused below to build the post-switch export version (Method A).
    # -----------------------------------------------------------------------

    if sweep_gas == "H2":
        # Run 4 physical schedule: 1000 ppm H2 from t=0, switched to 3.5%
        # (35000 ppm) at T_SWITCH_H2. compute_h2_conc(ppm) returns the
        # DISSOLVED atomic-H concentration c_H = K_s,inconel*sqrt(p_H2), so the
        # pressure step from 1000 ppm -> 3.5% scales c_H by sqrt(35) ~ 5.9.
        # ppm_early/ppm_late are the single source of truth for the schedule
        # (shared with the bound-pool conversion and the sparge sink).
        h2_early = compute_h2_conc(ppm_early)
        h2_late = compute_h2_conc(ppm_late)

        def recombination_flux(c, T, t):
            Kr = inconel_Kr_0 * ufl.exp(-inconel_E_Kr / (F.k_B * T))
            h2 = ufl.conditional(ufl.gt(t, t_switch_h2), h2_late, h2_early)
            return -2 * Kr * c**2 - INCONEL_KEX_MULT * Kr * h2 * c

    elif sweep_gas == "He":
        h2_late = 0.0

        def recombination_flux(c, T):
            Kr = inconel_Kr_0 * ufl.exp(-inconel_E_Kr / (F.k_B * T))
            return -2 * Kr * c**2

    elif sweep_gas == "He_then_H2":
        h2_late = compute_h2_conc(ppm_late)

        def recombination_flux(c, T, t):
            Kr = inconel_Kr_0 * ufl.exp(-inconel_E_Kr / (F.k_B * T))
            # UFL symbolic conditional: H2 turns on at T_SWITCH_H2.
            # `t` here is the symbolic simulation time supplied by FESTIM.
            h2 = ufl.conditional(ufl.gt(t, t_switch_h2), h2_late, 0.0)
            return -2 * Kr * c**2 - INCONEL_KEX_MULT * Kr * h2 * c

    else:
        raise ValueError(
            f"Unknown sweep_gas '{sweep_gas}'. Use 'He', 'H2', or 'He_then_H2'."
        )

    if sweep_gas == "He_then_H2":

        def h2_conc_export(t):
            return h2_late if t > t_switch_h2 else 0.0

    elif sweep_gas == "H2":

        def h2_conc_export(t):
            return h2_late if t > t_switch_h2 else h2_early

    else:
        h2_conc_export = h2_late

    # -----------------------------------------------------------------------
    # Liquid (CLLiF) free-surface release:
    #   J = (LIQUID_KD + LIQUID_KEX * c_H(t)) * c
    # (LIQUID_KEX = 0 in the final fit: the H2 response is carried by the
    # bound-pool conversion + volumetric sparge sink, not surface kinetics.)
    # -----------------------------------------------------------------------
    def _sat(c):
        """Langmuir saturation factor on the free-surface desorption."""
        if LIQUID_CSAT > 0:
            return 1.0 + c / LIQUID_CSAT
        return 1.0

    if sweep_gas == "H2":
        h2_early_liq = compute_h2_conc_henry(ppm_early)
        h2_late_liq = compute_h2_conc_henry(ppm_late)

        def liquid_surface_flux(c, T, t):
            c_H = ufl.conditional(ufl.gt(t, t_switch_h2), h2_late_liq, h2_early_liq)
            return -(LIQUID_KD + LIQUID_KEX * c_H) * c / _sat(c)

    elif sweep_gas == "He":
        h2_late_liq = 0.0

        def liquid_surface_flux(c, T):
            return -LIQUID_KD * c / _sat(c)

    elif sweep_gas == "He_then_H2":
        h2_late_liq = compute_h2_conc_henry(ppm_late)

        def liquid_surface_flux(c, T, t):
            c_H = ufl.conditional(ufl.gt(t, t_switch_h2), h2_late_liq, 0.0)
            return -(LIQUID_KD + LIQUID_KEX * c_H) * c / _sat(c)

    # k_eff(t) for the export, J = k_eff(t)*c.
    if sweep_gas == "He_then_H2":

        def liquid_keff_export(t):
            c_H = h2_late_liq if t > t_switch_h2 else 0.0
            return LIQUID_KD + LIQUID_KEX * c_H

    elif sweep_gas == "H2":

        def liquid_keff_export(t):
            c_H = h2_late_liq if t > t_switch_h2 else h2_early_liq
            return LIQUID_KD + LIQUID_KEX * c_H

    else:
        liquid_keff_export = LIQUID_KD

    # Paused speciation-model law (constant k, sparge in K_SPARGE):
    # k_liq = LIQUID_KD + K_SPARGE.get(run_id, 0.0)
    #
    # def liquid_surface_flux(c, T):
    #     return -k_liq * c
    #
    # liquid_keff_export = k_liq

    subfolder = f"{results_folder}/run_{run_id}/sweep_{sweep_gas}"

    # Config-dependent exports linger when a results folder is reused with a
    # different configuration (FESTIM never deletes files it does not write);
    # plot/eval include them via os.path.exists, silently inflating IV and the
    # mass balance. Remove any stale optional inventory CSVs up front.
    for _fname in (
        "inventory_bound.csv",
        "inventory_bound_slow.csv",
        "inventory_sparged.csv",
    ):
        _p = os.path.join(subfolder, _fname)
        if os.path.exists(_p):
            os.remove(_p)

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

    # Liquid free-surface release: linear desorption + isotopic exchange
    # J = -(k_d + k_ex*c_H)*c  (surface-limited regime; large flibe D).
    liquid_recomb_bc = F.ParticleFluxBC(
        value=liquid_surface_flux,
        subdomain=liquid_surface,
        species_dependent_value={"c": T},
        species=T,
    )
    liquid_recomb_bc._volume_subdomain = vol_cllif

    model.boundary_conditions = [
        liquid_recomb_bc,
        *recomb_bcs,
    ]

    model.temperature = temperature_K

    dt = F.Stepsize(
        initial_value=_env_float("BABY_DT_INIT", 10),
        growth_factor=_env_float("BABY_DT_GROWTH", 1.1),
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
    # kex_0 mirrors the BC: exchange channel = INCONEL_KEX_MULT * Kr (isothermal).
    inconel_kex_eff = (
        INCONEL_KEX_MULT
        * inconel_Kr_0
        * np.exp(-inconel_E_Kr / (8.617e-5 * temperature_K))
    )

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
            kex_0=inconel_kex_eff,
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

    model.exports = [
        # Concentration fields
        F.VTXSpeciesExport(
            filename=f"{subfolder}/T_cllif.bp", field=T, subdomain=vol_cllif
        ),
        F.VTXSpeciesExport(
            filename=f"{subfolder}/T_inconel.bp", field=T, subdomain=vol_inconel
        ),
        # ---- Liquid free surface release (linear, J = k_eff(t)*c) ----
        CylindricalSurfaceFluxMassTransfer(
            field=T,
            surface=liquid_surface,
            filename=f"{subfolder}/flux_liquid_surface.csv",
            k=liquid_keff_export,
            name="liquid surface",
            c_sat=LIQUID_CSAT,
        ),
        # ---- Recomb-eq (physical release) fluxes on all Inconel surfaces ----
        *recomb_eq_exports,
        # ---- Tritium inventory per region ----
        CylindricalTotalVolume(
            field=T, volume=vol_cllif, filename=f"{subfolder}/inventory_cllif.csv"
        ),
        CylindricalTotalVolume(
            field=T, volume=vol_inconel, filename=f"{subfolder}/inventory_inconel.csv"
        ),
    ]

    # Bound (TF) pool inventories, for mass balance.
    if T_bound is not None:
        model.exports.append(
            CylindricalTotalVolume(
                field=T_bound,
                volume=vol_cllif,
                filename=f"{subfolder}/inventory_bound.csv",
            )
        )
    if T_bound_slow is not None:
        model.exports.append(
            CylindricalTotalVolume(
                field=T_bound_slow,
                volume=vol_cllif,
                filename=f"{subfolder}/inventory_bound_slow.csv",
            )
        )

    # Cumulative sparge extraction (IV-bound; added to IV in plot/eval).
    if T_sparged is not None:
        model.exports.append(
            CylindricalTotalVolume(
                field=T_sparged,
                volume=vol_cllif,
                filename=f"{subfolder}/inventory_sparged.csv",
            )
        )

    return model, T, vol_cllif, vol_inconel


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BABY-1L 2D FESTIM runs")
    parser.add_argument("--runs", default="1,2,3,4",
                        help="comma-separated run ids (default: all four)")
    parser.add_argument("--out", default="results/baby_2d",
                        help="results folder (per-case folders for fitting)")
    args = parser.parse_args()
    run_ids = [int(r) for r in args.runs.split(",") if r.strip()]

    # Sweep-gas schedule per run.
    #   run 1, 2 : pure He for the whole run.
    #   run 3    : He, then 1000 ppm H2 switched on at day 18.21.
    #   run 4    : 1000 ppm H2 from the start, then 3.5% H2 at day 17.37.
    RUN_SWEEP = {
        1: "He",
        2: "He",
        3: "He_then_H2",
        4: "H2",
    }

    print("=== Salt-side fit parameters (Inconel: literature only) ===")
    print(f"  FLIBE_D_MULT       = {FLIBE_D_MULT:g}")
    print(f"  FLIBE_S_SCALE      = {FLIBE_S_SCALE:g}")
    print(f"  LIQUID_KD          = {LIQUID_KD:g}")
    print(f"  LIQUID_KEX         = {LIQUID_KEX:g}")
    print(f"  LIQUID_CSAT        = {LIQUID_CSAT:g}")
    print(f"  LAMBDA_SPARGE_HE   = {LAMBDA_SPARGE_HE:g}")
    print(f"  LAMBDA_SPARGE_1000 = {LAMBDA_SPARGE_1000:g} (exp {LAMBDA_EXP:g})")
    print(f"  KAPPA_CONV_1000    = {KAPPA_CONV_1000:g} (exp {KAPPA_EXP:g})")
    print(f"  KAPPA_SLOW_1000    = {KAPPA_SLOW_1000:g} (exp {KAPPA_SLOW_EXP:g})")
    print(f"  F_BOUND            = {F_BOUND}")
    print(f"  F_BOUND_SLOW       = {F_BOUND_SLOW}")
    if BABY_F_SAT > 0:
        print(f"  DOSE-DRIVEN: f_sat={BABY_F_SAT:g} D_half={BABY_D_HALF:g} (Hill n={HILL_N})")
    print(f"  TRITIUM_PROD[4]    = {TRITIUM_PRODUCTION[4]:g}")
    print(f"  penalty            = {penalty:g}")
    print(f"  salt law           = {os.environ.get('BABY_SALT_LAW', 'henry')}")
    print(f"  results folder     = {args.out}")

    for run_id in run_ids:
        sweep = RUN_SWEEP[run_id]
        print(f"\n=== Run {run_id} / {sweep} sweep ===")
        model, T, vol_cllif, vol_inconel = build_model(
            sweep_gas=sweep, run_id=run_id, results_folder=args.out
        )
        model.initialise()
        for export in model.exports:
            if hasattr(export, "bind_time_constant"):
                export.bind_time_constant(model.t)
        model.run()

        from dolfinx import geometry

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
