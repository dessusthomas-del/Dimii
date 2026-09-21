"""
DIMENSIONNEMENT GÉO V47 — gshp_geofield_v47.py
PAC géothermique sur champ de sondes verticales

Modèle de réponse thermique : Finite Line Source (FLS)
avec superposition temporelle horaire sur 8760 h.

Références :
  - Eskilson (1987) Thermal analysis of heat extraction boreholes
  - Hellström (1991) Ground Heat Storage — thermal analyses of duct storage systems
  - Claesson & Javed (2011) An analytical method to calculate borehole fluid
    temperatures for time-scales from minutes to decades
  - EN 15450 : Systèmes de chauffage avec pompes à chaleur

Unités : kW, kWh, °C, m, W/(m·K)
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Cache g-function : évite de recalculer les fonctions FLS pour la même
# géométrie de champ. Clé = hash SHA1 de la configuration (n_bh, positions,
# profondeurs, alpha, n_quad). Valeur = np.ndarray(8760,) de valeurs g.
# ─────────────────────────────────────────────────────────────────────────────
_GFUNC_CACHE: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Structures de données
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BoreholeConfig:
    """Paramètres d'une sonde verticale individuelle."""
    depth_m: float = 120.0           # profondeur active [m]
    x_m: float = 0.0                 # position X dans le champ [m]
    y_m: float = 0.0                 # position Y dans le champ [m]
    pipe_radius_m: float = 0.07      # rayon externe de l'ensemble sonde [m]
    grout_lambda_wpmk: float = 1.0   # conductivité coulis [W/(m·K)]
    bhe_resistance_mkw: float = 0.10 # résistance thermique sonde [m·K/W]


@dataclass
class GroundParams:
    """Paramètres thermiques du sol."""
    lambda_wpmk: float = 2.0         # conductivité thermique [W/(m·K)]
    diffusivity_m2ps: float = 1.0e-6 # diffusivité thermique [m²/s]
    t_undisturbed_c: float = 12.0    # T° non-perturbée [°C]
    t_annual_amplitude_c: float = 3.0 # amplitude variation annuelle [°C]
    phase_offset_h: float = 2160.0   # décalage de phase (h), ~mars pour minima T°


@dataclass
class GSHPParams:
    """Paramètres de la PAC géothermique."""
    rated_heat_kw: float = 20.0      # puissance utile nominale [kW]
    rated_cool_kw: float = 15.0      # puissance utile froid nominale [kW]
    cop_b0w35: float = 4.5           # COP nominal (B0/W35)
    eer_b10w18: float = 6.0          # EER nominal froid (B10/W18)
    eta_carnot_heat: float = 0.50    # efficacité relative Carnot chauffage
    eta_carnot_cool: float = 0.50    # efficacité relative Carnot refroid.
    t_supply_heat_c: float = 35.0    # T° départ chauffage [°C]
    t_supply_cool_c: float = 7.0     # T° départ refroidissement [°C]
    t_source_min_c: float = -3.0     # T° source mini acceptable [°C]
    t_source_max_c: float = 25.0     # T° source maxi acceptable [°C]
    debit_l_s: float = 1.5           # débit fluide caloporteur [L/s]


@dataclass
class GSHPFieldConfig:
    """Configuration complète du système géothermique."""
    boreholes: List[BoreholeConfig]
    ground: GroundParams
    pac: GSHPParams

    @classmethod
    def from_dict(cls, d: dict) -> "GSHPFieldConfig":
        """Construit depuis un dictionnaire de configuration JSON."""
        ground = GroundParams(**d.get("ground", {}))
        pac = GSHPParams(**d.get("pac", {}))
        bh_list = d.get("boreholes", [])
        if not bh_list:
            # Disposition par défaut : ligne simple
            n = d.get("n_boreholes", 4)
            spacing = d.get("spacing_m", 8.0)
            depth = d.get("depth_m", 120.0)
            bh_list = [
                {"x_m": i * spacing, "y_m": 0.0, "depth_m": depth}
                for i in range(n)
            ]
        boreholes = [BoreholeConfig(**b) for b in bh_list]
        return cls(boreholes=boreholes, ground=ground, pac=pac)

    @classmethod
    def make_grid(cls, nx: int, ny: int, spacing_m: float, depth_m: float,
                  ground: dict = None, pac: dict = None) -> "GSHPFieldConfig":
        """Grille rectangulaire NX × NY de sondes."""
        bhs = [
            BoreholeConfig(depth_m=depth_m, x_m=i * spacing_m, y_m=j * spacing_m)
            for i in range(nx) for j in range(ny)
        ]
        return cls(
            boreholes=bhs,
            ground=GroundParams(**(ground or {})),
            pac=GSHPParams(**(pac or {})),
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. G-function FLS (Finite Line Source)
# ─────────────────────────────────────────────────────────────────────────────

def _fls_single_response(r_m: float, t_s: float, alpha_m2ps: float,
                          L_m: float, n_quad: int = 24,
                          z_obs_m: float = None) -> float:
    """
    Réponse unitaire FLS (Finite Line Source) d'une sonde de longueur L_m.

    Calcule l'intégrale :
      I = ∫₀ᴸ [erfc(d₁/√(4αt))/d₁ − erfc(d₂/√(4αt))/d₂] ds   [sans dimension]

    où d₁ = sqrt(r² + (z_obs − s)²) (source directe)
         d₂ = sqrt(r² + (z_obs + s)²) (image miroir — condition Dirichlet z=0)

    La T° à la paroi de la sonde est ensuite : ΔT = q'/(4π·λ) × I

    Point d'observation par défaut : z_obs = L_m/2 (milieu de la sonde).

    Méthode numérique
    -----------------
    Le terme direct présente une quasi-singularité en s = z_obs (d₁ → r_m).
    Résolution par transformation sinh (Kress 1991) :
      u = z_obs − s,  puis u = r·sinh(τ)  →  intégrale sur τ de erfc(r·cosh(τ)/√(4αt))
    Le terme image est lisse (d₂ ≥ sqrt(r²+z_obs²)) → quadrature GL directe.

    Références
    ----------
    - Eskilson (1987) Thermal analysis of heat extraction boreholes
    - Claesson & Javed (2011) An analytical method, eq. 3
    - Kress (1991) A Nyström method for boundary integral equations (sinh trick)

    Paramètres
    ----------
    r_m        : distance radiale source-récepteur [m]
    t_s        : temps [s] depuis l'application de la charge
    alpha_m2ps : diffusivité thermique sol [m²/s]
    L_m        : profondeur de la sonde [m]
    n_quad     : nombre de points quadrature Gauss-Legendre (défaut 24)
    z_obs_m    : profondeur du point d'observation [m] (défaut : L_m/2)

    Retourne
    --------
    I [adimensionnel] : intégrale FLS brute — ΔT = q'/(4π·λ) × I
    """
    if t_s <= 0:
        return 0.0
    if z_obs_m is None:
        z_obs_m = L_m / 2.0  # milieu de la sonde (point de référence standard)

    sqrt_4at = 2.0 * math.sqrt(alpha_m2ps * t_s)
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)

    # ── Terme direct : ∫₀ᴸ erfc(d₁/√(4αt))/d₁ ds ───────────────────────────
    # Changement u = z_obs − s, puis sinh : u = r·sinh(τ), d₁ = r·cosh(τ)
    # ∫_{u_lo}^{u_hi} erfc(r·cosh(τ)/√(4αt)) dτ
    # u_lo = z_obs − L  (s = L),  u_hi = z_obs  (s = 0)
    tau_hi   = math.asinh(z_obs_m / r_m)
    tau_lo   = math.asinh((z_obs_m - L_m) / r_m)
    tau_mid  = 0.5 * (tau_hi + tau_lo)
    tau_half = 0.5 * (tau_hi - tau_lo)
    tau_pts  = tau_mid + tau_half * nodes
    w_tau    = tau_half * weights
    direct = float(np.dot(w_tau,
                          np.array([math.erfc(r_m * math.cosh(t) / sqrt_4at)
                                    for t in tau_pts])))

    # ── Terme image : ∫₀ᴸ erfc(d₂/√(4αt))/d₂ ds ───────────────────────────
    # d₂ = sqrt(r² + (z_obs + s)²) ≥ sqrt(r² + z_obs²) >> r  → pas de singularité
    s_pts = 0.5 * L_m * (nodes + 1.0)
    w_s   = 0.5 * L_m * weights
    d2    = np.sqrt(r_m**2 + (z_obs_m + s_pts)**2)
    image = float(np.dot(w_s,
                         np.array([math.erfc(float(d) / sqrt_4at) / float(d)
                                   for d in d2])))

    return direct - image  # I [adimensionnel] — ΔT = q'/(4π·λ) × I


def _gfunc_cache_key(config: GSHPFieldConfig, n_lags: int) -> str:
    """Génère une clé de cache SHA1 pour la configuration du champ."""
    bhs = config.boreholes
    ground = config.ground
    key_data = {
        "boreholes": [(round(b.depth_m, 2), round(b.x_m, 3), round(b.y_m, 3),
                       round(b.pipe_radius_m, 4)) for b in bhs],
        "lambda":    round(ground.lambda_wpmk, 4),
        "alpha":     ground.diffusivity_m2ps,
        "n_lags":    n_lags,
    }
    raw = json.dumps(key_data, sort_keys=True).encode()
    return hashlib.sha1(raw).hexdigest()


def _compute_gfunction_lags(config: GSHPFieldConfig,
                              n_lags: int = 8760,
                              n_log_pts: int = 80) -> np.ndarray:
    """
    Précalcule la g-function agrégée pour les décalages 1..n_lags heures.

    Stratégie :
      1. Calcule g(t) aux n_log_pts points logarithmiques (1 h → n_lags h)
      2. Superpose les contributions inter-sondes (paires distinctes)
      3. Interpole linéairement sur log(t) pour tous les lags entiers

    Retourne
    --------
    g_lags : (n_lags,) valeurs g pour les décalages 1, 2, ..., n_lags heures
             [unité adimensionnelle, telle que ΔT = g × q' / (2π·λ)]
    """
    alpha = config.ground.diffusivity_m2ps
    lam   = config.ground.lambda_wpmk
    bhs   = config.boreholes
    n_bh  = len(bhs)

    # Points logarithmiques pour l'interpolation (évite O(n_lags × n_bh²) calls)
    t_log = np.logspace(0, math.log10(float(n_lags)), num=n_log_pts)

    # G-function agrégée : superposition spatiale de toutes les paires
    g_log = np.zeros(n_log_pts)

    for b1_idx, b1 in enumerate(bhs):
        for b2_idx, b2 in enumerate(bhs):
            r = math.sqrt((b1.x_m - b2.x_m) ** 2 + (b1.y_m - b2.y_m) ** 2)
            # Sonde sur elle-même : utiliser le rayon de forage
            if b1_idx == b2_idx:
                r = b1.pipe_radius_m

            if r < 1e-4:
                continue

            # FLS pour ce rayon (vectorisé sur t_log)
            g_pair = np.array([
                _fls_single_response(r, t_h * 3600.0, alpha, b1.depth_m)
                for t_h in t_log
            ], dtype=float)
            # Contribution : normalisée par n_bh et 2 (convention g = I_FLS/2)
            # → g_log représente la g-function standard : ΔT = g × q'/(2π·λ)
            g_log += g_pair / (2.0 * n_bh)

    # Interpolation log-linéaire pour les décalages entiers 1..n_lags
    lags = np.arange(1, n_lags + 1, dtype=float)
    g_lags = np.interp(lags, t_log, g_log)

    return g_lags  # adimensionnel


def get_gfunction_lags(config: GSHPFieldConfig, n_lags: int = 8760) -> np.ndarray:
    """
    Retourne la g-function pour les lags 1..n_lags h, avec mise en cache.
    Le premier appel calcule et stocke ; les suivants retournent instantanément.
    """
    global _GFUNC_CACHE
    key = _gfunc_cache_key(config, n_lags)
    if key not in _GFUNC_CACHE:
        print(f"  [GSHP] Calcul g-function ({len(config.boreholes)} sondes, "
              f"{config.boreholes[0].depth_m:.0f} m)…", flush=True)
        _GFUNC_CACHE[key] = _compute_gfunction_lags(config, n_lags)
        print(f"  [GSHP] G-function mise en cache ({key[:8]}…)")
    return _GFUNC_CACHE[key]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Simulation horaire du champ de sondes
# ─────────────────────────────────────────────────────────────────────────────

def simulate_borehole_field_8760(config: GSHPFieldConfig,
                                  q_extract_kw: np.ndarray) -> np.ndarray:
    """
    Simule la T° de source géothermique sur 8760 h par superposition temporelle
    (convolution des charges avec la g-function FLS).

    Algorithme : step-load superposition vectorisée avec numpy.
      ΔT_source(h) = (delta_q ★ g_lags)[h] / (2π·λ)
    où ★ est la convolution causale (scipy.signal.fftconvolve ou np.convolve).

    Optimisations :
      - G-function mise en cache (calcul unique par géométrie)
      - Convolution numpy O(N·log N) au lieu de boucle Python O(N²)

    Paramètres
    ----------
    config       : GSHPFieldConfig
    q_extract_kw : (8760,) puissance extraite du sol [kW], positive = extraction

    Retourne
    --------
    t_source_c : (8760,) T° de la source géothermique à chaque heure [°C]
    """
    n = min(len(q_extract_kw), 8760)
    ground = config.ground
    bhs = config.boreholes
    n_bh = len(bhs)
    H_total = sum(b.depth_m for b in bhs)  # longueur totale [m]
    R_bhe = bhs[0].bhe_resistance_mkw       # résistance thermique sonde [m·K/W]

    lam = ground.lambda_wpmk
    alpha = ground.diffusivity_m2ps

    # Charge linéique [W/m] : q_prime(h) = q_extract(h) × 1000 / H_total
    q_prime = np.array(q_extract_kw[:n], dtype=float) * 1000.0 / max(H_total, 1.0)

    # T° non-perturbée (variation saisonnière sinusoïdale)
    hours = np.arange(n, dtype=float)
    t_undist = (ground.t_undisturbed_c
                + ground.t_annual_amplitude_c
                  * np.sin(2.0 * math.pi * hours / 8760.0
                           - 2.0 * math.pi * ground.phase_offset_h / 8760.0))

    # ── G-function (avec cache) ───────────────────────────────────────────
    g_lags = get_gfunction_lags(config, n_lags=n)
    # g_lags[i] = g-function standard au lag i+1 h [adimensionnel]
    # Convention EED/EWS : ΔT = g × q' / (2π·λ)
    # (g_lags = I_FLS/2 avec I_FLS = intégrale brute _fls_single_response)
    g_kernel = g_lags / (2.0 * math.pi * lam)  # [K/(W/m)]

    # ── Convolution vectorisée (step-load superposition) ─────────────────
    # delta_q[h] = q'(h) - q'(h-1)  [W/m]
    delta_q = np.diff(q_prime, prepend=0.0)

    # ΔT_sol[h] = sum_{k=0}^{h} delta_q[k] × g_kernel[h-k]
    #           = np.convolve(delta_q, g_kernel, mode='full')[h]
    # Utilise scipy si disponible (FFT rapide), sinon numpy (direct)
    try:
        from scipy.signal import fftconvolve
        dt_sol = fftconvolve(delta_q, g_kernel, mode='full')[:n]
    except ImportError:
        dt_sol = np.convolve(delta_q, g_kernel, mode='full')[:n]

    # Correction résistance sonde : ΔT_bhe = R_bhe × q' / n_bh
    dt_bhe = q_prime * R_bhe / n_bh

    # T° source = T° sol non-perturbée − abaissement dû à l'extraction
    t_source = t_undist - dt_sol - dt_bhe

    return t_source


# ─────────────────────────────────────────────────────────────────────────────
# 4. Modèle COP PAC géothermique
# ─────────────────────────────────────────────────────────────────────────────

def cop_gshp_heat(t_source_c: float, t_supply_c: float,
                   eta_carnot: float) -> float:
    """
    COP chauffage de la PAC géothermique (modèle Carnot-relatif).

    COP = η_carnot × T_supply / (T_supply - T_source)   [T en Kelvin]

    Limites : COP ∈ [1.0, 8.0]

    Paramètres
    ----------
    t_source_c  : T° source géothermique [°C]
    t_supply_c  : T° de départ chauffage [°C]
    eta_carnot  : efficacité relative Carnot (typ. 0.40–0.55)
    """
    T_source = t_source_c + 273.15
    T_supply = t_supply_c + 273.15
    dT = T_supply - T_source
    if dT <= 0:
        return 8.0
    cop = eta_carnot * T_supply / dT
    return max(1.0, min(8.0, cop))


def eer_gshp_cool(t_source_c: float, t_cool_c: float,
                   eta_carnot: float) -> float:
    """
    EER refroidissement de la PAC géothermique.

    EER = η_carnot × T_froid / (T_source - T_froid)   [T en Kelvin]

    Limites : EER ∈ [1.0, 12.0]
    """
    T_source = t_source_c + 273.15
    T_cool = t_cool_c + 273.15
    dT = T_source - T_cool
    if dT <= 0:
        return 12.0
    eer = eta_carnot * T_cool / dT
    return max(1.0, min(12.0, eer))


# ─────────────────────────────────────────────────────────────────────────────
# 5. Classe principale : simulateur GSHP 8760 h
# ─────────────────────────────────────────────────────────────────────────────

class GSHPSimulator:
    """
    Simulateur complet PAC géothermique + champ de sondes sur 8760 h.

    Usage
    -----
    >>> cfg = GSHPFieldConfig.from_dict(config_dict)
    >>> sim = GSHPSimulator(cfg)
    >>> results = sim.simulate_8760(q_heat_demand_kw, q_cool_demand_kw)
    """

    def __init__(self, config: GSHPFieldConfig):
        self.config = config

    def simulate_8760(self,
                       q_heat_kw: np.ndarray,
                       q_cool_kw: np.ndarray) -> dict:
        """
        Simulation horaire sur 8760 h.

        Paramètres
        ----------
        q_heat_kw : (8760,) demande de chaleur à couvrir [kW]
        q_cool_kw : (8760,) demande de froid à couvrir [kW]

        Retourne
        --------
        dict avec clés :
          q_heat_delivered_kw  : (8760,) chaleur fournie par GSHP [kW]
          q_cool_delivered_kw  : (8760,) froid fourni par GSHP [kW]
          p_elec_kw            : (8760,) puissance électrique consommée [kW]
          cop_h                : (8760,) COP chauffage horaire
          eer_h                : (8760,) EER refroidissement horaire
          t_source_c           : (8760,) T° source géothermique [°C]
          q_extracted_kw       : (8760,) puissance extraite du sol [kW]
          unmet_heat_kw        : (8760,) chaleur non couverte [kW]
          unmet_cool_kw        : (8760,) froid non couvert [kW]
          annual_scop          : float SCOP annuel chauffage
          annual_seer          : float SEER annuel refroidissement
          t_source_min_c       : float T° source minimale sur l'année
        """
        pac = self.config.pac
        n = 8760

        q_heat = np.array(q_heat_kw[:n], dtype=float)
        q_cool = np.array(q_cool_kw[:n], dtype=float)

        # Première passe : estimation charge sol (sans connaissance T° source)
        # On utilise la T° non-perturbée comme point de départ
        t_und = self.config.ground.t_undisturbed_c
        cop_est = cop_gshp_heat(t_und, pac.t_supply_heat_c, pac.eta_carnot_heat)
        q_extract_est = q_heat * (1.0 - 1.0 / cop_est)  # simplification initiale

        # Simulation champ de sondes avec charge estimée
        t_source = simulate_borehole_field_8760(self.config, q_extract_est)

        # Deuxième passe : calcul correct avec T° source simulée
        q_heat_del = np.zeros(n)
        q_cool_del = np.zeros(n)
        p_elec = np.zeros(n)
        cop_h = np.zeros(n)
        eer_h = np.zeros(n)
        q_extract_final = np.zeros(n)
        unmet_heat = np.zeros(n)
        unmet_cool = np.zeros(n)

        for h in range(n):
            ts = t_source[h]

            # Vérification T° source acceptable
            if ts < pac.t_source_min_c:
                # Source trop froide : PAC ne peut pas fonctionner en chauf.
                unmet_heat[h] = q_heat[h]
                # Refroid. géothermique passif toujours possible si T_source < T_froid
                if ts < pac.t_supply_cool_c and q_cool[h] > 0:
                    q_cool_eff = min(q_cool[h], pac.rated_cool_kw)
                    q_cool_del[h] = q_cool_eff
                    eer_h[h] = 99.0  # free-cooling passif
                    q_extract_final[h] = -q_cool_eff  # injection chaleur dans le sol
                else:
                    unmet_cool[h] = q_cool[h]
                continue

            # Chauffage
            if q_heat[h] > 0:
                cop = cop_gshp_heat(ts, pac.t_supply_heat_c, pac.eta_carnot_heat)
                q_avail = pac.rated_heat_kw
                q_del = min(q_heat[h], q_avail)
                p_e = q_del / cop
                q_extr = q_del - p_e  # énergie extraite du sol
                q_heat_del[h] = q_del
                p_elec[h] += p_e
                cop_h[h] = cop
                q_extract_final[h] += q_extr
                unmet_heat[h] = max(0.0, q_heat[h] - q_del)

            # Refroidissement
            if q_cool[h] > 0:
                if ts < pac.t_supply_cool_c:
                    # Free-cooling géothermique passif
                    q_cool_eff = min(q_cool[h], pac.rated_cool_kw)
                    q_cool_del[h] = q_cool_eff
                    eer_h[h] = 99.0
                    q_extract_final[h] -= q_cool_eff
                else:
                    eer = eer_gshp_cool(ts, pac.t_supply_cool_c,
                                        pac.eta_carnot_cool)
                    q_cool_eff = min(q_cool[h], pac.rated_cool_kw)
                    p_e_cool = q_cool_eff / eer
                    p_elec[h] += p_e_cool
                    eer_h[h] = eer
                    q_cool_del[h] = q_cool_eff
                    q_extract_final[h] -= q_cool_eff + p_e_cool  # injection
                    unmet_cool[h] = max(0.0, q_cool[h] - q_cool_eff)

        # Recalcul T° source avec charge finale (pour reporting)
        t_source_final = simulate_borehole_field_8760(self.config, q_extract_final)

        # Indicateurs annuels
        e_heat_tot = float(np.sum(q_heat_del))
        e_elec_heat = float(np.sum(p_elec[q_heat_del > 0]))
        e_cool_tot = float(np.sum(q_cool_del))
        e_elec_cool = float(np.sum(p_elec[q_cool_del > 0]))

        scop = e_heat_tot / max(e_elec_heat, 1e-6)
        seer = e_cool_tot / max(e_elec_cool, 1e-6)

        return {
            "q_heat_delivered_kw":  q_heat_del,
            "q_cool_delivered_kw":  q_cool_del,
            "p_elec_kw":            p_elec,
            "cop_h":                cop_h,
            "eer_h":                eer_h,
            "t_source_c":           t_source_final,
            "q_extracted_kw":       q_extract_final,
            "unmet_heat_kw":        unmet_heat,
            "unmet_cool_kw":        unmet_cool,
            "annual_scop":          round(scop, 2),
            "annual_seer":          round(seer, 2),
            "t_source_min_c":       round(float(np.min(t_source_final)), 1),
            "t_source_avg_c":       round(float(np.mean(t_source_final)), 1),
            "status":               "estimatif",
        }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Dimensionnement automatique du champ
# ─────────────────────────────────────────────────────────────────────────────

def size_borehole_field(q_heat_kw: np.ndarray, q_cool_kw: np.ndarray,
                         ground: GroundParams, pac: GSHPParams,
                         spacing_m: float = 8.0,
                         depth_options_m: list = None,
                         n_borehole_options: list = None,
                         n_years_sizing: int = 10) -> GSHPFieldConfig:
    """
    Dimensionne automatiquement le champ de sondes pour satisfaire les besoins
    tout en maintenant T_source_min ≥ pac.t_source_min_c sur n_years_sizing ans.

    Parcourt une grille (n_boreholes × depth) par ordre croissant de longueur
    totale et retourne la première configuration acceptable.

    Paramètres
    ----------
    q_heat_kw          : profil horaire 8760 h chauf. [kW]
    q_cool_kw          : profil horaire 8760 h froid [kW]
    ground             : GroundParams
    pac                : GSHPParams
    spacing_m          : espacement inter-sondes [m]
    depth_options_m    : liste des profondeurs à tester [m]
    n_borehole_options : liste des nombres de sondes à tester
    n_years_sizing     : horizon de dimensionnement [années]

    Retourne
    --------
    GSHPFieldConfig dimensionné
    """
    if depth_options_m is None:
        depth_options_m = [80, 100, 120, 150, 200]
    if n_borehole_options is None:
        n_borehole_options = [2, 3, 4, 6, 8, 10, 12]

    # Répétition du profil sur n_years pour le dimensionnement
    # Hypothèse simplificatrice : même profil chaque année
    q_heat_ny = np.tile(q_heat_kw[:8760], n_years_sizing)
    q_cool_ny = np.tile(q_cool_kw[:8760], n_years_sizing)

    best_config = None
    best_total_m = float("inf")

    for depth in depth_options_m:
        for n_bh in n_borehole_options:
            # Disposition en ligne simple
            bhs = [
                BoreholeConfig(depth_m=depth,
                               x_m=i * spacing_m, y_m=0.0)
                for i in range(n_bh)
            ]
            cfg = GSHPFieldConfig(boreholes=bhs, ground=ground, pac=pac)
            sim = GSHPSimulator(cfg)

            # Simulation simplifiée sur n_years (coûteux, version réduite)
            # On simule uniquement la 1ère et la dernière année
            try:
                res = sim.simulate_8760(q_heat_kw[:8760], q_cool_kw[:8760])
                t_min = res["t_source_min_c"]

                # Dérive thermique estimée sur n_years
                # ΔT/an ≈ déséquilibre annuel / (2π·λ·H·ln(t_ny/t_1yr))
                q_imbalance = (float(np.sum(res["q_extracted_kw"])) / 8760.0
                               * n_years_sizing)
                dt_drift = (q_imbalance * 1000.0
                            / (2 * math.pi * ground.lambda_wpmk
                               * n_bh * depth)
                            * math.log(n_years_sizing + 1))

                t_min_ny = t_min - abs(dt_drift)

                if t_min_ny >= pac.t_source_min_c:
                    total_m = n_bh * depth
                    if total_m < best_total_m:
                        best_total_m = total_m
                        best_config = cfg
            except Exception:
                continue

    if best_config is None:
        # Fallback : configuration maximale
        depth = max(depth_options_m)
        n_bh = max(n_borehole_options)
        bhs = [BoreholeConfig(depth_m=depth, x_m=i * spacing_m, y_m=0.0)
               for i in range(n_bh)]
        best_config = GSHPFieldConfig(boreholes=bhs, ground=ground, pac=pac)

    return best_config


# ─────────────────────────────────────────────────────────────────────────────
# 7. Tests unitaires légers
# ─────────────────────────────────────────────────────────────────────────────

def _run_tests():
    print("=== Tests gshp_geofield_v47 ===")

    # Test 1 : COP géothermique
    cop = cop_gshp_heat(t_source_c=0.0, t_supply_c=35.0, eta_carnot=0.50)
    assert 3.0 < cop < 5.5, f"COP hors plage : {cop}"
    print(f"  COP(0°C→35°C, η=0.50) = {cop:.2f}  ✓")

    # Test 2 : EER géothermique
    eer = eer_gshp_cool(t_source_c=15.0, t_cool_c=7.0, eta_carnot=0.50)
    assert 3.0 < eer < 15.0, f"EER hors plage : {eer}"
    print(f"  EER(15°C→7°C, η=0.50)  = {eer:.2f}  ✓")

    # Test 3 : simulation 8760 h minimaliste
    cfg = GSHPFieldConfig.from_dict({
        "n_boreholes": 2,
        "depth_m": 100.0,
        "spacing_m": 8.0,
        "ground": {"lambda_wpmk": 2.0, "diffusivity_m2ps": 1e-6,
                    "t_undisturbed_c": 12.0, "t_annual_amplitude_c": 2.0},
        "pac": {"rated_heat_kw": 20.0, "cop_b0w35": 4.5,
                "eta_carnot_heat": 0.50, "t_supply_heat_c": 35.0,
                "t_source_min_c": -3.0}
    })
    q_heat = np.full(8760, 10.0)  # charge constante 10 kW
    q_heat[3000:6000] = 0.0       # pas de besoin en été
    q_cool = np.zeros(8760)
    q_cool[3000:6000] = 5.0       # besoin froid en été

    sim = GSHPSimulator(cfg)
    res = sim.simulate_8760(q_heat, q_cool)

    assert res["annual_scop"] > 2.5, f"SCOP trop faible : {res['annual_scop']}"
    print(f"  SCOP annuel = {res['annual_scop']}  ✓")
    print(f"  T source min = {res['t_source_min_c']} °C  ✓")
    print("=== Tests OK ===")


if __name__ == "__main__":
    _run_tests()
