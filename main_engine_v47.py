"""
DIMENSIONNEMENT GÉO V47 — main_engine_v47.py
Orchestrateur principal du moteur de simulation multi-sources

Point d'entrée unique :
  1. Charge la configuration JSON du projet
  2. Charge les profils de charge (CSV 8760 h) et données météo
  3. Appelle dispatch_multisource_v47 pour la simulation horaire
  4. Calcule les indicateurs économiques (coûts, aides, TRS/VAN/TRI)
  5. Exporte les résultats (CSV horaire + JSON synthèse)

Interface avec Google Sheets :
  - Les fichiers JSON de sortie sont lus par les Apps Script
  - Les chemins d'entrée/sortie sont configurables

Usage en ligne de commande :
  python main_engine_v47.py --config project_config.json
                             --load load_profile_8760.csv
                             --weather meteo_8760.csv
                             --output ./results/

Unités : kW, kWh, °C, €
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from dispatch_multisource_v47 import MultiSourceConfig, MultiSourceDispatcher
from solar_thermal_v47 import synthetic_solar_8760
from ashp_v47 import synthetic_t_ext


# ─────────────────────────────────────────────────────────────────────────────
# 1. Chargement des données d'entrée
# ─────────────────────────────────────────────────────────────────────────────

def load_csv_8760(filepath: str,
                   col_timestamp: str = "timestamp",
                   col_heat: str = "heating_kw",
                   col_cool: str = "cooling_kw",
                   col_dhw: str = "dhw_kw") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Charge un fichier CSV 8760 h de profil de charge.

    Paramètres
    ----------
    filepath      : chemin vers le fichier CSV
    col_timestamp : nom colonne horodatage
    col_heat      : nom colonne chauffage [kW]
    col_cool      : nom colonne refroidissement [kW]
    col_dhw       : nom colonne ECS [kW]

    Retourne
    --------
    (q_heat_kw, q_cool_kw, q_dhw_kw) : tableaux (8760,)
    """
    q_heat = np.zeros(8760)
    q_cool = np.zeros(8760)
    q_dhw = np.zeros(8760)

    # Détection du séparateur
    with open(filepath, newline="", encoding="utf-8-sig") as f:
        sample = f.read(2048)
    sep = ";" if sample.count(";") > sample.count(",") else ","

    with open(filepath, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=sep)
        for i, row in enumerate(reader):
            if i >= 8760:
                break
            try:
                q_heat[i] = float(str(row.get(col_heat, "0")).replace(",", "."))
            except (ValueError, TypeError):
                pass
            try:
                q_cool[i] = float(str(row.get(col_cool, "0")).replace(",", "."))
            except (ValueError, TypeError):
                pass
            try:
                q_dhw[i] = float(str(row.get(col_dhw, "0")).replace(",", "."))
            except (ValueError, TypeError):
                pass

    return q_heat, q_cool, q_dhw


def _erbs_dhi_fraction(ghi_wm2: float, hour_of_year: int,
                        latitude_deg: float = 45.0) -> float:
    """
    Fraction diffuse DHI/GHI selon le modèle Erbs et al. (1982).
    Utilise l'indice de clarté kt = GHI / I₀ (irradiance extraterrestre).

    Référence : Erbs, Klein & Duffie (1982) Solar Energy, 28(4), 293–302.
    """
    if ghi_wm2 < 10.0:
        return 1.0  # nuit ou très faible rayonnement → tout diffus

    day_of_year = hour_of_year // 24 + 1
    hour_of_day = hour_of_year % 24

    # Déclinaison solaire (Cooper, 1969)
    delta = math.radians(23.45 * math.sin(math.radians(360.0 * (284 + day_of_year) / 365.0)))

    # Angle horaire (milieu de l'heure)
    omega = math.radians((hour_of_day + 0.5 - 12.0) * 15.0)

    lat_rad = math.radians(latitude_deg)
    cos_z = (math.sin(lat_rad) * math.sin(delta)
             + math.cos(lat_rad) * math.cos(delta) * math.cos(omega))

    if cos_z <= 0.05:
        return 1.0  # soleil bas → tout diffus

    I_sc = 1361.0  # constante solaire [W/m²]
    I0_h = I_sc * cos_z
    kt = min(max(ghi_wm2 / I0_h, 0.0), 1.0)

    # Modèle Erbs (3 régimes)
    if kt <= 0.22:
        fd = 1.0 - 0.09 * kt
    elif kt <= 0.80:
        fd = (0.9511 - 0.1604 * kt + 4.388 * kt ** 2
              - 16.638 * kt ** 3 + 12.336 * kt ** 4)
    else:
        fd = 0.165

    return max(0.0, min(1.0, fd))


def load_weather_8760(filepath: str,
                       latitude_deg: float = 45.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Charge un fichier CSV météo 8760 h.

    Colonnes auto-détectées (noms acceptés) :
      t_ext_c  : outdoor_c, t_ext_c, t_ext, temperature, temp, ta, outdoor
      ghi_wm2  : ghi_wm2, ghi, solar_wm2, solar, global_horizontal, g_h, irradiance
      dhi_wm2  : dhi_wm2, dhi, diffuse_horizontal, d_h  (optionnel)
      wind_ms  : wind_ms, wind, ws, vitesse_vent  (optionnel)

    Si DHI absent, fraction diffuse estimée par le modèle Erbs (1982).

    Retourne
    --------
    (t_ext, ghi, dhi, wind) : tableaux (8760,)
    """
    t_ext_c = np.zeros(8760)
    ghi = np.zeros(8760)
    dhi = np.zeros(8760)
    wind = np.zeros(8760)

    if not os.path.isfile(filepath):
        return t_ext_c, ghi, dhi, wind

    with open(filepath, newline="", encoding="utf-8-sig") as f:
        sample = f.read(2048)
    sep = ";" if sample.count(";") > sample.count(",") else ","

    # Correspondances de colonnes — format sample_8760.csv inclus
    aliases_t   = {"outdoor_c", "t_ext_c", "t_ext", "temperature", "temp", "ta",
                   "outdoor", "t_out", "t_outdoor"}
    aliases_ghi = {"ghi_wm2", "ghi", "solar_wm2", "solar", "global_horizontal",
                   "g_h", "irradiance", "rayonnement_wm2", "g_global"}
    aliases_dhi = {"dhi_wm2", "dhi", "diffuse_horizontal", "d_h",
                   "diffuse_wm2", "g_diffus"}
    aliases_w   = {"wind_ms", "wind", "ws", "vitesse_vent", "wind_speed"}

    dhi_found = False
    with open(filepath, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=sep)
        header_map = {k.lower().strip(): k for k in (reader.fieldnames or [])}

        col_t = next((header_map[a] for a in aliases_t if a in header_map), None)
        col_g = next((header_map[a] for a in aliases_ghi if a in header_map), None)
        col_d = next((header_map[a] for a in aliases_dhi if a in header_map), None)
        col_w = next((header_map[a] for a in aliases_w if a in header_map), None)
        dhi_found = col_d is not None

        for i, row in enumerate(reader):
            if i >= 8760:
                break

            def _v(col):
                if col is None:
                    return 0.0
                try:
                    return float(str(row.get(col, "0")).replace(",", "."))
                except (ValueError, TypeError):
                    return 0.0

            t_ext_c[i] = _v(col_t)
            ghi[i] = max(0.0, _v(col_g))
            dhi[i] = max(0.0, _v(col_d))
            wind[i] = max(0.0, _v(col_w))

    # Si DHI absent : estimation par modèle Erbs
    if not dhi_found:
        for i in range(8760):
            fd = _erbs_dhi_fraction(ghi[i], i, latitude_deg)
            dhi[i] = ghi[i] * fd

    return t_ext_c, ghi, dhi, wind


def load_combined_8760(filepath: str,
                        latitude_deg: float = 45.0,
                        dhw_constant_kw: float = 2.0
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                   np.ndarray, np.ndarray, np.ndarray]:
    """
    Charge un fichier CSV combiné (charge + météo) au format sample_8760.csv.

    Format attendu (séparateur `,` ou `;`, auto-détecté) :
      heating_kw, cooling_kw, outdoor_c, supply_c, solar_wm2

    La colonne `supply_c` n'est pas utilisée dans la simulation (elle décrit la T° de départ
    cible du bâtiment) — elle peut être utile pour le post-traitement.
    La colonne `dhw_kw` (ECS) n'est pas présente : une valeur constante est utilisée.

    Paramètres
    ----------
    filepath          : chemin vers le fichier CSV combiné
    latitude_deg      : latitude du site [°] (pour modèle Erbs DHI)
    dhw_constant_kw   : débit ECS constant si colonne absente [kW]

    Retourne
    --------
    (q_heat, q_cool, q_dhw, t_ext, ghi, dhi) : tableaux (8760,)
    """
    q_heat = np.zeros(8760)
    q_cool = np.zeros(8760)
    q_dhw  = np.zeros(8760)
    t_ext  = np.zeros(8760)
    ghi    = np.zeros(8760)
    dhi    = np.zeros(8760)

    if not os.path.isfile(filepath):
        print(f"  ⚠ Fichier combiné introuvable : {filepath}")
        return q_heat, q_cool, q_dhw, t_ext, ghi, dhi

    with open(filepath, newline="", encoding="utf-8-sig") as f:
        sample = f.read(2048)
    sep = ";" if sample.count(";") > sample.count(",") else ","

    # Colonnes charge
    aliases_heat = {"heating_kw", "heat_kw", "chauffage_kw", "q_heat"}
    aliases_cool = {"cooling_kw", "cool_kw", "refroidissement_kw", "q_cool"}
    aliases_dhw  = {"dhw_kw", "ecs_kw", "hot_water_kw", "q_dhw"}
    # Colonnes météo
    aliases_t    = {"outdoor_c", "t_ext_c", "t_ext", "outdoor", "temperature"}
    aliases_ghi  = {"solar_wm2", "ghi_wm2", "ghi", "solar", "irradiance"}
    aliases_dhi  = {"dhi_wm2", "dhi", "diffuse_horizontal", "d_h"}

    dhi_found = False
    with open(filepath, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=sep)
        hm = {k.lower().strip(): k for k in (reader.fieldnames or [])}

        col_heat = next((hm[a] for a in aliases_heat if a in hm), None)
        col_cool = next((hm[a] for a in aliases_cool if a in hm), None)
        col_dhw  = next((hm[a] for a in aliases_dhw  if a in hm), None)
        col_t    = next((hm[a] for a in aliases_t    if a in hm), None)
        col_g    = next((hm[a] for a in aliases_ghi  if a in hm), None)
        col_d    = next((hm[a] for a in aliases_dhi  if a in hm), None)
        dhi_found = col_d is not None

        def _v(row, col, default=0.0):
            if col is None:
                return default
            try:
                return float(str(row.get(col, str(default))).replace(",", "."))
            except (ValueError, TypeError):
                return default

        for i, row in enumerate(reader):
            if i >= 8760:
                break
            q_heat[i] = max(0.0, _v(row, col_heat))
            q_cool[i] = max(0.0, _v(row, col_cool))
            q_dhw[i]  = max(0.0, _v(row, col_dhw, dhw_constant_kw))
            t_ext[i]  = _v(row, col_t)
            ghi[i]    = max(0.0, _v(row, col_g))
            dhi[i]    = max(0.0, _v(row, col_d))

    # Remplissage ECS constante si colonne absente
    if col_dhw is None:
        q_dhw[:] = dhw_constant_kw

    # Estimation DHI si absente
    if not dhi_found:
        for i in range(8760):
            fd = _erbs_dhi_fraction(ghi[i], i, latitude_deg)
            dhi[i] = ghi[i] * fd

    print(f"  Fichier combiné chargé : {filepath}")
    print(f"    Chauffage annuel  : {np.sum(q_heat):>10,.0f} kWh")
    print(f"    Refroid. annuel   : {np.sum(q_cool):>10,.0f} kWh")
    print(f"    ECS annuel        : {np.sum(q_dhw):>10,.0f} kWh")
    print(f"    T_ext moy         : {np.mean(t_ext):>10.1f} °C")
    print(f"    GHI annuel        : {np.sum(ghi)/1000:>10,.0f} kWh/m²")
    print(f"    DHI {'(Erbs)' if not dhi_found else '(fichier)':8s}  : "
          f"{np.sum(dhi)/1000:>8,.0f} kWh/m²")

    return q_heat, q_cool, q_dhw, t_ext, ghi, dhi


def load_config(filepath: str) -> dict:
    """Charge le fichier de configuration JSON du projet."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Calculs économiques
# ─────────────────────────────────────────────────────────────────────────────

def compute_economics(annual: dict, config: dict) -> dict:
    """
    Calcule les indicateurs économiques annuels et pluriannuels.

    Paramètres
    ----------
    annual : indicateurs annuels du dispatch (kWh)
    config : section "economics" du fichier de configuration

    Retourne
    --------
    dict avec capex, opex, savings, TRS, VAN, TRI
    """
    eco = config.get("economics", {})
    src = config.get("sources", {})

    buy  = eco.get("electricity_buy_eur_kwh", 0.2516)
    sell = eco.get("electricity_sell_eur_kwh", 0.10)
    n_years = eco.get("study_years", 20)
    discount = eco.get("discount_rate", 0.04)
    elec_escalation = eco.get("electricity_escalation_rate", 0.03)

    # ── CAPEX estimatif ────────────────────────────────────────────────────
    capex = {}
    if src.get("gshp", {}).get("enabled"):
        gshp = src["gshp"]
        n_bh = len(gshp.get("boreholes", [])) or gshp.get("n_boreholes", 0)
        depth = gshp.get("depth_m", 120)
        capex["gshp_pac_eur"]    = gshp.get("capex_pac_eur", 0) or (
            gshp.get("pac", {}).get("rated_heat_kw", 20) * 800)
        capex["borehole_eur"]    = gshp.get("capex_borehole_eur", 0) or (
            n_bh * depth * 45)  # 45 €/m linéaire (indicatif)

    if src.get("ashp", {}).get("enabled"):
        ashp = src["ashp"]
        capex["ashp_eur"] = ashp.get("capex_eur", 0) or (
            ashp.get("rated_heat_kw", 20) * 600)

    if src.get("solar_thermal", {}).get("enabled"):
        st = src["solar_thermal"]
        capex["solar_thermal_eur"] = st.get("capex_eur", 0) or (
            st.get("area_m2", 20) * 600)

    if src.get("pvt", {}).get("enabled"):
        pvt = src["pvt"]
        capex["pvt_eur"] = pvt.get("capex_eur", 0) or (
            pvt.get("area_m2", 20) * 800)

    if src.get("pv", {}).get("enabled"):
        pv = src["pv"]
        capex["pv_eur"] = pv.get("capex_eur", 0) or (
            pv.get("peak_kw", 10) * 1200)

    if src.get("battery", {}).get("enabled"):
        bat = src["battery"]
        capex["battery_eur"] = bat.get("capex_eur", 0) or (
            bat.get("capacity_kwh", 10) * 600)

    if src.get("thermal_storage", {}).get("enabled"):
        ts = src["thermal_storage"]
        capex["thermal_storage_eur"] = ts.get("capex_eur", 0) or (
            ts.get("volume_m3", 1) * 800)

    capex_total = sum(capex.values())

    # ── Opex annuel ───────────────────────────────────────────────────────
    elec_cost = annual.get("grid_kwh", 0) * buy
    elec_revenue = annual.get("pv_injected_kwh", 0) * sell
    maintenance_eur = eco.get("annual_maintenance_eur",
                               capex_total * 0.01)  # 1% CAPEX par défaut

    opex_net = elec_cost - elec_revenue + maintenance_eur

    # ── Référence (sans installation) ────────────────────────────────────
    baseline_elec = eco.get("baseline_elec_kwh",
                             annual.get("heat_demand_kwh", 0)
                             + annual.get("cool_demand_kwh", 0))
    baseline_opex = baseline_elec * buy + maintenance_eur * 0.3

    annual_savings = baseline_opex - opex_net

    # ── Aides France (estimatif) ──────────────────────────────────────────
    aids = _estimate_aids(capex_total, annual, eco, src)

    net_investment = capex_total - aids["total_eur"]

    # ── TRS simple ────────────────────────────────────────────────────────
    trs = net_investment / max(annual_savings, 1.0)

    # ── VAN (Valeur Actuelle Nette) ───────────────────────────────────────
    van = -net_investment
    for y in range(1, n_years + 1):
        # Escalade des prix de l'énergie
        elec_factor = (1 + elec_escalation) ** y
        savings_y = annual_savings * elec_factor
        van += savings_y / (1 + discount) ** y

    # ── TRI (Taux de Rentabilité Interne) — Newton-Raphson ───────────────
    tri = _compute_tri(net_investment, annual_savings, n_years, elec_escalation)

    return {
        "capex": capex,
        "capex_total_eur":      round(capex_total, 0),
        "opex_net_eur":         round(opex_net, 0),
        "baseline_opex_eur":    round(baseline_opex, 0),
        "annual_savings_eur":   round(annual_savings, 0),
        "aids":                 aids,
        "net_investment_eur":   round(net_investment, 0),
        "trs_years":            round(trs, 1),
        "van_eur":              round(van, 0),
        "tri_pct":              round(tri * 100, 1) if tri else None,
        "study_years":          n_years,
        "electricity_buy_eur_kwh": buy,
        "status":               "estimatif_non_confirme",
    }


def _estimate_aids(capex_total: float, annual: dict, eco: dict, src: dict) -> dict:
    """Estimation indicative des aides françaises (MaPrimeRénov', CEE)."""
    aids = {}

    # MaPrimeRénov' géothermie (barème 2026 indicatif)
    income_cat = eco.get("income_category", "intermediaire")
    rates = {
        "tres_modeste": 0.50,
        "modeste":      0.40,
        "intermediaire":0.30,
        "superieure":   0.15,
    }
    rate_mpr = rates.get(income_cat, 0.30)

    if src.get("gshp", {}).get("enabled") or src.get("ashp", {}).get("enabled"):
        capex_pac = sum(v for k, v in {
            "gshp_pac_eur": 0, "borehole_eur": 0, "ashp_eur": 0
        }.items())  # sera recalculé depuis capex
        eligible = min(capex_total * 0.70, 20000)  # plafond indicatif
        aids["maprimerenov_eur"] = round(eligible * rate_mpr, 0)
    else:
        aids["maprimerenov_eur"] = 0

    # CEE (Certificats d'Économies d'Énergie) — estimation
    kwh_saved = annual.get("heat_demand_kwh", 0) * 0.60  # hypothèse 60% d'économie
    cee_price = eco.get("cee_price_eur_kwh_cumac", 0.006)
    aids["cee_eur"] = round(kwh_saved * 20 * cee_price, 0)  # 20 kWh_cumac / kWh_saved (aprox.)

    # TVA réduite (5.5%) — différentiel par rapport à 20%
    aids["tva_economie_eur"] = round(capex_total * (0.20 - 0.055), 0)

    aids["total_eur"] = round(sum(v for k, v in aids.items() if k != "total_eur"), 0)
    aids["status"] = "estimatif_non_confirme"
    aids["note"] = ("Montants indicatifs, à confirmer avec un conseiller agréé "
                    "et les organismes compétents (ANAH, obligataires CEE).")
    return aids


def _compute_tri(investment: float, annual_savings: float,
                  n_years: int, escalation: float) -> Optional[float]:
    """Calcule le TRI par dichotomie."""
    def npv_at(r):
        return -investment + sum(
            annual_savings * (1 + escalation) ** y / (1 + r) ** y
            for y in range(1, n_years + 1)
        )
    # Bracketing
    if npv_at(0.001) < 0:
        return None  # TRI < 0 (investissement non rentable sur la période)
    lo, hi = 0.001, 0.50
    for _ in range(50):
        mid = (lo + hi) / 2
        if npv_at(mid) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-5:
            break
    return (lo + hi) / 2


# ─────────────────────────────────────────────────────────────────────────────
# 3. Export des résultats
# ─────────────────────────────────────────────────────────────────────────────

def export_hourly_csv(hourly: dict, output_path: str, n: int = 8760):
    """Exporte le tableau horaire en CSV (8760 lignes)."""
    # Colonnes à exporter (toutes les arrays numpy du dict)
    cols = {k: v for k, v in hourly.items() if isinstance(v, np.ndarray)}
    fieldnames = ["hour"] + list(cols.keys())

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for h in range(n):
            row = {"hour": h}
            for k, arr in cols.items():
                row[k] = round(float(arr[h]), 4) if h < len(arr) else 0.0
            writer.writerow(row)

    print(f"  CSV horaire exporté : {output_path}")


def export_json_summary(annual: dict, economics: dict, config: dict,
                         output_path: str):
    """Exporte le JSON de synthèse (KPI annuels + économique)."""
    summary = {
        "version": "V47",
        "generated_at": datetime.now().isoformat(),
        "project": config.get("project", {}),
        "annual": annual,
        "economics": economics,
        "status": "estimatif",
        "note": ("Ces résultats sont des estimations de dimensionnement "
                 "et ne constituent pas un résultat contractuel ou réglementaire.")
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=_json_default)
    print(f"  JSON synthèse exporté : {output_path}")


def _json_default(obj):
    """Sérialisation JSON pour les types non natifs."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Non sérialisable : {type(obj)}")


def export_google_sheets_csv(annual: dict, economics: dict,
                               output_path: str):
    """
    Exporte un CSV compact lisible par Google Apps Script.
    Format : clé, valeur, unité, statut
    Permet l'import direct dans l'onglet Rentabilité de Google Sheets.
    """
    rows = [
        ("heat_demand_kwh",   annual.get("heat_demand_kwh", 0),   "kWh", "estimatif"),
        ("cool_demand_kwh",   annual.get("cool_demand_kwh", 0),   "kWh", "estimatif"),
        ("solar_th_kwh",      annual.get("solar_th_kwh", 0),      "kWh", "estimatif"),
        ("pvt_th_kwh",        annual.get("pvt_th_kwh", 0),        "kWh", "estimatif"),
        ("gshp_heat_kwh",     annual.get("gshp_heat_kwh", 0),     "kWh", "estimatif"),
        ("ashp_heat_kwh",     annual.get("ashp_heat_kwh", 0),     "kWh", "estimatif"),
        ("backup_kwh",        annual.get("backup_kwh", 0),        "kWh", "estimatif"),
        ("unmet_heat_kwh",    annual.get("unmet_heat_kwh", 0),    "kWh", "estimatif"),
        ("pv_yield_kwh",      annual.get("pv_yield_kwh", 0),      "kWh", "estimatif"),
        ("pv_self_kwh",       annual.get("pv_self_kwh", 0),       "kWh", "estimatif"),
        ("pv_injected_kwh",   annual.get("pv_injected_kwh", 0),   "kWh", "estimatif"),
        ("grid_kwh",          annual.get("grid_kwh", 0),          "kWh", "estimatif"),
        ("scop_system",       annual.get("scop_system", 0),       "—",   "estimatif"),
        ("capex_total_eur",   economics.get("capex_total_eur", 0),"EUR", "estimatif"),
        ("net_investment_eur",economics.get("net_investment_eur",0),"EUR","estimatif"),
        ("annual_savings_eur",economics.get("annual_savings_eur",0),"EUR","estimatif"),
        ("aids_total_eur",    economics.get("aids", {}).get("total_eur", 0), "EUR", "estimatif"),
        ("trs_years",         economics.get("trs_years", 0),      "ans", "estimatif"),
        ("van_eur",           economics.get("van_eur", 0),        "EUR", "estimatif"),
        ("tri_pct",           economics.get("tri_pct") or 0,      "%",   "estimatif"),
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["indicateur", "valeur", "unite", "statut"])
        for row in rows:
            writer.writerow(row)

    print(f"  CSV Google Sheets exporté : {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Orchestrateur principal
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(config_path: Optional[str] = None,
                   load_path: Optional[str] = None,
                   weather_path: Optional[str] = None,
                   combined_path: Optional[str] = None,
                   output_dir: str = "./results",
                   config_dict: Optional[dict] = None,
                   year: int = 0) -> dict:
    """
    Lance la simulation complète.

    Peut être appelé depuis la ligne de commande (via main) ou
    depuis un autre script Python / test unitaire.

    Paramètres
    ----------
    config_path   : chemin vers project_config.json (optionnel si config_dict)
    load_path     : CSV profil de charge 8760 h (colonnes : heating_kw, cooling_kw, dhw_kw)
    weather_path  : CSV météo 8760 h (colonnes : outdoor_c, solar_wm2 / ghi_wm2, dhi_wm2)
    combined_path : CSV combiné charge+météo format sample_8760.csv
                    (colonnes : heating_kw, cooling_kw, outdoor_c, supply_c, solar_wm2)
    output_dir    : dossier de sortie
    config_dict   : configuration directement en dict (priorité sur config_path)
    year          : année de fonctionnement

    Retourne
    --------
    dict complet (dispatch + économie)
    """
    print("=" * 60)
    print("  DIMENSIONNEMENT GÉO V47 — Moteur de simulation")
    print("=" * 60)

    # Chargement config
    if config_dict is not None:
        config = config_dict
    elif config_path:
        config = load_config(config_path)
        print(f"  Config : {config_path}")
    else:
        raise ValueError("Fournir config_path ou config_dict.")

    project = config.get("project", {})
    print(f"  Projet : {project.get('name', 'N/A')}")
    print(f"  Site   : {project.get('location', 'N/A')}")

    lat = project.get("latitude", 45.75)
    combined_path = config.get("combined_path")  # fichier type sample_8760.csv

    # ── Chargement profil de charge ───────────────────────────────────────
    if combined_path and os.path.isfile(combined_path):
        # Fichier combiné charge+météo (ex : sample_8760.csv)
        q_heat, q_cool, q_dhw, t_ext, ghi, dhi = load_combined_8760(
            combined_path,
            latitude_deg=lat,
            dhw_constant_kw=config.get("dhw_constant_kw", 2.0),
        )
    else:
        # Fichiers séparés charge / météo
        if load_path and os.path.isfile(load_path):
            q_heat, q_cool, q_dhw = load_csv_8760(load_path)
            print(f"  Profil de charge chargé : {load_path}")
            print(f"    Chauffage annuel : {np.sum(q_heat):,.0f} kWh")
            print(f"    Refroid. annuel  : {np.sum(q_cool):,.0f} kWh")
            print(f"    ECS annuel       : {np.sum(q_dhw):,.0f} kWh")
        else:
            print("  ⚠ Profil de charge synthétique (fichier non fourni)")
            t_ext_synth = synthetic_t_ext(t_annual_mean_c=11.5)
            q_heat = np.maximum(0.0, 15.0 - 0.8 * t_ext_synth)
            q_cool = np.zeros(8760)
            q_cool[3500:5500] = 5.0
            q_dhw = np.full(8760, 2.0)

        # ── Chargement météo ──────────────────────────────────────────────
        if weather_path and os.path.isfile(weather_path):
            t_ext, ghi, dhi, _ = load_weather_8760(weather_path, latitude_deg=lat)
            print(f"  Météo chargée : {weather_path}")
        else:
            print("  ⚠ Données météo synthétiques (fichier non fourni)")
            t_ext = synthetic_t_ext(t_annual_mean_c=11.5)
            ghi, dhi = synthetic_solar_8760(latitude_deg=lat, annual_ghi_kwh_m2=1300.0)

    # Charge électrique auxiliaire (non-PAC)
    p_elec_aux = np.full(8760, config.get("p_elec_aux_kw", 1.5))

    # Construction configuration système
    cfg = MultiSourceConfig.from_dict(config.get("sources", {}))
    cfg.backup_elec_kw = config.get("backup_elec_kw", 10.0)

    # Dispatch multi-sources
    print("\n  Dispatch 8760 h en cours...")
    dispatcher = MultiSourceDispatcher(cfg)
    dispatch_res = dispatcher.dispatch_8760(
        ghi, dhi, t_ext, q_heat, q_cool, q_dhw, p_elec_aux, year=year
    )
    print(f"  Sources actives : {dispatch_res['sources_active']}")

    ann = dispatch_res["annual"]
    print(f"\n  Résultats annuels :")
    print(f"    Demande chaleur     : {ann['heat_demand_kwh']:>10,.0f} kWh")
    print(f"    Solaire thermique   : {ann['solar_th_kwh']:>10,.0f} kWh")
    print(f"    GSHP chauffage      : {ann['gshp_heat_kwh']:>10,.0f} kWh")
    print(f"    ASHP chauffage      : {ann['ashp_heat_kwh']:>10,.0f} kWh")
    print(f"    PV production       : {ann['pv_yield_kwh']:>10,.0f} kWh")
    print(f"    SCOP système        : {ann['scop_system']:>10.2f}")
    print(f"    Non couvert chauf.  : {ann['unmet_heat_kwh']:>10,.0f} kWh")

    # Économie
    eco = compute_economics(ann, config)
    print(f"\n  Économie :")
    print(f"    CAPEX total         : {eco['capex_total_eur']:>10,.0f} €")
    print(f"    Aides estimées      : {eco['aids']['total_eur']:>10,.0f} €")
    print(f"    Investissement net  : {eco['net_investment_eur']:>10,.0f} €")
    print(f"    Économies/an        : {eco['annual_savings_eur']:>10,.0f} €")
    print(f"    TRS                 : {eco['trs_years']:>10.1f} ans")
    print(f"    VAN {eco['study_years']} ans             : {eco['van_eur']:>10,.0f} €")
    if eco["tri_pct"]:
        print(f"    TRI                 : {eco['tri_pct']:>10.1f} %")
    print(f"\n  ⚠ Résultats ESTIMATIFS — non contractuels")

    # Export
    os.makedirs(output_dir, exist_ok=True)
    csv_hourly   = os.path.join(output_dir, "results_hourly_v47.csv")
    json_summary = os.path.join(output_dir, "results_summary_v47.json")
    csv_sheets   = os.path.join(output_dir, "results_sheets_v47.csv")

    export_hourly_csv(dispatch_res["hourly"], csv_hourly)
    export_json_summary(ann, eco, config, json_summary)
    export_google_sheets_csv(ann, eco, csv_sheets)

    print("\n  Simulation terminée.")
    print("=" * 60)

    return {
        "dispatch": dispatch_res,
        "annual": ann,
        "economics": eco,
        "output_files": {
            "hourly_csv": csv_hourly,
            "summary_json": json_summary,
            "sheets_csv": csv_sheets,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Interface ligne de commande
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Dimensionnement Géo V47 — Moteur de simulation multi-sources"
    )
    parser.add_argument("--config",   required=False, help="Fichier project_config.json")
    parser.add_argument("--load",     required=False, help="CSV profil de charge 8760 h")
    parser.add_argument("--weather",  required=False, help="CSV météo 8760 h")
    parser.add_argument("--combined", required=False,
                        help="CSV combiné charge+météo (format sample_8760.csv)")
    parser.add_argument("--output",   default="./results", help="Dossier de sortie")
    parser.add_argument("--year",     type=int, default=0, help="Année de fonctionnement")
    parser.add_argument("--demo",     action="store_true",
                        help="Mode démonstration (sans fichiers d'entrée)")
    args = parser.parse_args()

    if args.demo or not args.config:
        # Configuration de démonstration complète
        demo_config = {
            "project": {
                "name": "Bâtiment démo V47",
                "location": "Lyon, France",
                "latitude": 45.75,
                "longitude": 4.85,
            },
            "sources": {
                "gshp": {
                    "enabled": False,  # désactivé pour test rapide
                },
                "ashp": {
                    "enabled": True,
                    "rated_heat_kw": 20.0,
                    "cop_b7w35": 4.2,
                    "rated_cool_kw": 12.0,
                    "t_ext_heat_min_c": -20.0,
                },
                "solar_thermal": {
                    "enabled": True,
                    "area_m2": 25.0,
                    "tilt_deg": 35.0,
                    "latitude_deg": 45.75,
                },
                "pv": {
                    "enabled": True,
                    "peak_kw": 10.0,
                    "tilt_deg": 30.0,
                    "latitude_deg": 45.75,
                },
                "battery": {
                    "enabled": True,
                    "capacity_kwh": 15.0,
                    "power_charge_kw": 5.0,
                    "power_discharge_kw": 5.0,
                },
            },
            "backup_elec_kw": 5.0,
            "p_elec_aux_kw": 1.5,
            "economics": {
                "electricity_buy_eur_kwh": 0.2516,
                "electricity_sell_eur_kwh": 0.10,
                "study_years": 20,
                "discount_rate": 0.04,
                "electricity_escalation_rate": 0.03,
                "income_category": "intermediaire",
                "annual_maintenance_eur": 800,
            },
        }
        results = run_simulation(
            config_dict=demo_config,
            output_dir=args.output,
            year=args.year,
        )
    else:
        results = run_simulation(
            config_path=args.config,
            load_path=args.load,
            weather_path=args.weather,
            combined_path=args.combined,
            output_dir=args.output,
            year=args.year,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
