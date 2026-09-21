"""
DiMii V47 — Application Web de Dimensionnement Géothermique
Déployable sur Streamlit Community Cloud
"""

import os
import sys
import json
import tempfile
import io
from pathlib import Path

import numpy as np
import streamlit as st

# ─── Configuration de la page ────────────────────────────────────────────────
st.set_page_config(
    page_title="DiMii V47 — Dimensionnement Géo",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ajout du répertoire courant au path pour les imports
sys.path.insert(0, str(Path(__file__).parent))

# ─── Constantes ──────────────────────────────────────────────────────────────
CAPEX_KEYS_TO_STRIP = {"capex_eur", "capex_borehole_eur", "capex_pac_eur", "capex_total_eur"}

MONTHS = ["Jan","Fév","Mar","Avr","Mai","Jun","Jul","Aoû","Sep","Oct","Nov","Déc"]
MONTH_HOURS = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]


# ─── Valeurs par défaut ───────────────────────────────────────────────────────
def _default_project():
    return {
        "name": "Mon Projet",
        "location": "Lyon, France",
        "latitude": 45.75,
        "longitude": 4.85,
        "building_type": "Tertiaire",
        "floor_area_m2": 1000.0,
        "dpe_class": "C",
    }

def _default_sources():
    return {
        "gshp": {
            "enabled": True,
            "n_boreholes": 4,
            "spacing_m": 8.0,
            "depth_m": 120.0,
            "ground": {
                "lambda_wpmk": 2.0,
                "diffusivity_m2ps": 1.0e-6,
                "t_undisturbed_c": 12.0,
                "t_annual_amplitude_c": 3.0,
            },
            "pac": {
                "rated_heat_kw": 20.0,
                "rated_cool_kw": 10.0,
                "cop_b0w35": 4.5,
                "eer_b10w18": 5.0,
                "eta_carnot_heat": 0.50,
                "eta_carnot_cool": 0.50,
                "t_supply_heat_c": 35.0,
                "t_supply_cool_c": 7.0,
            },
            "capex_eur": 30000.0,  # UI seulement
        },
        "ashp": {
            "enabled": False,
            "rated_heat_kw": 20.0,
            "rated_cool_kw": 15.0,
            "cop_b7w35": 4.1,
            "eer_b35w7": 3.5,
            "t_supply_heat_c": 35.0,
            "t_supply_cool_c": 7.0,
            "t_ext_heat_min_c": -20.0,
            "t_ext_heat_max_c": 20.0,
            "t_ext_cool_min_c": 10.0,
            "t_ext_cool_max_c": 46.0,
            "defrost_enabled": True,
            "inverter": True,
            "capex_eur": 12000.0,  # UI seulement
        },
        "solar_thermal": {
            "enabled": True,
            "area_m2": 20.0,
            "eta0": 0.74,
            "a1_wpm2k": 3.5,
            "a2_wpm2k2": 0.015,
            "t_in_c": 45.0,
            "t_out_c": 65.0,
            "tilt_deg": 35.0,
            "azimuth_deg": 0.0,
            "latitude_deg": 45.75,
            "longitude_deg": 4.85,
            "capex_eur": 8000.0,  # UI seulement
        },
        "pv": {
            "enabled": False,
            "peak_kw": 10.0,
            "tilt_deg": 30.0,
            "azimuth_deg": 0.0,
            "latitude_deg": 45.75,
            "longitude_deg": 4.85,
            "gamma_pv": -0.0040,
            "noct_c": 45.0,
            "eta_inverter": 0.97,
            "capex_eur": 12000.0,  # UI seulement
        },
        "battery": {
            "enabled": False,
            "capacity_kwh": 10.0,
            "soc_min_kwh": 1.0,
            "soc_max_kwh": 9.5,
            "soc_initial_kwh": 5.0,
            "power_charge_kw": 5.0,
            "power_discharge_kw": 5.0,
            "eta_charge": 0.95,
            "eta_discharge": 0.95,
            "capex_eur_kwh": 600.0,
        },
    }

def _default_economics():
    return {
        "electricity_buy_eur_kwh": 0.2516,
        "electricity_sell_eur_kwh": 0.10,
        "gas_eur_kwh": 0.09,
        "elec_escalation_rate": 0.03,
        "discount_rate": 0.04,
        "study_years": 20,
        "income_category": "intermediaire",
        "baseline_heating_system": "gaz_condensation",
        "baseline_cop": 0.95,
    }


# ─── Initialisation session state ────────────────────────────────────────────
def _init_state():
    if "page" not in st.session_state:
        st.session_state.page = "🏠 Projet"
    if "project" not in st.session_state:
        st.session_state.project = _default_project()
    if "sources" not in st.session_state:
        st.session_state.sources = _default_sources()
    if "economics" not in st.session_state:
        st.session_state.economics = _default_economics()
    if "simulation_results" not in st.session_state:
        st.session_state.simulation_results = None
    if "csv_path" not in st.session_state:
        st.session_state.csv_path = None
    if "simulation_ran" not in st.session_state:
        st.session_state.simulation_ran = False


# ─── Construction du config dict ─────────────────────────────────────────────
def _build_config_dict():
    """Construit le config dict pour run_simulation(), en supprimant les clés capex UI."""
    project = st.session_state.project.copy()
    src_raw = st.session_state.sources
    eco = st.session_state.economics.copy()

    # Nettoyage : supprimer les clés capex UI qui ne sont pas des champs dataclass
    clean_sources = {}
    for src_name, src_val in src_raw.items():
        if isinstance(src_val, dict):
            clean_sources[src_name] = {
                k: v for k, v in src_val.items()
                if k not in CAPEX_KEYS_TO_STRIP
            }
        else:
            clean_sources[src_name] = src_val

    # Synchroniser latitude/longitude depuis project vers sources solaires
    lat = project.get("latitude", 45.75)
    lon = project.get("longitude", 4.85)
    for src_name in ["solar_thermal", "pv"]:
        if src_name in clean_sources and isinstance(clean_sources[src_name], dict):
            clean_sources[src_name]["latitude_deg"] = lat
            clean_sources[src_name]["longitude_deg"] = lon
    if "gshp" in clean_sources:
        # Injecter la latitude dans gshp si utilisée par le moteur
        clean_sources["gshp"]["latitude"] = lat

    config = {
        "project": project,
        "sources": clean_sources,
        "backup_elec_kw": 10.0,
        "p_elec_aux_kw": 1.5,
        "economics": eco,
    }
    if st.session_state.csv_path and os.path.isfile(st.session_state.csv_path):
        config["combined_path"] = st.session_state.csv_path

    return config


# ─── Calcul capex total (pour affichage UI) ───────────────────────────────────
def _calc_capex_display():
    src = st.session_state.sources
    capex = {}
    if src.get("gshp", {}).get("enabled"):
        capex["Géothermie (sondes+PAC)"] = src["gshp"].get("capex_eur", 0)
    if src.get("ashp", {}).get("enabled"):
        capex["PAC aérothermique"] = src["ashp"].get("capex_eur", 0)
    if src.get("solar_thermal", {}).get("enabled"):
        capex["Solaire thermique"] = src["solar_thermal"].get("capex_eur", 0)
    if src.get("pv", {}).get("enabled"):
        capex["Photovoltaïque"] = src["pv"].get("capex_eur", 0)
    if src.get("battery", {}).get("enabled"):
        bat = src["battery"]
        capex["Batterie"] = bat.get("capacity_kwh", 10) * bat.get("capex_eur_kwh", 600)
    return capex


# ─── Agrégation mensuelle depuis horaire ────────────────────────────────────
def _monthly_sum(hourly_array):
    """Agrège un tableau 8760h en 12 valeurs mensuelles."""
    result = []
    idx = 0
    for h in MONTH_HOURS:
        result.append(float(np.sum(hourly_array[idx:idx+h])))
        idx += h
    return result


# ─── PAGE 1 : Projet ─────────────────────────────────────────────────────────
def page_projet():
    st.title("🏠 Description du projet")
    st.caption("Renseignez les informations générales du projet de dimensionnement.")

    proj = st.session_state.project

    col1, col2 = st.columns(2)
    with col1:
        proj["name"] = st.text_input("Nom du projet", value=proj["name"])
        proj["location"] = st.text_input("Localisation", value=proj["location"])
        proj["building_type"] = st.selectbox(
            "Type de bâtiment",
            ["Résidentiel collectif", "Maison individuelle", "Tertiaire", "Industrie", "Hôpital / ERP"],
            index=["Résidentiel collectif","Maison individuelle","Tertiaire","Industrie","Hôpital / ERP"].index(
                proj.get("building_type","Tertiaire")) if proj.get("building_type","Tertiaire") in
                ["Résidentiel collectif","Maison individuelle","Tertiaire","Industrie","Hôpital / ERP"] else 2
        )
    with col2:
        proj["latitude"] = st.number_input("Latitude [°N]", value=float(proj["latitude"]), min_value=41.0, max_value=51.5, step=0.1, format="%.2f")
        proj["longitude"] = st.number_input("Longitude [°E]", value=float(proj["longitude"]), min_value=-5.0, max_value=9.0, step=0.1, format="%.2f")
        proj["floor_area_m2"] = st.number_input("Surface plancher [m²]", value=float(proj["floor_area_m2"]), min_value=50.0, max_value=100000.0, step=50.0)

    col3, col4 = st.columns(2)
    with col3:
        proj["dpe_class"] = st.selectbox("Classe DPE", ["A","B","C","D","E","F","G"],
            index=["A","B","C","D","E","F","G"].index(proj.get("dpe_class","C")))
    with col4:
        dhw_kw = st.number_input("Puissance ECS constante [kW]", value=proj.get("dhw_constant_kw", 2.0), min_value=0.0, max_value=50.0, step=0.5)
        proj["dhw_constant_kw"] = dhw_kw

    st.session_state.project = proj

    st.info("📍 Les coordonnées GPS sont utilisées pour le calcul solaire et la géologie de référence.")

    # Carte indicative
    if 41.0 <= proj["latitude"] <= 51.5 and -5.0 <= proj["longitude"] <= 9.0:
        import pandas as pd
        df_loc = pd.DataFrame({"lat": [proj["latitude"]], "lon": [proj["longitude"]]})
        st.map(df_loc, zoom=7)


# ─── PAGE 2 : Données ────────────────────────────────────────────────────────
def page_donnees():
    st.title("📂 Données de charge et météo")
    st.caption("Chargez un fichier CSV 8760 h ou utilisez les profils synthétiques.")

    tab1, tab2 = st.tabs(["📤 Import CSV", "🧪 Profils synthétiques"])

    with tab1:
        st.markdown("""
**Format attendu** (`sample_8760.csv`) — 8760 lignes, colonnes séparées par virgule :

| heating_kw | cooling_kw | outdoor_c | supply_c | solar_wm2 |
|---|---|---|---|---|
| 12.5 | 0.0 | 5.2 | 45.0 | 80.0 |
| … | … | … | … | … |
""")
        uploaded = st.file_uploader("Chargez votre fichier CSV 8760 h", type=["csv"])
        if uploaded is not None:
            try:
                import pandas as pd
                df = pd.read_csv(uploaded)
                if len(df) < 8700:
                    st.error(f"❌ Le fichier contient {len(df)} lignes (minimum 8700 requis).")
                else:
                    # Sauvegarde en fichier temporaire
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="wb")
                    uploaded.seek(0)
                    tmp.write(uploaded.read())
                    tmp.close()
                    st.session_state.csv_path = tmp.name
                    st.session_state.simulation_ran = False
                    st.success(f"✅ Fichier chargé : {len(df)} lignes, {len(df.columns)} colonnes")
                    st.dataframe(df.head(5), use_container_width=True)

                    # Statistiques rapides
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if "heating_kw" in df.columns:
                            st.metric("Chauffage annuel", f"{df['heating_kw'].sum()/1000:.0f} MWh")
                    with col2:
                        if "cooling_kw" in df.columns:
                            st.metric("Refroidissement annuel", f"{df['cooling_kw'].sum()/1000:.0f} MWh")
                    with col3:
                        if "outdoor_c" in df.columns:
                            st.metric("T° moy. ext.", f"{df['outdoor_c'].mean():.1f} °C")
            except Exception as e:
                st.error(f"Erreur lecture CSV : {e}")

        elif st.session_state.csv_path:
            st.success(f"✅ Fichier chargé précédemment : `{os.path.basename(st.session_state.csv_path)}`")
            if st.button("🗑 Supprimer le fichier (utiliser synthétique)"):
                st.session_state.csv_path = None
                st.session_state.simulation_ran = False
                st.rerun()

    with tab2:
        st.info("""
**Mode synthétique** — Le moteur génère automatiquement des profils horaires
basés sur la latitude et la classe DPE si aucun CSV n'est fourni.

Paramètres utilisés :
- Météo : profil sinusoïdal (T° moy. annuelle selon latitude)
- Charge : proportionnelle à la T° extérieure
- GHI annuel : ~1 200–1 400 kWh/m² selon la latitude
""")
        lat = st.session_state.project.get("latitude", 45.75)
        t_mean = max(8.0, min(15.0, 20.0 - 0.3 * lat))
        ghi_est = max(900.0, min(1600.0, 2500.0 - 20.0 * lat))
        col1, col2 = st.columns(2)
        with col1:
            st.metric("T° annuelle moyenne estimée", f"{t_mean:.1f} °C")
        with col2:
            st.metric("GHI annuel estimé", f"{ghi_est:.0f} kWh/m²")
        if not st.session_state.csv_path:
            st.success("✅ Mode synthétique activé (aucun CSV chargé)")
        else:
            st.warning("⚠ Un CSV est chargé — le mode synthétique ne sera pas utilisé")


# ─── PAGE 3 : Système ────────────────────────────────────────────────────────
def page_systeme():
    st.title("⚙️ Configuration du système")
    st.caption("Activez et paramétrez chaque source d'énergie.")

    src = st.session_state.sources

    # ── GSHP ──────────────────────────────────────────────────────────────
    with st.expander("🌍 PAC géothermique sur sondes (GSHP)", expanded=src["gshp"]["enabled"]):
        src["gshp"]["enabled"] = st.checkbox("Activer GSHP", value=src["gshp"]["enabled"], key="chk_gshp")
        if src["gshp"]["enabled"]:
            col1, col2, col3 = st.columns(3)
            with col1:
                src["gshp"]["n_boreholes"] = st.number_input("Nombre de sondes", value=int(src["gshp"]["n_boreholes"]), min_value=1, max_value=30, key="gshp_nb")
                src["gshp"]["depth_m"] = st.number_input("Profondeur [m]", value=float(src["gshp"]["depth_m"]), min_value=50.0, max_value=300.0, step=10.0, key="gshp_depth")
            with col2:
                src["gshp"]["spacing_m"] = st.number_input("Espacement [m]", value=float(src["gshp"]["spacing_m"]), min_value=4.0, max_value=20.0, step=1.0, key="gshp_sp")
                src["gshp"]["pac"]["rated_heat_kw"] = st.number_input("Puissance chauf. [kW]", value=float(src["gshp"]["pac"]["rated_heat_kw"]), min_value=5.0, max_value=500.0, step=5.0, key="gshp_heat")
            with col3:
                src["gshp"]["pac"]["cop_b0w35"] = st.number_input("COP B0/W35", value=float(src["gshp"]["pac"]["cop_b0w35"]), min_value=2.0, max_value=7.0, step=0.1, format="%.1f", key="gshp_cop")
                src["gshp"]["capex_eur"] = st.number_input("CAPEX estimé [€]", value=float(src["gshp"]["capex_eur"]), min_value=0.0, step=1000.0, key="gshp_capex")
            col4, col5 = st.columns(2)
            with col4:
                src["gshp"]["ground"]["lambda_wpmk"] = st.slider("Conductivité sol [W/(m·K)]", 1.0, 4.0, float(src["gshp"]["ground"]["lambda_wpmk"]), 0.1, key="gshp_lambda")
            with col5:
                src["gshp"]["ground"]["t_undisturbed_c"] = st.slider("T° sol non-perturbée [°C]", 8.0, 18.0, float(src["gshp"]["ground"]["t_undisturbed_c"]), 0.5, key="gshp_t0")

    # ── ASHP ──────────────────────────────────────────────────────────────
    with st.expander("💨 PAC aérothermique (ASHP)", expanded=src["ashp"]["enabled"]):
        src["ashp"]["enabled"] = st.checkbox("Activer ASHP", value=src["ashp"]["enabled"], key="chk_ashp")
        if src["ashp"]["enabled"]:
            col1, col2, col3 = st.columns(3)
            with col1:
                src["ashp"]["rated_heat_kw"] = st.number_input("Puissance chauf. [kW]", value=float(src["ashp"]["rated_heat_kw"]), min_value=2.0, max_value=200.0, step=2.0, key="ashp_heat")
                src["ashp"]["rated_cool_kw"] = st.number_input("Puissance refroid. [kW]", value=float(src["ashp"]["rated_cool_kw"]), min_value=2.0, max_value=200.0, step=2.0, key="ashp_cool")
            with col2:
                src["ashp"]["cop_b7w35"] = st.number_input("COP B7/W35", value=float(src["ashp"]["cop_b7w35"]), min_value=2.0, max_value=6.0, step=0.1, format="%.1f", key="ashp_cop")
                src["ashp"]["eer_b35w7"] = st.number_input("EER A35/W7", value=float(src["ashp"]["eer_b35w7"]), min_value=1.5, max_value=8.0, step=0.1, format="%.1f", key="ashp_eer")
            with col3:
                src["ashp"]["capex_eur"] = st.number_input("CAPEX estimé [€]", value=float(src["ashp"]["capex_eur"]), min_value=0.0, step=500.0, key="ashp_capex")
                src["ashp"]["inverter"] = st.checkbox("Inverseur (modulation)", value=src["ashp"]["inverter"], key="ashp_inv")

    # ── Solaire thermique ─────────────────────────────────────────────────
    with st.expander("☀️ Capteurs solaires thermiques", expanded=src["solar_thermal"]["enabled"]):
        src["solar_thermal"]["enabled"] = st.checkbox("Activer solaire thermique", value=src["solar_thermal"]["enabled"], key="chk_st")
        if src["solar_thermal"]["enabled"]:
            col1, col2, col3 = st.columns(3)
            with col1:
                src["solar_thermal"]["area_m2"] = st.number_input("Surface capteurs [m²]", value=float(src["solar_thermal"]["area_m2"]), min_value=2.0, max_value=500.0, step=2.0, key="st_area")
                src["solar_thermal"]["tilt_deg"] = st.slider("Inclinaison [°]", 0, 90, int(src["solar_thermal"]["tilt_deg"]), key="st_tilt")
            with col2:
                src["solar_thermal"]["eta0"] = st.number_input("η₀ (rendement optique)", value=float(src["solar_thermal"]["eta0"]), min_value=0.40, max_value=0.95, step=0.01, format="%.2f", key="st_eta0")
                src["solar_thermal"]["a1_wpm2k"] = st.number_input("a₁ [W/(m²·K)]", value=float(src["solar_thermal"]["a1_wpm2k"]), min_value=0.5, max_value=10.0, step=0.1, format="%.1f", key="st_a1")
            with col3:
                src["solar_thermal"]["capex_eur"] = st.number_input("CAPEX estimé [€]", value=float(src["solar_thermal"]["capex_eur"]), min_value=0.0, step=500.0, key="st_capex")
                src["solar_thermal"]["azimuth_deg"] = st.slider("Azimut [° depuis S]", -90, 90, int(src["solar_thermal"]["azimuth_deg"]), key="st_az")

    # ── PV ───────────────────────────────────────────────────────────────
    with st.expander("⚡ Panneaux photovoltaïques (PV)", expanded=src["pv"]["enabled"]):
        src["pv"]["enabled"] = st.checkbox("Activer PV", value=src["pv"]["enabled"], key="chk_pv")
        if src["pv"]["enabled"]:
            col1, col2, col3 = st.columns(3)
            with col1:
                src["pv"]["peak_kw"] = st.number_input("Puissance crête [kWc]", value=float(src["pv"]["peak_kw"]), min_value=1.0, max_value=1000.0, step=1.0, key="pv_peak")
                src["pv"]["tilt_deg"] = st.slider("Inclinaison [°]", 0, 60, int(src["pv"]["tilt_deg"]), key="pv_tilt")
            with col2:
                src["pv"]["eta_inverter"] = st.number_input("Rend. onduleur", value=float(src["pv"]["eta_inverter"]), min_value=0.85, max_value=0.99, step=0.01, format="%.2f", key="pv_inv")
                src["pv"]["azimuth_deg"] = st.slider("Azimut [° depuis S]", -90, 90, int(src["pv"]["azimuth_deg"]), key="pv_az")
            with col3:
                src["pv"]["capex_eur"] = st.number_input("CAPEX estimé [€]", value=float(src["pv"]["capex_eur"]), min_value=0.0, step=500.0, key="pv_capex")

    # ── Batterie ──────────────────────────────────────────────────────────
    with st.expander("🔋 Stockage batterie", expanded=src["battery"]["enabled"]):
        src["battery"]["enabled"] = st.checkbox("Activer batterie", value=src["battery"]["enabled"], key="chk_bat")
        if src["battery"]["enabled"]:
            col1, col2, col3 = st.columns(3)
            with col1:
                src["battery"]["capacity_kwh"] = st.number_input("Capacité [kWh]", value=float(src["battery"]["capacity_kwh"]), min_value=2.0, max_value=500.0, step=2.0, key="bat_cap")
                soc_min_pct = st.slider("SoC min [%]", 5, 30, int(src["battery"]["soc_min_kwh"]/src["battery"]["capacity_kwh"]*100), key="bat_soc_min")
                src["battery"]["soc_min_kwh"] = src["battery"]["capacity_kwh"] * soc_min_pct / 100.0
            with col2:
                src["battery"]["power_charge_kw"] = st.number_input("Puissance charge [kW]", value=float(src["battery"]["power_charge_kw"]), min_value=1.0, max_value=200.0, step=1.0, key="bat_pch")
                src["battery"]["power_discharge_kw"] = st.number_input("Puissance décharge [kW]", value=float(src["battery"]["power_discharge_kw"]), min_value=1.0, max_value=200.0, step=1.0, key="bat_pdch")
            with col3:
                src["battery"]["capex_eur_kwh"] = st.number_input("Coût [€/kWh]", value=float(src["battery"]["capex_eur_kwh"]), min_value=100.0, max_value=2000.0, step=50.0, key="bat_cost")
                cap = src["battery"]["capacity_kwh"]
                st.metric("CAPEX batterie", f"{cap * src['battery']['capex_eur_kwh']:,.0f} €")

    st.session_state.sources = src

    # Récapitulatif CAPEX
    st.markdown("---")
    st.subheader("💰 Récapitulatif CAPEX estimé")
    capex = _calc_capex_display()
    if capex:
        col1, col2 = st.columns(2)
        with col1:
            for label, val in capex.items():
                st.metric(label, f"{val:,.0f} €")
        with col2:
            st.metric("**TOTAL**", f"**{sum(capex.values()):,.0f} €**")
    else:
        st.warning("Aucune source activée.")


# ─── PAGE 4 : Simulation ─────────────────────────────────────────────────────
def page_simulation():
    import plotly.express as px
    import plotly.graph_objects as go

    st.title("🔥 Simulation")
    st.caption("Lancez le calcul horaire 8760 h de votre installation.")

    # Vérification : au moins une source active
    src = st.session_state.sources
    any_active = any(src[s].get("enabled", False) for s in ["gshp", "ashp", "solar_thermal"])
    if not any_active:
        st.warning("⚠ Activez au moins une source thermique (GSHP, ASHP ou Solaire) dans la page Système.")
        return

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_btn = st.button("▶ Lancer la simulation", type="primary", use_container_width=True)
    with col_info:
        if st.session_state.csv_path:
            st.info(f"📄 Données : fichier CSV chargé")
        else:
            st.info("🧪 Données : profils synthétiques")

    if run_btn:
        st.session_state.simulation_ran = False
        st.session_state.simulation_results = None

        config = _build_config_dict()
        proj = st.session_state.project

        with st.spinner("⏳ Simulation en cours (8760 h)…"):
            try:
                from main_engine_v47 import run_simulation
                with tempfile.TemporaryDirectory() as tmpdir:
                    result = run_simulation(config_dict=config, output_dir=tmpdir)

                # Garder les tableaux numpy en mémoire (copie)
                if "dispatch" in result and "hourly" in result["dispatch"]:
                    hourly = result["dispatch"]["hourly"]
                    result["_hourly_arrays"] = {
                        k: v.copy() if hasattr(v, "copy") else v
                        for k, v in hourly.items()
                    }

                st.session_state.simulation_results = result
                st.session_state.simulation_ran = True
                st.success("✅ Simulation terminée !")
            except Exception as e:
                st.error(f"❌ Erreur simulation : {e}")
                import traceback
                st.code(traceback.format_exc())
                return

    # ── Affichage des résultats ──────────────────────────────────────────
    if st.session_state.simulation_ran and st.session_state.simulation_results:
        result = st.session_state.simulation_results
        ann = result.get("annual", {})
        eco = result.get("economics", {})

        st.markdown("---")
        st.subheader("📊 Résultats annuels")

        # KPIs
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Demande chaleur", f"{ann.get('heat_demand_kwh',0)/1000:.1f} MWh")
        with col2:
            st.metric("SCOP système", f"{ann.get('scop_system',0):.2f}")
        with col3:
            st.metric("Réseau électrique", f"{ann.get('grid_kwh',0)/1000:.1f} MWh")
        with col4:
            st.metric("PV produit", f"{ann.get('pv_yield_kwh',0)/1000:.1f} MWh")

        col5, col6, col7, col8 = st.columns(4)
        with col5:
            st.metric("Non couvert (chauf.)", f"{ann.get('unmet_heat_kwh',0):.0f} kWh")
        with col6:
            st.metric("Solaire thermique", f"{ann.get('solar_th_kwh',0)/1000:.1f} MWh")
        with col7:
            st.metric("GSHP chaleur", f"{ann.get('gshp_heat_kwh',0)/1000:.1f} MWh")
        with col8:
            st.metric("ASHP chaleur", f"{ann.get('ashp_heat_kwh',0)/1000:.1f} MWh")

        # Graphique 1 : Camembert sources de chaleur
        st.subheader("🥧 Répartition des sources de chaleur")
        heat_sources = {
            "Solaire thermique": ann.get("solar_th_kwh", 0),
            "GSHP": ann.get("gshp_heat_kwh", 0),
            "ASHP": ann.get("ashp_heat_kwh", 0),
            "Appoint électrique": ann.get("backup_kwh", 0),
            "Stockage": ann.get("storage_discharge_kwh", 0),
        }
        heat_filtered = {k: v for k, v in heat_sources.items() if v > 0}
        if heat_filtered:
            fig_pie = px.pie(
                names=list(heat_filtered.keys()),
                values=list(heat_filtered.values()),
                color_discrete_sequence=px.colors.qualitative.Set2,
                title="Sources de chaleur [kWh/an]",
            )
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig_pie, use_container_width=True)

        # Graphique 2 : Barres mensuelles
        st.subheader("📅 Production mensuelle")
        hourly_arrays = result.get("_hourly_arrays", {})
        if hourly_arrays:
            import pandas as pd
            monthly_data = {"Mois": MONTHS}
            keys_monthly = [
                ("q_solar_th_kw", "Solaire th."),
                ("q_gshp_heat_kw", "GSHP"),
                ("q_ashp_heat_kw", "ASHP"),
                ("q_backup_kw", "Appoint"),
                ("p_pv_kw", "PV"),
            ]
            for key, label in keys_monthly:
                if key in hourly_arrays and np.sum(hourly_arrays[key]) > 0:
                    monthly_data[label] = _monthly_sum(hourly_arrays[key])

            df_monthly = pd.DataFrame(monthly_data)
            value_cols = [c for c in df_monthly.columns if c != "Mois"]
            if value_cols:
                fig_bar = px.bar(
                    df_monthly, x="Mois", y=value_cols,
                    barmode="stack",
                    title="Production mensuelle par source [kWh]",
                    labels={"value": "kWh", "variable": "Source"},
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                st.plotly_chart(fig_bar, use_container_width=True)

        # Téléchargements
        st.subheader("⬇️ Téléchargement des résultats")
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            ann_json = json.dumps(ann, indent=2, ensure_ascii=False, default=str)
            st.download_button("📋 Bilan annuel (JSON)", ann_json, "bilan_annuel.json", "application/json")
        with col_dl2:
            eco_json = json.dumps(eco, indent=2, ensure_ascii=False, default=str)
            st.download_button("💶 Résultats économiques (JSON)", eco_json, "economie.json", "application/json")

        if hourly_arrays:
            try:
                import pandas as pd
                hours_idx = list(range(8760))
                df_out = pd.DataFrame({"heure": hours_idx})
                for k, arr in hourly_arrays.items():
                    if isinstance(arr, np.ndarray) and arr.shape == (8760,):
                        df_out[k] = arr
                csv_buf = io.StringIO()
                df_out.to_csv(csv_buf, index=False, float_format="%.3f")
                st.download_button("📊 Données horaires (CSV)", csv_buf.getvalue(),
                                   "resultats_horaires.csv", "text/csv")
            except Exception:
                pass


# ─── PAGE 5 : Rentabilité ────────────────────────────────────────────────────
def page_rentabilite():
    import plotly.graph_objects as go

    st.title("💶 Analyse économique")
    st.caption("Paramètres économiques et indicateurs de rentabilité.")

    eco = st.session_state.economics

    st.subheader("⚙️ Paramètres économiques")
    col1, col2, col3 = st.columns(3)
    with col1:
        eco["electricity_buy_eur_kwh"] = st.number_input(
            "Prix achat électricité [€/kWh]",
            value=float(eco["electricity_buy_eur_kwh"]), min_value=0.05, max_value=1.0, step=0.01, format="%.4f", key="eco_buy")
        eco["electricity_sell_eur_kwh"] = st.number_input(
            "Prix vente surplus PV [€/kWh]",
            value=float(eco["electricity_sell_eur_kwh"]), min_value=0.0, max_value=0.5, step=0.01, format="%.2f", key="eco_sell")
        eco["gas_eur_kwh"] = st.number_input(
            "Prix gaz [€/kWh]",
            value=float(eco["gas_eur_kwh"]), min_value=0.01, max_value=0.5, step=0.01, format="%.3f", key="eco_gas")
    with col2:
        eco["elec_escalation_rate"] = st.number_input(
            "Hausse énergie/an [%]",
            value=float(eco["elec_escalation_rate"]*100), min_value=0.0, max_value=10.0, step=0.1, format="%.1f", key="eco_esc") / 100.0
        eco["discount_rate"] = st.number_input(
            "Taux d'actualisation [%]",
            value=float(eco["discount_rate"]*100), min_value=0.0, max_value=10.0, step=0.1, format="%.1f", key="eco_disc") / 100.0
        eco["study_years"] = st.slider("Durée d'étude [ans]", 10, 30, int(eco["study_years"]), key="eco_years")
    with col3:
        eco["income_category"] = st.selectbox(
            "Catégorie revenus (MaPrimeRénov')",
            ["tres_modeste", "modeste", "intermediaire", "superieure"],
            index=["tres_modeste","modeste","intermediaire","superieure"].index(eco.get("income_category","intermediaire")),
            key="eco_income")
        eco["baseline_heating_system"] = st.selectbox(
            "Système de référence",
            ["gaz_condensation", "fioul", "electrique_direct", "pac_air_air"],
            index=["gaz_condensation","fioul","electrique_direct","pac_air_air"].index(eco.get("baseline_heating_system","gaz_condensation")),
            key="eco_baseline")

    st.session_state.economics = eco

    # ── Résultats économiques si simulation disponible ─────────────────
    if st.session_state.simulation_ran and st.session_state.simulation_results:
        result = st.session_state.simulation_results
        eco_res = result.get("economics", {})
        ann = result.get("annual", {})

        st.markdown("---")
        st.subheader("📈 Indicateurs de rentabilité")

        st.caption("⚠️ Résultats estimatifs — non contractuels. Vérifiez les hypothèses avant usage officiel.")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            capex = eco_res.get("capex_total_eur", sum(_calc_capex_display().values()))
            st.metric("Investissement net", f"{eco_res.get('net_investment_eur', capex):,.0f} €")
        with col2:
            st.metric("Économies annuelles", f"{eco_res.get('annual_savings_eur',0):,.0f} €/an")
        with col3:
            trs = eco_res.get("trs_years")
            st.metric("Retour sur invest.", f"{trs:.1f} ans" if trs else "N/A")
        with col4:
            van = eco_res.get("van_eur")
            st.metric("VAN", f"{van:,.0f} €" if van else "N/A")

        col5, col6 = st.columns(2)
        with col5:
            tri = eco_res.get("tri_pct")
            st.metric("TRI", f"{tri:.1f} %" if tri else "N/A")
        with col6:
            opex = eco_res.get("opex_net_eur")
            st.metric("Coût annuel d'exploitation", f"{opex:,.0f} €/an" if opex else "N/A")

        # Aides
        aids = eco_res.get("aids", {})
        if aids:
            st.subheader("🏛 Aides françaises estimées")
            for nom, val in aids.items():
                if isinstance(val, (int, float)) and val > 0:
                    label = nom.replace("_eur", "").replace("_", " ").title()
                    st.metric(label, f"{val:,.0f} €")

        # Graphique VAN cumulée
        n_years = eco.get("study_years", 20)
        annual_savings = eco_res.get("annual_savings_eur", 0)
        net_invest = eco_res.get("net_investment_eur", sum(_calc_capex_display().values()))
        esc = eco.get("elec_escalation_rate", 0.03)
        disc = eco.get("discount_rate", 0.04)

        if annual_savings > 0 and net_invest > 0:
            cumul = [-net_invest]
            van_cumul = -net_invest
            for y in range(1, n_years+1):
                savings_y = annual_savings * (1+esc)**y / (1+disc)**y
                van_cumul += savings_y
                cumul.append(van_cumul)

            fig_van = go.Figure()
            fig_van.add_trace(go.Scatter(
                x=list(range(n_years+1)),
                y=cumul,
                mode="lines+markers",
                name="VAN cumulée",
                line=dict(color="#2E7D32", width=2),
                fill="tozeroy",
                fillcolor="rgba(46,125,50,0.1)",
            ))
            fig_van.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Seuil de rentabilité")
            fig_van.update_layout(
                title="VAN cumulée sur la durée d'étude",
                xaxis_title="Années",
                yaxis_title="VAN cumulée [€]",
                height=400,
            )
            st.plotly_chart(fig_van, use_container_width=True)
    else:
        st.info("▶ Lancez la simulation (page Simulation) pour obtenir les indicateurs économiques.")


# ─── Navigation et routage ───────────────────────────────────────────────────
def main():
    _init_state()

    # Sidebar
    with st.sidebar:
        st.markdown("## 🌍 DiMii V47")
        st.caption("Dimensionnement Géothermique")
        st.markdown("---")

        pages = ["🏠 Projet", "📂 Données", "⚙️ Système", "🔥 Simulation", "💶 Rentabilité"]
        for page in pages:
            active = st.session_state.page == page
            if st.button(page, use_container_width=True, type="primary" if active else "secondary"):
                st.session_state.page = page
                st.rerun()

        st.markdown("---")
        if st.session_state.simulation_ran:
            st.success("✅ Simulation disponible")
        else:
            st.info("⏳ Simulation non lancée")

        # Statut sources
        src = st.session_state.sources
        active_srcs = [s.upper() for s in ["gshp","ashp","solar_thermal","pv","battery"] if src[s].get("enabled")]
        if active_srcs:
            st.caption("Sources actives :")
            for s in active_srcs:
                st.caption(f"  ✅ {s}")

        st.markdown("---")
        st.caption("Version V47 · Estimatif")
        st.caption("© 2026 DiMii")

    # Routage
    page = st.session_state.page
    if page == "🏠 Projet":
        page_projet()
    elif page == "📂 Données":
        page_donnees()
    elif page == "⚙️ Système":
        page_systeme()
    elif page == "🔥 Simulation":
        page_simulation()
    elif page == "💶 Rentabilité":
        page_rentabilite()


if __name__ == "__main__":
    main()
