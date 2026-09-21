# DiMii V47 — Dimensionnement Géothermique

Application web de dimensionnement énergétique et comparaison économique de solutions techniques pour les bâtiments.

## Fonctionnalités

- **PAC géothermique sur sondes (GSHP)** : modèle FLS g-function, simulation 8760 h
- **PAC aérothermique (ASHP)** : COP variable selon température extérieure
- **Solaire thermique** : modèle ASHRAE avec données synthétiques ou réelles
- **Photovoltaïque** : rendement fonction température, orientation et inclinaison
- **Stockage batterie** : gestion SOC, auto-consommation PV
- **Analyse économique** : VAN, TRS, TRI, aides françaises (MaPrimeRénov', CEE)

## Lancement local

```bash
pip install -r requirements.txt
streamlit run app.py
```

Ouvrez http://localhost:8501 dans votre navigateur.

## Déploiement Streamlit Community Cloud (gratuit, accessible depuis partout)

Voir `GUIDE_DEPLOIEMENT.md` pour les instructions complètes.

## Structure des fichiers

| Fichier | Rôle |
|---------|------|
| `app.py` | Interface Streamlit multi-pages |
| `main_engine_v47.py` | Orchestrateur de simulation |
| `gshp_geofield_v47.py` | Modèle géothermie sur sondes |
| `ashp_v47.py` | Modèle PAC aérothermique |
| `solar_thermal_v47.py` | Modèle solaire thermique et PVT |
| `pv_battery_v47.py` | Modèle PV et stockage batterie |
| `dispatch_multisource_v47.py` | Dispatch multi-sources 8760 h |
| `requirements.txt` | Dépendances Python |

## Avertissement

Les résultats sont **estimatifs** et non contractuels. Vérifiez les hypothèses avant tout usage officiel.

© 2026 DiMii V47
