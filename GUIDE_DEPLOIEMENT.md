# Guide de déploiement — DiMii V47

## Objectif

Rendre l'application accessible en ligne depuis n'importe quel navigateur (PC, Mac, Chromebook, tablette, smartphone), **gratuitement**, via Streamlit Community Cloud.

---

## Option A — Lancement en local (PC / Chromebook)

### Prérequis
- Python 3.9 ou supérieur installé
- Accès à un terminal (PowerShell sous Windows, Terminal sous Mac/Linux, Terminal Linux sur Chromebook)

### Étapes

1. **Décompressez l'archive** `dimii_v47_app.zip` dans un dossier de votre choix, par exemple `C:\DiMii\` ou `~/dimii/`.

2. **Ouvrez un terminal** dans ce dossier.

3. **Installez les dépendances** :
   ```bash
   pip install -r requirements.txt
   ```

4. **Lancez l'application** :
   ```bash
   streamlit run app.py
   ```

5. **Ouvrez votre navigateur** à l'adresse affichée (généralement http://localhost:8501).

> Sur Chromebook : activez Linux (Paramètres → Avancé → Développeurs → Environnement Linux), puis suivez les étapes ci-dessus dans le terminal Linux.

---

## Option B — Déploiement en ligne (Streamlit Community Cloud) — recommandé

Cette option rend l'application accessible à **tout le monde** via une URL publique, sans installation côté utilisateur.

### Étape 1 : Créer un compte GitHub (si vous n'en avez pas)

Allez sur https://github.com et créez un compte gratuit.

### Étape 2 : Créer un nouveau dépôt GitHub

1. Cliquez sur **New repository** (bouton vert en haut à droite).
2. Nom du dépôt : `dimii-v47` (ou ce que vous voulez).
3. Laissez-le **Public** (requis pour Streamlit Cloud gratuit).
4. Cliquez **Create repository**.

### Étape 3 : Uploader les fichiers

Dans votre nouveau dépôt, cliquez **Add file → Upload files**, puis glissez-déposez **tous les fichiers** du dossier `dimii_v47/` extrait de l'archive :

```
app.py
main_engine_v47.py
gshp_geofield_v47.py
ashp_v47.py
solar_thermal_v47.py
pv_battery_v47.py
dispatch_multisource_v47.py
requirements.txt
README.md
.streamlit/config.toml
```

> **Important** : pour `.streamlit/config.toml`, créez d'abord le dossier `.streamlit` sur GitHub (uploadez le fichier avec le chemin `.streamlit/config.toml`).

Cliquez **Commit changes**.

### Étape 4 : Déployer sur Streamlit Community Cloud

1. Allez sur https://share.streamlit.io et connectez-vous avec votre compte GitHub.
2. Cliquez **New app**.
3. Sélectionnez votre dépôt `dimii-v47`.
4. Branch : `main`
5. Main file path : `app.py`
6. Cliquez **Deploy!**

Le déploiement prend 2 à 5 minutes. Vous obtiendrez une URL du type :
```
https://votre-nom-dimii-v47-app-xxxx.streamlit.app
```

Cette URL peut être partagée avec n'importe qui. Elle fonctionne depuis n'importe quel navigateur.

### Étape 5 (optionnel) : Mettre à jour l'application

Pour mettre à jour l'application, modifiez les fichiers dans GitHub (ou uploadez de nouveaux fichiers). Streamlit redéploie automatiquement en quelques secondes.

---

## Option C — Partage via ngrok (local → accessible en ligne)

Si vous ne voulez pas GitHub mais souhaitez partager votre instance locale temporairement :

1. Lancez l'application localement (Option A).
2. Installez ngrok : https://ngrok.com (gratuit)
3. Dans un second terminal :
   ```bash
   ngrok http 8501
   ```
4. Partagez l'URL `https://xxxx.ngrok.io` générée.

> L'URL n'est valide que tant que votre ordinateur est allumé et ngrok actif.

---

## Vérification que tout fonctionne

Après déploiement (local ou cloud), vérifiez les points suivants :

1. ✅ La page **Projet** s'affiche avec la carte de localisation.
2. ✅ La page **Système** permet d'activer/désactiver les sources.
3. ✅ La page **Simulation** lance un calcul et affiche les graphiques.
4. ✅ La page **Rentabilité** affiche la courbe VAN et les aides.
5. ✅ Les boutons de téléchargement fonctionnent.

---

## Résolution des problèmes courants

| Problème | Solution |
|----------|----------|
| `ModuleNotFoundError: No module named 'streamlit'` | Relancez `pip install -r requirements.txt` |
| `Port 8501 already in use` | Lancez avec `streamlit run app.py --server.port 8502` |
| Streamlit Cloud : erreur de déploiement | Vérifiez que tous les fichiers sont bien dans le dépôt, y compris `requirements.txt` |
| Simulation lente (> 30 s) | Normal pour la première exécution ; le modèle FLS est lourd |
| Graphiques vides | Vérifiez qu'au moins une source est activée dans la page Système |

---

## Support

Pour toute question, ouvrez une issue sur le dépôt GitHub ou contactez l'équipe DiMii.

> ⚠️ Les résultats de DiMii V47 sont **estimatifs** et non contractuels. Ne pas les utiliser comme résultats réglementaires sans vérification par un bureau d'études qualifié.
