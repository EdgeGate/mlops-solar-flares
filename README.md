# 🌞 Solar Flares Classification Pipeline

## 🌟 Vue d’ensemble

Ce projet met en place un **pipeline MLOps complet** pour la classification en temps réel des éruptions solaires à partir des données de flux **X-ray des satellites GOES**.
Le modèle prédit l’intensité selon l’échelle NOAA (**A, B, C, M, X**) et permet d’anticiper les effets critiques sur :

* **Satellites et GPS** (perturbations de signaux)
* **Réseaux électriques** (surcharges, coupures)
* **Aviation** (exposition aux radiations polaires)
* **Missions spatiales** (protection astronautes et équipements)

Objectif : fournir une **alerte précoce** pour les classes **M** et **X**.

---

## 🚀 Installation & Démarrage

### Prérequis

* **Docker & Docker Compose**
* **Python 3.10+**
* **8GB RAM minimum**
* Accès internet (NOAA)

### Étapes rapides

```bash
git clone <repo>
cd solar-flares-pipeline

# Variables (optionnel)
cp .env.example .env

# Démarrage complet
docker-compose up -d

# Vérification
docker-compose ps
```

### Services disponibles

| Service    | URL                                            | Rôle                                   |
| ---------- | ---------------------------------------------- | -------------------------------------- |
| Airflow    | [http://localhost:8080](http://localhost:8080) | Orchestration ingestion / entraînement |
| MLflow     | [http://localhost:5000](http://localhost:5000) | Suivi et registre des modèles          |
| API        | [http://localhost:8000](http://localhost:8000) | Prédictions temps réel                 |
| Prometheus | [http://localhost:9090](http://localhost:9090) | Collecte des métriques                 |
| Grafana    | [http://localhost:3000](http://localhost:3000) | Tableaux de bord                       |

---

## 📊 Pipeline & Modèle

### Données

* **SWPC (temps réel)** : flux minute, latence 2–5 min
* **NCEI (historique)** : archives NetCDF, latence \~48h

### Modèle ML

* **Gradient Boosting Classifier** (scikit-learn)
* Gestion du déséquilibre par **poids équilibrés**
* Seuil ajusté pour améliorer la détection **M/X**
* Validation : **StratifiedKFold (3-fold)**

---

## 🔧 Déploiement & API

### Local

```bash
docker-compose up -d
```

### Production

```bash
docker-compose -f docker-compose.prod.yml up -d
```

### Endpoints principaux

| Endpoint      | Méthode | Description                      |
| ------------- | ------- | -------------------------------- |
| `/health`     | GET     | Santé du service                 |
| `/ready`      | GET     | Vérifie que le modèle est chargé |
| `/predict`    | POST    | Prédiction d’une instance        |
| `/model-info` | GET     | Infos version du modèle          |
| `/metrics`    | GET     | Exposition Prometheus            |

### Exemple requête

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [[1e-6, 12, 720, "G16"]]}'
```

---

## 🔄 Automatisation & Réentraînement

* **Airflow DAGs** :

  * `xrs_update_12h` : mise à jour des données toutes les 12h
  * `solar_flares_train_12h` : réentraînement automatique deux fois par jour

* **MLflow Registry** :

  * Suivi des runs et artefacts
  * Promotion auto si les métriques sont valides (F1 ≥ 0.8)

---

## 📈 Monitoring

* **Prometheus** : latence API, erreurs HTTP, versions du modèle
* **Grafana** : dashboards (santé API, performances du modèle, fraîcheur des données)
* **Airflow** : suivi DAG ingestion et entraînement

---

## 🚨 Maintenance & Support

* Logs :

  ```bash
  docker-compose logs -f
  ```
* Redémarrage automatique des services en prod
* Volumes persistants PostgreSQL
* Vérification modèle actif :

  ```bash
  curl http://localhost:8000/model-info
  ```
