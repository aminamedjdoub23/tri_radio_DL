# Livrables du projet

Ce fichier sert de checklist pour le rendu final du projet de deep learning médical.

## 1. Dépôt GitHub

Lien du dépôt :

```text
https://github.com/aminamedjdoub23/tri_radio_DL
```

Le dépôt contient :

- le code source PyTorch ;
- les scripts d'entraînement ;
- les notebooks d'analyse ;
- le démonstrateur Streamlit ;
- le rapport en Markdown ;
- les consignes d'installation et d'exécution.

Les dossiers lourds ou générés ne sont pas versionnés :

- `data/raw/`
- `data/openi/`
- `outputs/`
- `mlruns/`
- `.venv/`

## 2. Rapport

Fichier principal :

```text
report/rapport.md
```

Fichier de vérification avec la consigne :

```text
report/conformite_consigne.md
```

État des lieux pour le binôme :

```text
report/etat_des_lieux_binome.md
```

Preuves d'exécution locale :

```text
report/preuves_execution.md
report/mlflow_quick_results.csv
report/mlflow_cpu_medium_results.csv
report/openi_text_results.csv
```

Le rapport est structuré selon les sections demandées :

- Problème
- Données
- Analyse exploratoire
- Préparation
- Modélisation supervisée
- Détection d'anomalies
- Modélisation multimodale
- Évaluation
- Tracking MLflow
- Démonstrateur
- Analyse critique
- Conclusion et perspectives

À compléter éventuellement après entraînement plus long :

- les captures MLflow ;
- les captures du démonstrateur ;
- les commentaires sur des performances finales plus robustes.

## 3. Code source

Dossier :

```text
src/
```

Contenu :

- `src/data/` : chargement ChestMNIST et OpenI ;
- `src/models/` : CNN simple, ResNet18, ViT, autoencodeur, texte, multimodal ;
- `src/training/` : entraînements et évaluation ;
- `src/utils/` : métriques, seed, MLflow, figures ;
- `src/app/` : application Streamlit.

## 4. Notebooks

Fichiers :

```text
notebooks/01_eda_chestmnist.ipynb
notebooks/02_results_analysis.ipynb
```

Rôle :

- le premier notebook sert à l'analyse exploratoire de ChestMNIST ;
- le second sert à comparer les runs MLflow après entraînement.

## 5. Expériences MLflow

Commande :

```bash
mlflow ui --backend-store-uri mlruns
```

MLflow doit montrer :

- les paramètres des modèles ;
- les métriques train/validation/test ;
- les artefacts ;
- les figures ROC/AUC ;
- les checkpoints des meilleurs modèles.

Le dossier `mlruns/` peut être joint séparément si l'enseignant demande les traces d'expériences. Il n'est pas poussé sur GitHub pour éviter d'alourdir le dépôt.

## 6. Modèles entraînés

Les checkpoints sont générés dans :

```text
outputs/
```

Exemples attendus après entraînement :

- `outputs/best_simple_cnn.pt`
- `outputs/best_transfer.pt`
- `outputs/best_vit.pt`
- `outputs/best_autoencoder.pt`

Ces fichiers ne sont pas versionnés dans GitHub, car ils peuvent être lourds. Ils peuvent être rendus séparément si nécessaire.

## 7. Démonstrateur

Commande :

```bash
streamlit run src/app/streamlit_app.py
```

Fonctionnalités :

- upload d'une radiographie ;
- affichage des probabilités par pathologie ;
- affichage du score d'anomalie ;
- champ texte optionnel pour discuter de la partie multimodale ;
- avertissement clair : prototype pédagogique, pas dispositif médical.

## 8. Commandes minimales à exécuter

Installation du pipeline principal :

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Notebooks, si nécessaire :

```bash
pip install -r requirements-notebooks.txt
```

Entraînements minimaux pour un rendu crédible :

```bash
python -m src.training.train_supervised --model simple_cnn --config config.yaml
python -m src.training.train_supervised --model transfer --config config.yaml
python -m src.training.train_supervised --model vit --config config.yaml
python -m src.training.train_autoencoder --config config.yaml
```

Alternative réaliste sur CPU si le full run n'est pas faisable :

```bash
python -m src.training.train_supervised --model transfer --config config_cpu_medium.yaml
python -m src.training.train_autoencoder --config config_cpu_medium.yaml
```

Multimodal, uniquement si le CSV OpenI est préparé :

```bash
python -m src.training.train_text --config config_openi_text.yaml
python -m src.training.train_multimodal --config config_openi_multimodal.yaml
```

## 9. Checklist avant rendu final

- [x] Installer les dépendances du pipeline principal dans `.venv`.
- [ ] Exécuter le notebook EDA.
- [x] Entraîner les trois modèles supervisés en mode rapide.
- [x] Entraîner l'autoencodeur en mode rapide.
- [x] Produire un run CPU intermédiaire `transfer + autoencoder`.
- [x] Lancer MLflow et relever les métriques disponibles.
- [x] Compléter les tableaux du rapport avec les vrais résultats disponibles.
- [ ] Ajouter des captures MLflow et Streamlit si demandées.
- [x] Tester le démonstrateur via un smoke test de démarrage.
- [ ] Vérifier que le dépôt GitHub est accessible au binôme.

## Remarque importante

Le dépôt ne contient pas de résultats inventés. Les tableaux marqués "À compléter" doivent être remplis uniquement après entraînement réel des modèles.
