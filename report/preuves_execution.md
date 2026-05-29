# Preuves d'exécution locale

Ce document résume les preuves générées localement avant l'envoi du projet.

## Dataset

Le fichier ChestMNIST 64 est présent localement :

```text
data/raw/chestmnist_64.npz
```

Vérification :

| Fichier | Taille | MD5 |
|---|---:|---|
| `chestmnist_64.npz` | 401604127 octets | `9de6cd0b934ebb5b7426cfba5efbae16` |

Ce MD5 correspond au hash officiel MedMNIST pour `chestmnist_64.npz`.

## Environnement

Les dépendances ont été installées dans `.venv`.

| Élément | Valeur |
|---|---|
| Python | 3.12.10 |
| PyTorch | 2.12.0+cpu |
| torchvision | 0.27.0+cpu |
| CUDA | non disponible |

## Validations réalisées

Commandes exécutées :

```bash
.venv\Scripts\python.exe -m compileall src
.venv\Scripts\python.exe -c "import torch, torchvision, timm, medmnist, mlflow, streamlit"
```

Résultat : syntaxe et imports OK.

## Runs rapides MLflow

Les runs rapides ont été exécutés avec :

```text
config_quick.yaml
```

Résumé exporté :

```text
report/mlflow_quick_results.csv
```

Artefacts générés localement :

```text
outputs_quick/
mlruns_quick/
```

Ces dossiers ne sont pas poussés sur GitHub car ils contiennent des checkpoints, figures et traces générées. Ils peuvent être joints séparément si le professeur demande les artefacts bruts.

## Commandes de reproduction rapide

```bash
.venv\Scripts\python.exe -m src.training.train_supervised --model simple_cnn --config config_quick.yaml
.venv\Scripts\python.exe -m src.training.train_supervised --model transfer --config config_quick.yaml
.venv\Scripts\python.exe -m src.training.train_supervised --model vit --config config_quick.yaml
.venv\Scripts\python.exe -m src.training.train_autoencoder --config config_quick.yaml
```

## Limite

Les runs rapides prouvent que le pipeline fonctionne, mais ils utilisent un sous-échantillon et une seule epoch. Les métriques ne doivent pas être présentées comme performance finale robuste.

## OpenI officiel

Les liens officiels OpenI/NLM utilisés par le script sont :

```text
https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz
https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz
```

Lors du test du 29 mai 2026, le téléchargement officiel des rapports XML a fonctionné et a permis de produire :

```text
data/openi/openi_prepared.csv
```

Le CSV contient 7470 lignes image-rapport. Les images PNG OpenI restent à télécharger/extracter avant d'entraîner réellement `train_multimodal.py`.
