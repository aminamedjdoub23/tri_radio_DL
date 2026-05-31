# Preuves d'exécution locale

Ce document résume les preuves réellement régénérées sur cette machine le 31 mai 2026.

## Dataset

Le fichier ChestMNIST 64 est présent localement :

```text
data/raw/chestmnist_64.npz
```

Vérification :

| Fichier             |           Taille |
| ------------------- | ---------------: |
| `chestmnist_64.npz` | 401604127 octets |

Le fichier a été téléchargé automatiquement via `medmnist` lors des entraînements.

## Environnement

L'environnement principal a été créé dans `.venv`.

| Élément     | Valeur                        |
| ----------- | ----------------------------- |
| Python      | 3.13.1                        |
| PyTorch     | non relevée (CUDA disponible) |
| torchvision | non relevée                   |
| CUDA        | disponible                    |
| CPU         | AMD64 Family 25 Model 68      |
| GPU         | NVIDIA GeForce GTX 1650 Ti    |

Remarque :

- `requirements.txt` est maintenant suffisant pour le pipeline principal ;
- `jupyter` a été déplacé dans `requirements-notebooks.txt` car son installation échouait ici à cause des chemins Windows trop longs ;
- les notebooks restent optionnels et distincts du pipeline d'entraînement.

## Validations réalisées

Commandes exécutées :

```bash
.venv\Scripts\python.exe -m compileall src
.venv\Scripts\python.exe -c "import torch, torchvision, timm, medmnist, mlflow, streamlit; print('imports ok')"
```

Résultat : syntaxe et imports OK.

Vérification GPU :

```bash
.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Résultat : CUDA disponible, GPU détecté.

## Smoke test Streamlit

L'application a été démarrée en mode headless sur le port 8502 puis interrogée localement.

Commande utilisée :

```bash
python -m streamlit run src/app/streamlit_app.py --server.headless true --server.port 8502
```

Résultat :

| Vérification        | Valeur |
| ------------------- | ------ |
| Réponse HTTP locale | `200`  |

Cela prouve que l'application démarre correctement sur cette machine.

## Runs rapides MLflow

Les runs rapides ont été exécutés avec :

```text
config_quick.yaml
```

Commandes :

```bash
.venv\Scripts\python.exe -m src.training.train_supervised --model simple_cnn --config config_quick.yaml
.venv\Scripts\python.exe -m src.training.train_supervised --model transfer --config config_quick.yaml
.venv\Scripts\python.exe -m src.training.train_supervised --model vit --config config_quick.yaml
.venv\Scripts\python.exe -m src.training.train_autoencoder --config config_quick.yaml
```

Résumé exporté :

```text
report/mlflow_quick_results.csv
```

Artefacts réellement générés localement :

```text
outputs_quick/
mlruns_quick/
```

Checkpoints présents :

- `outputs_quick/best_simple_cnn.pt`
- `outputs_quick/best_transfer.pt`
- `outputs_quick/best_vit.pt`
- `outputs_quick/best_autoencoder.pt`

## Runs CPU intermédiaires

Pour obtenir des résultats plus crédibles que les quick runs sur cette machine CPU, une configuration intermédiaire a été ajoutée :

```text
config_cpu_medium.yaml
```

Commandes exécutées :

```bash
.venv\Scripts\python.exe -m src.training.train_supervised --model transfer --config config_cpu_medium.yaml
.venv\Scripts\python.exe -m src.training.train_autoencoder --config config_cpu_medium.yaml
```

Résumé exporté :

```text
report/mlflow_cpu_medium_results.csv
```

Artefacts réellement générés localement :

```text
outputs_cpu_medium/
mlruns_cpu_medium/
```

Checkpoints présents :

- `outputs_cpu_medium/best_transfer.pt`
- `outputs_cpu_medium/best_autoencoder.pt`

## OpenI officiel

Les liens officiels OpenI/NLM utilisés par le script sont :

```text
https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz
https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz
```

Le 30 mai 2026, le téléchargement officiel des rapports XML a fonctionné et a permis de produire :

```text
data/openi/openi_prepared.csv
```

Le CSV contient 7470 lignes image-rapport.

Un entraînement texte seul OpenI a été exécuté avec :

```bash
.venv\Scripts\python.exe -m src.training.train_text --config config_openi_text.yaml
```

Résumé exporté :

```text
report/openi_text_results.csv
```

Résumé exporté (OpenI multimodal) :

```text
report/openi_multimodal_results.csv
```

Artefacts réellement générés localement :

```text
outputs_openi_text/
mlruns_openi_text/
```

Checkpoint présent :

- `outputs_openi_text/best_openi_text.pt`

Configuration prévue pour l'entraînement OpenI image seule + multimodal :

```text
config_openi_multimodal.yaml
```

Entraînement OpenI image seule + multimodal exécuté :

```bash
.venv\Scripts\python.exe -m src.training.train_multimodal --config config_openi_multimodal.yaml
```

Artefacts générés :

```text
outputs_openi_multimodal/
mlruns_openi_multimodal/
```

Checkpoints présents :

- `outputs_openi_multimodal/best_openi_image.pt`
- `outputs_openi_multimodal/best_openi_multimodal.pt`
- `outputs_openi_multimodal/openi_tfidf_vectorizer.joblib`

## Runs finaux GPU

Un run final GPU a été exécuté avec :

```text
config_final.yaml
```

Modèles entraînés :

- simple CNN ;
- transfer learning (DenseNet121) ;
- ViT tiny.

Artefacts générés :

```text
outputs_final/
mlruns_final/
```

Résumé exporté :

```text
report/mlflow_final_results.csv
```

Checkpoints présents :

- `outputs_final/best_simple_cnn.pt`
- `outputs_final/best_transfer.pt`
- `outputs_final/best_vit.pt`

## Limites

- les quick runs prouvent le fonctionnement du pipeline, mais ne constituent pas des résultats finaux robustes ;
- les runs CPU intermédiaires sont plus crédibles, mais restent des entraînements partiels sur sous-échantillon ;
- les captures d'écran MLflow et Streamlit sont ajoutées en annexes du rapport.
