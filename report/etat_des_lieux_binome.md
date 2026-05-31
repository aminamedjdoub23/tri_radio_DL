# État des lieux pour le binôme

Dernière mise à jour : 31 mai 2026.

## Ce qui est prêt

Le dépôt GitHub contient une base complète et cohérente avec la consigne :

- code PyTorch modulaire ;
- chargement ChestMNIST via MedMNIST ;
- classification multi-label 14 pathologies ;
- CNN simple, ResNet18 et ViT ;
- autoencodeur convolutionnel pour score d'anomalie ;
- preuve de concept OpenI texte seul réellement exécutée ;
- preuve de concept multimodale image + texte exécutée ;
- tracking MLflow ;
- démonstrateur Streamlit ;
- rapport structuré ;
- checklist de livrables ;
- fichier de conformité avec la consigne.

## Environnement validé localement

L'environnement `.venv` a été recréé et validé localement.

Versions vérifiées :

| Élément     | Valeur                   |
| ----------- | ------------------------ |
| Python      | 3.13.1                   |
| PyTorch     | non relevée (CUDA dispo) |
| torchvision | non relevée              |
| CUDA        | disponible               |
| CPU         | AMD64 Family 25 Model 68 |

Le projet a été validé avec :

```bash
.venv\Scripts\python.exe -m compileall src
.venv\Scripts\python.exe -c "import torch, torchvision, timm, medmnist, mlflow, streamlit"
```

Point pratique :

- `requirements.txt` installe maintenant le pipeline principal ;
- `requirements-notebooks.txt` porte la dépendance `jupyter`, laissée optionnelle à cause des chemins Windows trop longs sur cette machine.

## Pourquoi il y a deux configs locales de validation

### 1. Config rapide

```text
config_quick.yaml
```

Elle utilise :

- ChestMNIST 64 ;
- images redimensionnées à 64 ;
- 1 epoch ;
- sous-échantillon de 512 images train, 128 validation, 128 test ;
- modèles non pré-entraînés dans cette config rapide.

Objectif :

- prouver que tout le pipeline tourne localement ;
- générer des checkpoints, figures et traces MLflow.

### 2. Config CPU intermédiaire

```text
config_cpu_medium.yaml
```

Elle utilise :

- ChestMNIST 64 ;
- images redimensionnées à 128 ;
- 3 epochs ;
- sous-échantillon de 4096 images train, 512 validation, 512 test ;
- ResNet18 pré-entraîné ;
- autoencodeur plus long que le quick run.

Objectif :

- produire des résultats plus crédibles que les quick runs sur CPU ;
- sans prétendre remplacer un entraînement complet GPU.

## Runs exécutés sur cette machine

### Runs rapides

```bash
.venv\Scripts\python.exe -m src.training.train_supervised --model simple_cnn --config config_quick.yaml
.venv\Scripts\python.exe -m src.training.train_supervised --model transfer --config config_quick.yaml
.venv\Scripts\python.exe -m src.training.train_supervised --model vit --config config_quick.yaml
.venv\Scripts\python.exe -m src.training.train_autoencoder --config config_quick.yaml
```

Résultats rapides obtenus :

| Run         | Test loss | F1 macro | Précision macro | Rappel macro | Seuil AE | Taux atypique test |
| ----------- | --------: | -------: | --------------: | -----------: | -------: | -----------------: |
| simple_cnn  |    0.4739 |   0.0000 |          0.0000 |       0.0000 |        - |                  - |
| transfer    |    0.2394 |   0.0000 |          0.0000 |       0.0000 |        - |                  - |
| vit         |    0.1717 |   0.0000 |          0.0000 |       0.0000 |        - |                  - |
| autoencoder |         - |        - |               - |            - |   0.0738 |             0.0547 |

Les AUC macro sont non définies sur ce très petit sous-échantillon parce que certaines classes n'ont qu'une seule valeur dans le test. C'est normal pour un run rapide, et c'est une raison de ne pas présenter ces résultats comme résultats finaux.

### Runs CPU intermédiaires

```bash
.venv\Scripts\python.exe -m src.training.train_supervised --model transfer --config config_cpu_medium.yaml
.venv\Scripts\python.exe -m src.training.train_autoencoder --config config_cpu_medium.yaml
```

Résultats CPU intermédiaires obtenus :

| Run         | Test loss | AUC macro test | F1 macro | Précision macro | Rappel macro | Seuil AE | Taux atypique test |
| ----------- | --------: | -------------: | -------: | --------------: | -----------: | -------: | -----------------: |
| transfer    |    0.1794 |         0.6727 |   0.0022 |          0.0102 |       0.0012 |        - |                  - |
| autoencoder |         - |              - |        - |               - |            - | 0.000611 |             0.0645 |

Interprétation :

- le run `transfer` commence à donner une AUC macro exploitable, mais le seuil 0.5 reste trop conservateur sur ce sous-échantillon, d'où un F1 quasi nul ;
- l'autoencodeur intermédiaire produit un seuil beaucoup plus stable et des erreurs de reconstruction nettement plus faibles que dans le quick run ;
- ces runs sont plus crédibles techniquement que la config rapide, mais restent insuffisants pour une conclusion expérimentale finale.

## Artefacts générés localement

### ChestMNIST quick

- `outputs_quick/best_simple_cnn.pt`
- `outputs_quick/best_transfer.pt`
- `outputs_quick/best_vit.pt`
- `outputs_quick/best_autoencoder.pt`
- figures ROC/AUC par modèle supervisé
- figure de reconstructions de l'autoencodeur
- traces MLflow dans `mlruns_quick/`

### ChestMNIST CPU intermédiaire

- `outputs_cpu_medium/best_transfer.pt`
- `outputs_cpu_medium/best_autoencoder.pt`
- figures ROC/AUC pour `transfer`
- figure de reconstructions AE
- traces MLflow dans `mlruns_cpu_medium/`

### OpenI texte seul

- `data/openi/openi_prepared.csv`
- `outputs_openi_text/best_openi_text.pt`
- traces MLflow dans `mlruns_openi_text/`

## OpenI texte seul exécuté

Commande utilisée :

```bash
.venv\Scripts\python.exe -m src.data.prepare_openi_official --reports-only
.venv\Scripts\python.exe -m src.training.train_text --config config_openi_text.yaml
```

Résultat texte OpenI :

| Modèle       | AUC macro test | F1 macro test | Précision macro | Rappel macro | Loss test |
| ------------ | -------------: | ------------: | --------------: | -----------: | --------: |
| TF-IDF + MLP |        0.96845 |       0.49645 |         0.93095 |      0.35906 |   0.11480 |

Les images PNG OpenI ont été téléchargées et `train_multimodal.py` a été exécuté, ce qui complète la comparaison image seule / texte seul / multimodal sur OpenI.

## Runs finaux GPU (ChestMNIST)

Les trois modèles supervisés ont été entraînés avec `config_final.yaml` :

| Modèle     | AUC macro test | F1 macro test | Loss test |
| ---------- | -------------: | ------------: | --------: |
| simple_cnn |         0.6683 |        0.0000 |    0.1780 |
| transfer   |         0.8170 |        0.1166 |    0.1544 |
| vit        |         0.8060 |        0.0952 |    0.1574 |

## Démonstrateur testé

Un smoke test Streamlit a été fait en local :

```bash
python -m streamlit run src/app/streamlit_app.py --server.headless true --server.port 8502
```

Résultat :

- l'application démarre ;
- la réponse HTTP locale est `200`.

## Ce qui reste à faire pour un rendu final très solide

- exécuter ou capturer les notebooks si le professeur veut des preuves notebook explicites ;
- ajouter des mesures de temps plus précises si un rendu très détaillé est exigé.

## À dire si on rend maintenant

Le code et les livrables sont complets, les artefacts locaux existent réellement sur cette machine, et le pipeline est démontré de bout en bout :

- ChestMNIST téléchargé et entraînements exécutés ;
- checkpoints et traces MLflow présents ;
- Streamlit validé ;
- OpenI texte seul et multimodal exécutés ;
- runs finaux GPU simple CNN / transfer / ViT exécutés ;
- rapport et exports mis à jour avec de vraies métriques.

En revanche, il faut rester honnête :

- les quick runs ne sont pas des résultats finaux ;
- les runs CPU intermédiaires restent utiles pour comparer la progression, mais ne remplacent pas les runs finaux GPU ;
- les résultats restent locaux et nécessiteraient une validation externe pour conclure.
