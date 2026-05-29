# État des lieux pour le binôme

Dernière mise à jour : 29 mai 2026.

## Ce qui est prêt

Le dépôt GitHub contient maintenant une base complète et cohérente avec la consigne :

- code PyTorch modulaire ;
- chargement ChestMNIST via MedMNIST ;
- classification multi-label 14 pathologies ;
- CNN simple, ResNet18 et ViT ;
- autoencodeur convolutionnel pour score d'anomalie ;
- preuve de concept OpenI image + texte prévue dans le code ;
- tracking MLflow ;
- démonstrateur Streamlit ;
- rapport structuré ;
- checklist de livrables ;
- fichier de conformité avec la consigne.

## Environnement validé localement

L'environnement `.venv` a été créé et les dépendances sont installées localement.

Versions vérifiées :

| Élément | Valeur |
|---|---|
| Python | 3.12.10 |
| PyTorch | 2.12.0+cpu |
| torchvision | 0.27.0+cpu |
| CUDA | non disponible |
| CPU | AMD64 Family 25 Model 68 |

Le projet a été validé avec :

```bash
.venv\Scripts\python.exe -m compileall src
.venv\Scripts\python.exe -c "import torch, torchvision, timm, medmnist, mlflow, streamlit"
```

## Pourquoi il y a une config rapide

Le téléchargement automatique de ChestMNIST 224 a échoué : le fichier pèse environ 3,89 Go et le téléchargement s'est interrompu. Pour pouvoir générer des artefacts locaux sans bloquer le projet, une configuration rapide a été ajoutée :

```text
config_quick.yaml
```

Elle utilise :

- ChestMNIST 64 ;
- images redimensionnées à 64 ;
- 1 epoch ;
- sous-échantillon de 512 images train, 128 validation, 128 test ;
- modèles non pré-entraînés dans cette configuration rapide.

Important : ces résultats rapides servent à vérifier le pipeline et produire des artefacts, pas à conclure scientifiquement sur les performances.

## Runs rapides exécutés

Commande utilisée :

```bash
.venv\Scripts\python.exe -m src.training.train_supervised --model simple_cnn --config config_quick.yaml
.venv\Scripts\python.exe -m src.training.train_supervised --model transfer --config config_quick.yaml
.venv\Scripts\python.exe -m src.training.train_supervised --model vit --config config_quick.yaml
.venv\Scripts\python.exe -m src.training.train_autoencoder --config config_quick.yaml
```

Résultats rapides obtenus :

| Run | Test loss | F1 macro | Précision macro | Rappel macro | Seuil AE | Taux atypique test |
|---|---:|---:|---:|---:|---:|---:|
| simple_cnn | 0.4738 | 0.0000 | 0.0000 | 0.0000 | - | - |
| transfer | 0.2382 | 0.0000 | 0.0000 | 0.0000 | - | - |
| vit | 0.1717 | 0.0000 | 0.0000 | 0.0000 | - | - |
| autoencoder | - | - | - | - | 0.0738 | 0.0547 |

Les AUC macro sont non définies sur ce très petit sous-échantillon parce que certaines classes n'ont qu'une seule valeur dans le test. C'est normal pour un run rapide, et c'est une raison de ne pas présenter ces résultats comme résultats finaux.

## Artefacts générés localement

Les fichiers suivants existent localement dans `outputs_quick/` :

- `best_simple_cnn.pt`
- `best_transfer.pt`
- `best_vit.pt`
- `best_autoencoder.pt`
- figures ROC/AUC par modèle supervisé ;
- figure de reconstructions de l'autoencodeur.

Les traces MLflow rapides existent localement dans :

```text
mlruns_quick/
```

Ces dossiers ne sont pas poussés sur GitHub pour éviter d'ajouter des fichiers lourds. Ils peuvent être transmis séparément si nécessaire.

## Ce qui reste à faire pour un rendu final solide

Pour un vrai rendu expérimental, il faut lancer les entraînements complets, idéalement sur GPU ou avec plus de temps :

```bash
.venv\Scripts\python.exe -m src.training.train_supervised --model simple_cnn --config config.yaml
.venv\Scripts\python.exe -m src.training.train_supervised --model transfer --config config.yaml
.venv\Scripts\python.exe -m src.training.train_supervised --model vit --config config.yaml
.venv\Scripts\python.exe -m src.training.train_autoencoder --config config.yaml
```

Pour OpenI, il faut préparer le CSV local :

```text
data/openi/openi_prepared.csv
```

Un script officiel a été ajouté :

```bash
.venv\Scripts\python.exe -m src.data.prepare_openi_official
```

Il utilise les archives OpenI/NLM `NLMCXR_reports.tgz` et `NLMCXR_png.tgz`. Lors du test du 29 mai 2026, le téléchargement officiel des rapports a fonctionné et a produit `data/openi/openi_prepared.csv` avec 7470 lignes image-rapport. Un modèle texte seul TF-IDF + MLP a été entraîné sur 4 labels OpenI : Atelectasis, Cardiomegaly, Effusion et Pleural.

Résultat texte OpenI :

| Modèle | AUC macro test | F1 macro test | Précision macro | Rappel macro | Loss test |
|---|---:|---:|---:|---:|---:|
| TF-IDF + MLP | 0.9685 | 0.4965 | 0.9310 | 0.3591 | 0.1148 |

Le téléchargement officiel des images PNG a été tenté, mais il a dépassé 30 minutes. Les images OpenI restent donc à télécharger/extracter si on veut entraîner `train_multimodal.py`.

Puis renseigner `openi.label_columns` dans `config.yaml` avant de lancer :

```bash
.venv\Scripts\python.exe -m src.training.train_text --config config.yaml
.venv\Scripts\python.exe -m src.training.train_multimodal --config config.yaml
```

## À dire si on rend maintenant

Le code et les livrables sont complets. Les résultats rapides prouvent que le pipeline tourne localement. En revanche, les métriques rapides ne doivent pas être interprétées comme performances finales, car elles sont obtenues sur un sous-échantillon CPU très réduit.
