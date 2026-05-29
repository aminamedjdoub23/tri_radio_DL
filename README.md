# Projet Deep Learning - Tri radiologique thoracique

Base de projet PyTorch pour un TD/projet de classification multi-label sur radiographies thoraciques.

Le choix principal est volontairement simple : ChestMNIST est utilisé pour la classification supervisée image, OpenI est prévu pour une preuve de concept image + texte, MLflow trace les expériences, et Streamlit sert de démonstrateur local. L'objectif est de produire un travail clair et défendable, pas un système clinique.

## Choix techniques

- **ChestMNIST / MedMNIST** : dataset imposé par la consigne, déjà séparé en train/validation/test, ce qui limite le risque de fuite de données.
- **Multi-label avec sigmoid + BCEWithLogitsLoss** : chaque pathologie est prédite indépendamment, ce qui correspond au fait qu'une radiographie peut avoir plusieurs labels.
- **CNN simple** : baseline entraînée depuis zéro pour mesurer ce qu'un modèle léger apprend sans connaissances externes.
- **ResNet pré-entraîné** : transfert d'apprentissage simple et classique pour comparer avec une architecture ayant appris des motifs visuels généraux.
- **ViT via timm** : modèle à attention demandé par la consigne, utilisé prudemment car ChestMNIST est petit.
- **Autoencodeur convolutionnel** : score d'anomalie par erreur de reconstruction. Le seuil proposé est le percentile 95 sur validation, simple à expliquer.
- **OpenI pour la multimodalité** : plus faisable que MIMIC-CXR pour un binôme étudiant. Les scripts attendent un CSV préparé localement.
- **TF-IDF + MLP texte** : solution légère, reproductible et suffisante pour une preuve de concept.
- **Fusion intermédiaire** : concaténation des embeddings image et texte, facile à justifier et à comparer.

## Installation

Recommandation : utiliser Python 3.11 ou 3.12. Le projet dépend de PyTorch/torchvision, et certaines versions très récentes de Python peuvent ne pas encore avoir toutes les roues disponibles selon la machine.

```bash
cd project
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Avec GPU CUDA, installer au besoin une version PyTorch adaptée depuis le site officiel PyTorch avant les autres dépendances.

Vérification rapide :

```bash
python -m compileall src
python -c "import torch, torchvision, timm, medmnist, mlflow, streamlit; print('installation ok')"
```

## Entraînements

Les commandes ci-dessous ne prétendent pas produire des résultats avant exécution réelle. Les métriques seront écrites dans MLflow et les meilleurs modèles dans `outputs/`.

```bash
python -m src.training.train_supervised --model simple_cnn --config config.yaml
python -m src.training.train_supervised --model transfer --config config.yaml
python -m src.training.train_supervised --model vit --config config.yaml
python -m src.training.train_autoencoder --config config.yaml
```

Les runs ChestMNIST téléchargent les données via MedMNIST dans `data/raw`. Selon la machine, le modèle ViT peut être plus long que les deux CNN.

Pour OpenI, préparer d'abord un CSV local avec les colonnes indiquées dans `data/README.md`, puis :

```bash
python -m src.training.train_text --config config.yaml
python -m src.training.train_multimodal --config config.yaml
```

Si OpenI n'est pas encore préparé, laisser `openi.label_columns: []` dans `config.yaml` et ne lancer que la partie ChestMNIST.

## MLflow

```bash
mlflow ui --backend-store-uri mlruns
```

Puis ouvrir l'URL affichée par MLflow, généralement `http://127.0.0.1:5000`.

## Démonstrateur Streamlit

Après avoir entraîné au moins un modèle supervisé et l'autoencodeur :

```bash
streamlit run src/app/streamlit_app.py
```

Le démonstrateur affiche les probabilités par pathologie et un score d'anomalie. Il indique explicitement que ce n'est pas un dispositif médical.

## Fichiers utiles pour le rendu

- `notebooks/01_eda_chestmnist.ipynb` : analyse exploratoire à exécuter après installation.
- `notebooks/02_results_analysis.ipynb` : récupération des métriques MLflow après entraînement.
- `report/rapport.md` : brouillon de rapport structuré, à compléter avec les vrais résultats.
- `report/conformite_consigne.md` : vérification point par point avec la consigne.
- `report/rapport_plan.md` : plan court si vous voulez garder une version synthétique.

## Limites connues

- ChestMNIST est une version réduite et standardisée de radiographies, moins réaliste qu'un flux hospitalier.
- Les labels multi-label sont déséquilibrés : les métriques macro et par classe sont plus informatives que l'accuracy.
- Le ViT peut être moins performant si les données ou le temps d'entraînement sont limités.
- Le score AE détecte une reconstruction inhabituelle, pas une pathologie clinique certaine.
- OpenI est adapté à une preuve de concept, mais trop petit pour conclure solidement sur l'apport réel du texte.
