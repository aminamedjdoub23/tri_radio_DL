# Données

## ChestMNIST

ChestMNIST est téléchargé automatiquement par `medmnist` dans `data/raw`.

Le dataset est utilisé pour la partie obligatoire de classification supervisée image :

- classification multi-label sur 14 pathologies ;
- split officiel `train`, `val`, `test` fourni par MedMNIST ;
- sortie du modèle avec 14 logits ;
- activation sigmoid uniquement pour l'inférence et les métriques ;
- loss `BCEWithLogitsLoss`.

## OpenI

OpenI n'est pas téléchargé automatiquement par ce projet. Il faut préparer localement un CSV, par exemple :

```text
data/openi/openi_prepared.csv
```

Colonnes minimales attendues :

- `image_path` : chemin vers l'image ;
- `report` : texte du compte-rendu ;
- colonnes de labels binaires choisies pour la preuve de concept.

Dans `config.yaml`, renseigner :

```yaml
paths:
  openi_csv: data/openi/openi_prepared.csv
openi:
  label_columns: ["Atelectasis", "Cardiomegaly"]
```

La preuve de concept OpenI sert à comparer image seule, texte seul et fusion image + texte. Elle ne remplace pas ChestMNIST pour la partie obligatoire.
