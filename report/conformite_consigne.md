# Vérification de conformité à la consigne

Ce document relie les exigences du sujet aux fichiers du projet.

## Synthèse

Le projet est cohérent avec la consigne : il couvre la classification multi-label ChestMNIST, trois architectures image, la détection d'anomalies par autoencodeur, une preuve de concept multimodale OpenI, MLflow, Streamlit, un pipeline reproductible et un rapport structuré.

Des résultats réellement obtenus localement existent maintenant pour :

- les 3 runs rapides ChestMNIST ;
- un run CPU intermédiaire ResNet18 ;
- un run CPU intermédiaire autoencodeur ;
- un run OpenI texte seul.
- un run OpenI image seule + multimodal ;
- un run final GPU pour simple CNN, transfer learning et ViT.

La comparaison OpenI image seule / multimodal image + texte a été exécutée localement après téléchargement des images PNG officielles.

Des captures MLflow et Streamlit sont ajoutées en annexes du rapport.

## Correspondance exigence / implémentation

| Exigence                                                     | Statut                | Fichiers                                                         |
| ------------------------------------------------------------ | --------------------- | ---------------------------------------------------------------- |
| ChestMNIST / ChestMNIST+ obligatoire                         | Couvert               | `src/data/chestmnist_dataset.py`, `config.yaml`                  |
| Classification multi-label 14 pathologies                    | Couvert               | `src/models/*`, `src/training/train_supervised.py`               |
| Sigmoïde par classe + BCE                                    | Couvert               | `src/training/train_supervised.py`, `src/training/evaluate.py`   |
| CNN simple depuis zéro                                       | Couvert               | `src/models/simple_cnn.py`                                       |
| CNN pré-entraîné transfer learning                           | Couvert               | `src/models/transfer_model.py`                                   |
| ViT ou hybride CNN/Transformer                               | Couvert               | `src/models/vit_model.py`                                        |
| AE ou VAE pour anomalie                                      | Couvert               | `src/models/autoencoder.py`, `src/training/train_autoencoder.py` |
| Score d'anomalie et seuil justifié                           | Couvert               | seuil p95 validation dans `train_autoencoder.py`                 |
| Exemples de reconstructions AE                               | Couvert               | `src/utils/plots.py`, artefact MLflow                            |
| Multimodal image + texte                                     | Couvert               | `src/training/train_text.py`, `src/training/train_multimodal.py` |
| Comparer image seule / texte seul / multimodal               | Couvert               | `train_text.py`, `train_multimodal.py`                           |
| Fusion justifiée                                             | Couvert               | `report/rapport.md`                                              |
| MLflow obligatoire                                           | Couvert               | scripts dans `src/training/`                                     |
| Choix d'optimisation et régularisation documentés            | Couvert               | section Préparation dans `report/rapport.md`                     |
| Paramètres, métriques, artefacts, figures, meilleurs modèles | Couvert               | MLflow dans les scripts d'entraînement                           |
| Démonstrateur applicatif                                     | Couvert               | `src/app/streamlit_app.py`                                       |
| Upload image, prédictions, score anomalie                    | Couvert               | `streamlit_app.py`                                               |
| Texte complémentaire si multimodal disponible                | Couvert               | `streamlit_app.py`                                               |
| Cohérence run MLflow / modèle exposé                         | Couvert               | section Démonstrateur dans `report/rapport.md`                   |
| Train/validation/test propre                                 | Couvert               | splits MedMNIST et splits OpenI dans les scripts                 |
| Seed fixe                                                    | Couvert               | `src/utils/seed.py`                                              |
| Sauvegarde du meilleur modèle                                | Couvert               | scripts d'entraînement                                           |
| Rapport structuré selon les sections imposées                | Couvert               | `report/rapport.md`                                              |
| README clair                                                 | Couvert               | `README.md`                                                      |
| Documentation matériel et temps                              | Partiellement couvert | tableau dans `report/rapport.md`                                 |

## Points à compléter après exécution

Ces éléments dépendent d'un entraînement réel et ne doivent pas être inventés :

- mesures plus fines de temps si un rendu très détaillé est attendu ;
- exécuter ou capturer les notebooks si l'enseignant le demande explicitement.

## Commentaire sur OpenI

La consigne demande une composante multimodale si les données sont disponibles. Le projet prévoit OpenI avec un CSV préparé localement, et ce CSV a maintenant été généré sur cette machine à partir des rapports officiels NLM/OpenI. Les images PNG OpenI ont été téléchargées et la preuve de concept image+texte a été exécutée localement.
