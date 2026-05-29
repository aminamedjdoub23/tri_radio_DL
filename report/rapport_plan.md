# Plan de rapport

## 1. Problème

Présenter le tri radiologique comme une aide à la priorisation, pas comme un diagnostic automatique. Expliquer que la tâche IA est une classification multi-label de pathologies thoraciques, complétée par une détection de cas atypiques.

## 2. Données

Décrire ChestMNIST/ChestMNIST+ comme dataset principal obligatoire. Justifier OpenI pour la preuve de concept multimodale car il contient images et comptes-rendus, contrairement à ChestMNIST.

## 3. Analyse exploratoire

Analyser la distribution des 14 labels, le déséquilibre, quelques exemples d'images, et les co-occurrences. Pour OpenI, décrire brièvement la longueur des rapports et les labels disponibles.

## 4. Préparation

Présenter resize, normalisation, augmentations simples, encodage multi-label et split train/validation/test. Expliquer que le split officiel MedMNIST est conservé pour éviter des choix arbitraires.

## 5. Modélisation supervisée

Comparer :

- CNN simple entraîné depuis zéro ;
- ResNet pré-entraîné en transfer learning ;
- ViT compact.

Justifier sigmoid + BCE pour le multi-label.

## 6. Détection d'anomalies

Décrire l'autoencodeur convolutionnel, l'erreur MSE de reconstruction et le seuil au percentile 95 de validation. Expliquer que ce score indique une image atypique pour le modèle, pas une anomalie clinique certaine.

## 7. Modélisation multimodale

Présenter OpenI, l'encodeur image CNN, l'encodeur texte TF-IDF + MLP et la fusion intermédiaire par concaténation. Comparer image seule, texte seul et multimodal.

## 8. Évaluation

Rapporter AUC macro, AUC par classe, F1 macro, précision, rappel et loss validation. Ajouter ROC ou PR si le temps le permet.

## 9. Tracking MLflow

Montrer l'organisation des runs, les paramètres, métriques, figures, artefacts et checkpoints. Relier clairement le meilleur modèle au démonstrateur.

## 10. Démonstrateur

Décrire Streamlit : upload d'image, probabilités par pathologie, score d'anomalie, texte optionnel. Ajouter l'avertissement que ce n'est pas un outil médical.

## 11. Analyse critique

Discuter déséquilibre, résolution limitée, généralisation, coût calculatoire, limites OpenI, limites de l'AE et seuil choisi.

## 12. Conclusion et perspectives

Résumer ce qui fonctionne, ce qui reste fragile, et proposer des pistes : meilleure calibration, modèles plus robustes, validation externe, gestion patient-level si identifiants disponibles.

