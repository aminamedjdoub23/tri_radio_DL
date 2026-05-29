# Rapport - Système d'aide au tri radiologique thoracique

## 1. Problème

Le projet porte sur un système d'aide au tri radiologique à partir de radiographies thoraciques. L'objectif n'est pas de produire un diagnostic automatique, mais de prioriser ou signaler des cas potentiellement importants dans un cadre pédagogique.

La tâche principale est une classification multi-label : une même radiographie peut être associée à plusieurs pathologies. Le modèle produit donc 14 scores indépendants, un par label, avec une activation sigmoïde en sortie pour l'inférence. Pendant l'entraînement, la fonction de coût utilisée est `BCEWithLogitsLoss`, car elle combine de façon stable les logits et la binary cross entropy.

## 2. Données

Le dataset principal est ChestMNIST, fourni par MedMNIST. Ce choix respecte la contrainte du sujet et évite de construire manuellement un split, puisque MedMNIST fournit déjà des ensembles train, validation et test.

ChestMNIST contient des radiographies thoraciques en niveaux de gris et 14 labels de pathologies. Les labels sont déséquilibrés, ce qui rend l'accuracy peu informative. Les métriques retenues sont donc l'AUC macro, l'AUC par classe, le F1-score macro, la précision macro, le rappel macro et la loss de validation.

Pour la partie multimodale, le projet prévoit OpenI. Ce choix est plus réaliste pour un binôme étudiant que MIMIC-CXR, car OpenI est plus simple à préparer localement. Cette partie est une preuve de concept : elle ne remplace pas ChestMNIST pour la classification supervisée obligatoire.

## 3. Analyse exploratoire

L'analyse exploratoire prévue dans `notebooks/01_eda_chestmnist.ipynb` vérifie :

- le nombre d'images dans les splits train, validation et test ;
- la prévalence des 14 labels ;
- quelques exemples de radiographies ;
- le déséquilibre entre classes.

Les observations attendues sont un déséquilibre marqué et des co-occurrences possibles entre pathologies. Ces éléments justifient l'utilisation de métriques macro et par classe.

## 4. Préparation

Les images ChestMNIST sont redimensionnées à la taille définie dans `config.yaml`. Les modèles supervisés utilisent une normalisation simple et des augmentations légères sur le train : flip horizontal et petite rotation. Ces augmentations restent volontairement limitées pour ne pas transformer excessivement des images médicales.

Le split officiel MedMNIST est conservé pour limiter les choix arbitraires et réduire le risque de fuite de données. La seed est fixée dans tous les scripts pour améliorer la reproductibilité.

Configuration et contraintes de calcul à compléter après exécution :

| Élément | Valeur |
|---|---|
| Machine utilisée | À compléter |
| CPU | À compléter |
| GPU | À compléter |
| RAM | À compléter |
| Version Python | À compléter |
| Version PyTorch | À compléter |
| Temps CNN simple | À compléter |
| Temps ResNet18 | À compléter |
| Temps ViT | À compléter |
| Temps autoencodeur | À compléter |

Les choix d'optimisation restent simples : AdamW pour les modèles supervisés, Adam pour l'autoencodeur, batch size modéré, dropout dans les classifieurs et weight decay pour limiter le surapprentissage. Le meilleur modèle est sauvegardé selon la métrique de validation. Aucun scheduler complexe n'est imposé, car l'objectif est de garder un pipeline lisible et reproductible.

## 5. Modélisation supervisée image

Trois architectures sont comparées :

- CNN simple entraîné depuis zéro : baseline légère, facile à comprendre et rapide à entraîner.
- ResNet18 pré-entraîné : transfert d'apprentissage classique, utile pour comparer avec un modèle ayant déjà appris des motifs visuels généraux.
- ViT compact via `timm` : modèle à attention demandé par la consigne, choisi en version petite pour rester faisable.

Les trois modèles produisent 14 logits. La sigmoïde est appliquée uniquement pour les métriques et l'affichage des probabilités, pas avant la loss.

Tableau à compléter après entraînement :

| Modèle | AUC macro test | F1 macro test | Précision macro | Rappel macro | Commentaire |
|---|---:|---:|---:|---:|---|
| CNN simple | À compléter | À compléter | À compléter | À compléter | Baseline |
| ResNet18 | À compléter | À compléter | À compléter | À compléter | Transfert |
| ViT tiny | À compléter | À compléter | À compléter | À compléter | Attention |

## 6. Détection d'anomalies

La détection d'anomalies est réalisée avec un autoencodeur convolutionnel. Le modèle apprend à reconstruire des images considérées comme normales, définies ici comme les radiographies sans label positif. Cette stratégie est simple à expliquer : l'AE apprend une reconstruction de cas sans pathologie annotée, puis une image mal reconstruite est considérée comme atypique pour le modèle.

Le score d'anomalie est l'erreur moyenne de reconstruction MSE. Le seuil est fixé au percentile 95 des erreurs sur la validation normale. Ce seuil est simple, reproductible et défendable, mais il ne correspond pas à une validation clinique.

Le script sauvegarde aussi une figure d'exemples original/reconstruction dans MLflow. Elle sert à vérifier visuellement que l'autoencodeur apprend une reconstruction plausible et à discuter les limites du score.

Tableau à compléter :

| Métrique AE | Valeur |
|---|---:|
| MSE moyenne validation normale | À compléter |
| Seuil percentile 95 | À compléter |
| MSE moyenne test normal | À compléter |
| MSE moyenne test complet | À compléter |
| Taux atypique sur test complet | À compléter |

Limite importante : un score élevé indique une reconstruction inhabituelle, pas une pathologie certaine. À l'inverse, une image pathologique peut parfois être bien reconstruite.

## 7. Modélisation multimodale

La preuve de concept multimodale utilise OpenI avec un CSV préparé localement. Le projet compare :

- image seule : encodeur ResNet18 puis classifieur ;
- texte seul : TF-IDF puis MLP ;
- fusion multimodale : concaténation des embeddings image et texte.

La fusion intermédiaire est choisie parce qu'elle est simple, claire et suffisante pour un projet étudiant. Elle permet de tester si le compte-rendu apporte une information complémentaire à l'image.

Tableau à compléter si OpenI est préparé :

| Modèle OpenI | AUC macro test | F1 macro test | Commentaire |
|---|---:|---:|---|
| Image seule | À compléter | À compléter | Baseline visuelle |
| Texte seul | À compléter | À compléter | Baseline texte |
| Multimodal | À compléter | À compléter | Fusion image + texte |

## 8. Évaluation

Les métriques principales sont :

- AUC macro : performance globale robuste au seuil ;
- AUC par classe : identification des pathologies difficiles ;
- F1 macro : équilibre précision/rappel en contexte déséquilibré ;
- précision et rappel macro : analyse des faux positifs et faux négatifs ;
- loss validation : suivi de l'apprentissage.

Les courbes ROC sont sauvegardées comme artefacts MLflow pour la partie ChestMNIST supervisée.

## 9. Tracking MLflow

MLflow est utilisé pour tracer :

- les paramètres des runs ;
- les métriques train, validation et test ;
- les AUC par classe pour ChestMNIST ;
- les figures ;
- les checkpoints des meilleurs modèles ;
- le fichier de configuration.

Après entraînement, l'interface MLflow se lance avec :

```bash
mlflow ui --backend-store-uri mlruns
```

## 10. Démonstrateur

Le démonstrateur Streamlit permet :

- d'uploader une radiographie ;
- d'afficher les probabilités par pathologie ;
- d'afficher un score d'anomalie si l'autoencodeur est disponible ;
- de saisir un texte optionnel et de lancer la prédiction image + texte si le checkpoint multimodal OpenI et le vectorizer TF-IDF sont disponibles.

L'application affiche clairement qu'il s'agit d'un prototype pédagogique et non d'un outil médical réel.

## 11. Analyse critique

Les principales limites sont :

- ChestMNIST est une version réduite et standardisée, moins réaliste qu'un flux hospitalier ;
- les labels sont déséquilibrés ;
- les performances peuvent dépendre fortement du temps d'entraînement et du matériel ;
- le ViT peut être fragile si les données ou les epochs sont limités ;
- l'AE ne détecte pas directement une pathologie clinique ;
- OpenI est adapté à une preuve de concept, mais trop limité pour conclure fortement sur l'intérêt de la multimodalité.

## 12. Conclusion et perspectives

Le projet propose une chaîne complète et défendable : classification multi-label image, comparaison de trois familles de modèles, détection d'images atypiques, preuve de concept multimodale, suivi MLflow et démonstrateur Streamlit.

Les perspectives possibles sont une meilleure calibration des seuils, une validation externe, une analyse par classe plus poussée, et une préparation plus rigoureuse d'un dataset multimodal avec identifiants patients pour éviter toute fuite.
