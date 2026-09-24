# CNN from scratch vs Transfer Learning — Cats vs Dogs

## Titre & objectif du projet

Ce projet compare deux approches de classification d'images (chats vs chiens) :
1. Un **CNN entraîné from scratch** (4 blocs conv + BatchNorm + Dropout).
2. Un **modèle en Transfer Learning** basé sur **ResNet18** pré-entraîné sur ImageNet,
   avec backbone gelé et classifieur final ré-entraîné.

L'objectif est de mesurer l'impact du transfer learning sur la **convergence**, la
**performance** (accuracy, précision, recall) et la **robustesse**, à jeu de données égal.

## Environnement

```bash
pip install -r requirements.txt
```

ou avec conda :

```bash
conda env create -f environment.yml
conda activate cnn-catsdogs
```

Le notebook fonctionne aussi bien en local (avec GPU CUDA) que sur **Google Colab**
(cellules de montage Google Drive incluses, à décommenter).

## Organisation des données

- Jeu de données : [Cats vs Dogs (Kaggle)](https://www.kaggle.com/c/dogs-vs-cats), version
  déjà triée en `train/` et `test/` : [lien de téléchargement](https://s3.amazonaws.com/content.udacity-data.com/nd089/Cat_Dog_data.zip).
- Après téléchargement, dézipper de façon à obtenir :

```
Cat_Dog_data/
├─ train/
│  ├─ cat/
│  └─ dog/
└─ test/
   ├─ cat/
   └─ dog/
```

- Placer ce dossier `Cat_Dog_data/` à la racine du projet (ou adapter la variable
  `DATA_DIR` dans le notebook). **Ne pas** pousser ce dossier sur GitHub (voir `.gitignore`).

## Commandes pour entraîner

Tout se pilote depuis `notebook.ipynb`, section par section :

- **From scratch** (section 6) :
  - `EPOCHS_A = 10`, `BATCH_SIZE = 32`
  - Optimiseurs testés : `SGD` (`lr=0.01`, `momentum=0.9`, `StepLR` step=5, gamma=0.5) et
    `Adam` (`lr=1e-3`, `CosineAnnealingLR`)
  - Régularisation : `Dropout(0.4)` dans le classifieur, `BatchNorm2d` après chaque conv
  - Le meilleur des deux runs (selon `val_acc`) est conservé automatiquement.

- **Transfer learning** (section 7) :
  - Base : `ResNet18` (poids `IMAGENET1K_V1`), **backbone gelé**, seule la tête `fc`
    (avec `Dropout(0.4)`) est ré-entraînée.
  - `EPOCHS_B = 8`, mêmes optimiseurs testés (`SGD` et `Adam`) sur les paramètres
    entraînables uniquement.

Pour ajuster les hyperparamètres, modifier les constantes en haut de chaque cellule
d'expérience (`EPOCHS_*`, `LR_*`, `BATCH_SIZE`).

## Commandes pour évaluer / recharger le modèle

Les meilleurs modèles sont sauvegardés localement (non poussés sur GitHub) :

- `best_model_scratch.pt`
- `best_model_transfer.pt`

La section 10 du notebook les recharge et évalue sur le **jeu de test** :

```python
final_model_a = CNNFromScratch().to(device)
final_model_a.load_state_dict(torch.load("best_model_scratch.pt", map_location=device))
```

## Journalisation (bonus)

Les 4 runs d'entraînement (scratch-SGD, scratch-Adam, transfer-SGD, transfer-Adam) sont
loggés avec **TensorBoard** (inclus dans PyTorch, aucun compte externe requis) dans
`runs/<nom_du_run>/`. Pour visualiser :

```bash
tensorboard --logdir runs
```

ou, dans le notebook (Colab/Jupyter) :

```
%load_ext tensorboard
%tensorboard --logdir runs
```

Le dossier `runs/` n'est **pas** poussé sur GitHub (voir `.gitignore`).

## Résultats

Exécution sur GPU Colab (T4), dataset complet (18 000 train / 4 500 val / 2 500 test),
seed=42 :

| Modèle | Optimiseur retenu | Test Loss | Test Accuracy | Test Precision | Test Recall |
|---|---|---|---|---|---|
| CNN from scratch | Adam | 0.278 | 87.5 % | 86.5 % | 89.0 % |
| Transfer Learning (ResNet18) | Adam | 0.060 | 97.6 % | 98.5 % | 96.7 % |

Courbes loss / accuracy / précision / recall (train vs val, scratch vs transfer) :
voir sortie de la **section 8** du notebook (`plot_comparison`).

**Analyse :**

Les deux modèles apprennent, mais à des vitesses très différentes. Le CNN from scratch
démarre à 63-68 % d'accuracy en entraînement dès la première époque et progresse
lentement, pour atteindre environ 89 % de val accuracy après 10 époques complètes. Le
modèle en transfer learning, lui, démarre déjà à plus de 91 % d'accuracy train et 97 %
val accuracy **dès la première époque**, et se stabilise autour de 97-98 % en quelques
époques seulement — il converge donc nettement plus vite.

Sur le jeu de test (jamais vu pendant l'entraînement), l'écart se confirme : 87.5 %
d'accuracy pour le CNN from scratch contre 97.6 % pour le transfer learning, avec une
loss de test presque 5 fois plus faible pour ce dernier (0.060 contre 0.278). La precision
et le recall suivent la même tendance, ce qui indique que le transfer learning n'est pas
seulement plus précis mais aussi plus équilibré entre les deux classes.

Cet écart s'explique par le fait que ResNet18 a déjà appris, sur ImageNet, des
représentations visuelles génériques (contours, textures, formes) très pertinentes pour
distinguer chats et chiens. Le CNN from scratch doit au contraire apprendre ces
représentations de zéro à partir des 18 000 images d'entraînement, ce qui demande plus
de données, plus d'époques et est plus sensible au surapprentissage.

## Limites & pistes d'amélioration

- Jeu de données relativement petit pour un entraînement from scratch robuste.
- Backbone ResNet18 entièrement gelé : un **fine-tuning partiel** (dégeler les dernières
  couches convolutionnelles) pourrait améliorer encore les résultats.
- Pas de recherche exhaustive d'hyperparamètres (grid/random search) — seulement une
  comparaison SGD vs Adam à learning rate fixe par défaut raisonnable.
- Pas de test sur des images hors distribution (robustesse au bruit, autres races, etc.).

## Reproductibilité

Une seed fixe (`SEED = 42`) est appliquée à Python, NumPy, PyTorch et CUDA en tout début
de notebook (`set_seed()`), et réappliquée avant chaque run d'entraînement.
