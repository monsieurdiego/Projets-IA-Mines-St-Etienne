# Compte Rendu : Implémentation d'un Réseau de Neurones pour la Régression

## 1. Contexte et Objectifs

Ce projet porte sur le développement d'un réseau de neurones artificiel destiné à la prédiction de la perméabilité de matériaux composites. L'ensemble de données utilisé contient 120 configurations de microstructures caractérisées par quatre paramètres d'entrée et une valeur de perméabilité à prédire.

**Paramètres d'entrée (X)** :

- Fraction volumique de fibres (Vf)
- Rayon minimal des fibres (Rmin)
- Rayon maximal des fibres (Rmax)
- Paramètre de distribution spatiale (epsi)

**Variable cible (K)** : Perméabilité effective du matériau (γ)

## 2. Architecture Technique

### 2.1 Modules Développés

Le projet s'articule autour de deux modules Python principaux :

**`regression_ressources.py`** : bibliothèque d'utilitaires pour le traitement des données

- Lecture et importation de fichiers CSV multiples
- Génération d'indices polynomiaux pour régression étendue
- Division des données en ensembles d'apprentissage et de validation
- Fonctions de visualisation des résultats

**`nn_ressources.py`** : implémentation complète du réseau de neurones

- Initialisation des poids (uniforme et aléatoire)
- Propagation avant (forward propagation)
- Rétropropagation (backpropagation) avec calcul du gradient
- Normalisation des données dans l'intervalle [0,1]
- Versions optimisées pour le traitement matriciel

### 2.2 Prétraitement des Données

Une étape de normalisation min-max a été appliquée aux données d'entrée et de sortie pour ramener toutes les valeurs dans l'intervalle [0,1]. Cette transformation améliore la convergence de l'algorithme d'apprentissage.

Les données ont été subdivisées selon une proportion 80/20 :

- 80% pour l'ensemble d'apprentissage (96 échantillons)
- 20% pour l'ensemble de validation (24 échantillons)

La fonction de séparation inclut une option de seed fixe permettant de garantir la reproductibilité des résultats lors des expérimentations successives.

## 3. Implémentation du Réseau de Neurones

### 3.1 Fonction d'Activation

Le réseau utilise la fonction sigmoïde comme fonction d'activation :

$$\sigma(x) = \frac{1}{1 + e^{-\lambda x}}$$

où λ est un paramètre d'échelle contrôlant la pente de la fonction.

### 3.2 Propagation Avant

L'algorithme de propagation avant calcule la sortie du réseau pour un vecteur d'entrée donné. À chaque couche _s_ :

1. Calcul de la combinaison linéaire : $\alpha^{(s)} = W^{(s)} \cdot [\beta^{(s-1)}, 1]$
2. Application de la fonction d'activation : $\beta^{(s)} = \sigma(\alpha^{(s)})$

où le terme biais est implicitement intégré par l'ajout de 1 au vecteur β.

Deux implémentations ont été développées :

- `FwdPropagateNNSingle` : traite un vecteur à la fois
- `FwdPropagateNN` : traite un batch complet (plus efficace)

### 3.3 Rétropropagation

L'algorithme de rétropropagation calcule le gradient de la fonction de coût par rapport aux poids du réseau. Le gradient est obtenu par :

$$G^{(s)} = e^{(s)} \cdot (\beta^{(s-1)})^T$$

où $e^{(s)} = \delta^{(s)} \cdot \sigma'(\alpha^{(s)})$

La dérivée de la sigmoïde utilisée est :

$$\sigma'(x) = \frac{\lambda e^{-\lambda x}}{(1 + e^{-\lambda x})^2}$$

Deux versions de la rétropropagation ont été implémentées :

- `BackPropagateNNSingle` : traitement échantillon par échantillon
- `BackPropagateNN_Block` : traitement matriciel complet (performant)

### 3.4 Optimisation des Poids

Une ébauche d'algorithme d'optimisation par descente de gradient a été initiée. La mise à jour des poids suit la règle :

$$W^{(s)}_{i,j} \leftarrow W^{(s)}_{i,j} - \rho \cdot \frac{\partial L}{\partial W^{(s)}_{i,j}}$$

où ρ est le taux d'apprentissage.

## 4. Analyse Exploratoire des Données

Une exploration préliminaire des données a été réalisée pour visualiser la relation entre chaque paramètre d'entrée et la perméabilité. Quatre graphiques de dispersion ont été générés, montrant la perméabilité en fonction de chacun des quatre paramètres d'entrée.

Cette analyse permet d'identifier les tendances linéaires ou non-linéaires et d'évaluer qualitativement la complexité du problème de régression.

## 5. Comparaison avec TensorFlow/Keras

À titre de référence, une implémentation équivalente a été réalisée avec TensorFlow/Keras. Un réseau séquentiel avec une couche cachée de 2 neurones a été entraîné sur les mêmes données.

Configuration du modèle de référence :

- Couche d'entrée : 4 neurones
- Couche cachée : 2 neurones (activation sigmoïde)
- Couche de sortie : 1 neurone (activation sigmoïde)
- Optimiseur : Adam
- Fonction de coût : erreur quadratique moyenne
- Nombre d'époques : 5

Cette implémentation sert de benchmark pour valider les performances du réseau développé manuellement.

## 6. Points d'Amélioration Identifiés

### 6.1 Code Incomplet

Plusieurs fonctions présentes dans le notebook restent incomplètes ou contiennent des erreurs :

**Fonction `FwdPropagateNNSingle`** :

- Utilisation incorrecte de la variable `x` au lieu de `beta` dans la boucle
- Absence de gestion du paramètre `lastLayerLinear`
- Structure de retour incorrecte (manque les listes alpha et beta)

**Fonction `optimisationPoids`** :

- Variables non définies (`lambd`, `nlayers`)
- Structure de boucle d'optimisation non finalisée
- Absence de mécanisme de convergence ou d'arrêt

### 6.2 Suggestions Techniques

1. **Finaliser l'algorithme d'entraînement** : implémenter correctement la descente de gradient stochastique ou par batch
2. **Critère d'arrêt** : ajouter une condition basée sur la convergence du coût ou un nombre maximal d'itérations
3. **Validation croisée** : utiliser l'ensemble de validation pour évaluer la généralisation
4. **Régularisation** : envisager l'ajout de termes L1/L2 pour éviter le surapprentissage
5. **Métriques de performance** : calculer MSE, MAE, R² sur les ensembles d'apprentissage et de validation
6. **Visualisation de l'apprentissage** : tracer les courbes d'évolution du coût au fil des itérations

## 7. Synthèse

Le projet constitue une base solide pour l'implémentation d'un réseau de neurones à partir de zéro. Les fonctionnalités essentielles (propagation avant, rétropropagation, gestion des données) ont été développées et testées.

Les principaux acquis sont :

- Compréhension approfondie des mécanismes internes d'un réseau de neurones
- Implémentation efficace du traitement matriciel pour accélérer les calculs
- Mise en place d'une infrastructure modulaire et réutilisable

Les prochaines étapes consistent à finaliser la boucle d'entraînement et à valider les performances du modèle sur des données non vues lors de l'apprentissage.

---

_Document rédigé le 14 novembre 2025_
