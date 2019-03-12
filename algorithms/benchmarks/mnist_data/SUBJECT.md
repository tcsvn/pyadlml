# TP Chaînes de Markov Cachées

Réda DEHAK
reda.dehak@lrde.epita.fr

Le but de ce TP est de manipuler des chaînes de markov cachées. Nous allons
utiliser la base de données de chiffre manuscrit online pendigits. Cette base
représente les chiffres 0-9 en écriture manuscrite. Un compte rendu est
à rendre pour le dimanche 12/6/2016 à 23h59.

Vous pouvez télécharger directement tous les données depuis ici :
https://www.lrde.epita.fr/~reda/MLEA/TP5/tp5.tgz

1. Vous trouverez dans le répertoire le script loadUnipenData pour le
   chargement des données au format UNIPEN.

2. Afficher quelques exemples pour vérifier le script de chargement.

3. Dans ce TP, nous allons nous limiter à l'utilisation de la toolbox MATLAB
   pour les HMMs, un descriptif des fonctions fournies est disponible à cette
   adresse. Cette toolbox implémente uniquement les HMMs discret. Pour cette
   raison, nous allons commencer par quantifier les données 2D en 256 valeurs
   différentes :

        1. Normaliser tous les tracés par rapport à la moyenne et la variance
           du tracé (Attention aux valeurs PEN_UP et PEN_DOWN).

        2. Utiliser la fonction kmeans sur l'ensemble des données
           d'apprentissage pour fabriquer un codebook de 256 valeurs. Attention
           aux valeurs PEN_UP et PEN_DOWN.

        3. Coder la base d'apprentissage et de test avec ce codebook.

4. En utilisant les fonctions de la toolbox matlab, entraîner un hmm par classe
   et tester les résultats de la classification de la base de test.

5. Essayer différentes configurations.

6. Comparer vos performance avec l'algorithme de la DTW.
   Quelle sont les avantages d'utilisation d'un HMM?

