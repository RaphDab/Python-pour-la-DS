# Python-pour-la-DS
Algorithme de trading mélangeant prédiction de la valeur d'actions et analyse de sentiments sur des articles de nouvelles financières concernant l'entreprise Apple.

Ce projet se divise en trois grandes parties:
- entraînement et test d'un modèle de NLP
- entraînement et test d'un modèle de prédiction du cours de l'action de Apple
- création d'un algorithme de trading.

## Partie I : Entraînement et test d'un modèle de NLP
### A - Entraînement d'un modèle de NLP sur le dataset IMDb (Training.ipynb)

  L'idée est de comparer trois approches différentes pour construire un modèle d'analyse de sentiments de A à Z. On teste dans un premier temps deux approches plutôt classiques : une **régression logistique** et un **Random Forest** fine-tunés. Par la suite, on essaie une approche basée sur des réseaux neuronaux complexes avec les **LSTMs** (Long Short-Term Memory) particulièrement adaptés à ce genre de tâches.
  Sans à priori à première vue, nous souhaitons tester par nous-mêmes à quel point un modèle simple peut rivaliser avec des LSTMs même fine-tunés.
Nous sélectionnerons le modèle qui offre le plus de garanties pour la suite du projet.

**Récupération et traitement des données** : Récupération des données du dataset IMDb, nettoyage, ré-agencement et ajout de colonnes du fichier CSV pour entraîner les modèles de régression logistique et de Random Forest. Pour la partie LSTM, étapes de nettoyage du texte, tokenisation et padding des séquences.

**Analyse descriptive et représentation graphique** : 
- Analyse descriptive du dataset IMDb (répartition des labels, distribution de la longueur des critiques, mots les plus fréquents (wordcloud), recherche de corrélation entre longueur des critiques et sentiment exprimé)).
- Suivi des performances des trois modèles entraînés : Régression logistique (matrice de confusion), Random Forest (matrice de confusion, graphique de suivi des performances selon le nombre d'arbres, graphique mettant en lumière le lien entre le temps d'entraînement et le nombre d'arbres sélectionné), LSTM (analyse supplémentaire du dataset avant les étapes de tokenisation et padding, suivi de l'évolution de la métrique "accuracy" sur les 10 époques d'entraînement et analyse de celle-ci pour fine-tuning, confrontation des performances de chaque version du modèle).

**Modélisation** : Test et comparaison de trois modèles distincts 
- ***Régression logistique*** : modèle simple, adapté à des problèmes binaires et facilement interprétable; sert de modèle de référence dans l'entraînement des modèles suivants. 
- ***Random Forest*** : capable de capturer des dépendances non-linéaires entre les données, a tendance à bien généraliser à de nouvelles données.
- ***LSTMs***: parfaitement adaptés à la gestion de séquence de données et donc aux tâches de type NLP, leur mémoire à long terme leur permet d'interpréter des longues séquences de texte et d'en saisir les dépendances à longue échelle. Permet de comparer les performances d'un modèle assez complexe à celles d'une simple régression logistique.

### B - Test du modèle sur des articles de presse récupérés via NewsAPI (Analyse_de_sentiments.ipynb)

  Dans cette partie, nous récupérons, nettoyons et analysons des articles de presse tirés du site Forbes.com et en relation avec l'actualité de Apple. Nous réalisons une série de statistiques descriptives sur ces articles afin d'en comprendre les spécificités. Enfin, nous testons notre modèle entraîné en A sur ces articles et nous comparons ses prédictions à celles d'un modèle Transformers extrêmement performant.

**Récupération et traitement des données** : Utilisation de l’API NewsAPI pour récupérer des articles de presse en rapport avec Apple. Analyse de la structure HTML de leur site web pour repérer les balises contenant le cœur de l’article. Récupération des titres, descriptions et contenus des articles pour les préparer à être traités par un modèle d’analyse de sentiments.

**Analyse descriptive et représentation graphique** : Statistiques descriptives sur la longueur des articles et la longueur de leurs attributs (titre, description) et affichage d'un bubble chart pour comprendre les potentielles corrélations entre ces deux dernières. Affichage des mots les plus fréquents, mise en évidence de stopwords et suppression de ces derniers.

**Modélisation** : Test du modèle entraîné précédemment sur le corpus d'articles récupérés. Conclusion.



### C - Analyse de sentiments sur des posts extraits de Reddit (Post_Reddit.ipynb)

  Enfin, nous récupérons des articles Reddit en relation avec l'actualité de l'entreprise Apple et procédons à des étapes similaires à celles réalisées en B. Nous nettoyons ces articles et procédons à nouveau à des statistiques descriptives poussées. L'objectif est de diversifier les sources d'informations de notre algorithme de trading pour obtenir les résultats les plus justes possibles.

**Récupération et traitement des données** : Utilisation de l’API Reddit pour récupérer les posts récents en rapport avec l’entreprise Apple. Pré-traitement des données récupérées et nettoyage avant de les faire analyser par notre modèle d’analyse de sentiments.

**Analyse descriptive et représentation graphique** : 



## Partie II : Entraînement et test d'un modèle de prédiction du cours de l'action de Apple

Dans cette deuxième grande partie, nous cherchons à entraîner un modèle à prédire le prix de l'action APPL. Pour ce faire, nous récupérons des données du site de Yahoo Finance et nous comparons les performances de 3 modèles : un modèle de **Naive Forecasting**, un modèle de **Moving Average** (deux modèles très simples) et un modèle plus compliqué à base de **LSTMs**. À nouveau, l'objectif n'est pas d'utiliser un modèle complexe à tout prix mais bien de comparer les performances de ces trois modèles et de garder celui qui est le plus fiable.

**Récupération et traitement des données** : Récupération des données financières du cours de l’action Apple via Yahoo Finance, organisation des données récupérées pour créer des ensembles d’entraînement, de validation et de test pour entraîner un modèle LSTM.

**Analyse descriptive et représentation graphique** :

**Modélisation** :

## Partie III : Création d'un algorithme de trading reprenant les deux parties précédentes

Cette ultime étape vise à regrouper le travail effectué dans les deux parties précédentes pour offrir à l'utilisateur une "interface" dédiée à notre algorithme de trading. Nous demandons à l'utilisateur de rentrer la date du jour puis d'exécuter notre fichier. À la fin de ce dernier, après avoir récupéré les articles de presse et les posts Reddit correspondants aux 48 heures avant cette date et avoir fait tourné notre modèle de prédiction du cours de l'action AAPL, nous indiquons à notre utilisateur s'il doit ou non acheter une action AAPL en ce jour.
