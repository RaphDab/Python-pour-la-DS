# **Titre: Algorithme de trading sur l'action AAPL basé sur de la prédiction de séries temporelles et du NLP**

**Problématique : Quelle est l'efficacité relative des réseaux de neurones profonds par rapport aux méthodes d'apprentissage machine traditionnelles et à quel point leurs prédictions sont-elles assez satisfaisantes pour les intégrer à notre algorithme de trading ?**

Ce projet se divise en trois grandes parties:
- entraînement et test d'un modèle de NLP
- entraînement et test d'un modèle de prédiction du cours de l'action de Apple
- création d'un algorithme de trading.

## Partie I : Entraînement et test d'un modèle de NLP

L'objectif général de cette partie est de récupérer diverses sources d'information parlant des actaulités récentes de Apple, puis à l'aide du meilleur modèle de NLP possible, en déduire si les nouvelles autour de Apple sont bonnes ou non. Cela donnerait un premier feu vert pour l'algorithme de trading.

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

**Pistes d'amélioration** : Dans cette partie, nous aurions pu davantage fine-tuner nos paramètres pour chacun des modèles entraînés. Nous avons fait le choix de ne pas modifier outre mesure les hyper-paramètres de la régression logistique afin de montrer qu’un modèle simple pouvait être très efficace ici. Nous aurions également pu essayer d’entraîner davantage le modèle de Random Forest mais les résultats peu satisfaisants de celui-ci à premier abord ne laisser pas présager d’amélioration significative possible. Toutefois, en ce qui concerne la partie LSTM, il aurait été nécessaire de passer plus de temps sur la réduction de l’overfitting : la modification du taux de dropout, de la taille du batch-size ou encore l’introduction d’une régularisation L1 ou L2 auraient pu bénéficier à notre modèle.

### B - Test du modèle sur des articles de presse récupérés via NewsAPI (PressArticles.ipynb)

  Dans cette partie, nous récupérons, nettoyons et analysons des articles de presse tirés du site Forbes.com et en relation avec l'actualité de Apple. Nous réalisons une série de statistiques descriptives sur ces articles afin d'en comprendre les spécificités. Enfin, nous testons notre modèle entraîné en A sur ces articles et nous comparons ses prédictions à celles d'un modèle Transformers extrêmement performant.

**Récupération et traitement des données** : Utilisation de l’API NewsAPI pour récupérer des articles de presse en rapport avec Apple. Analyse de la structure HTML de leur site web pour repérer les balises contenant le cœur de l’article. Récupération des titres, descriptions et contenus des articles pour les préparer à être traités par un modèle d’analyse de sentiments.

**Analyse descriptive et représentation graphique** : Statistiques descriptives sur la longueur des articles et la longueur de leurs attributs (titre, description) et affichage d'un bubble chart pour comprendre les potentielles corrélations entre ces deux dernières. Affichage des mots les plus fréquents, mise en évidence de stopwords et suppression de ces derniers.

**Modélisation** : Test du modèle entraîné précédemment sur le corpus d'articles récupérés. Conclusion.



### C - Analyse de sentiments sur des posts extraits de Reddit (RedditPosts.ipynb)

  Enfin, nous récupérons des articles Reddit en relation avec l'actualité de l'entreprise Apple et procédons à des étapes similaires à celles réalisées en B. Nous nettoyons ces articles et procédons à nouveau à des statistiques descriptives poussées. L'objectif est de diversifier les sources d'informations de notre algorithme de trading pour obtenir les résultats les plus justes possibles.

**Récupération et traitement des données** : Utilisation de l’API Reddit pour récupérer les posts récents en rapport avec l’entreprise Apple puis pré-traitement des données récupérées et nettoyage avant de les faire analyser par notre modèle d’analyse de sentiments.
Mise en évidence, par un nuage de mots, et suppression de certains posts  non exploitables. Récupération des comments de chaque posts et nettoyage des comments. 

**Analyse descriptive et représentation graphique** : Représentation graphique du nombre cumulé de post écrits dans le temps et lien avec le cours d'Apple. Calcul du coefficient de correlation entre le nombre de posts publiés sur Reddit et le cours d'Apple. Puis création d'un dataframe regroupant les auteurs des posts et représentation graphique de la contribution de chaque auteur. Enfin, étude de statistiques descriptives relatives à l'occurence des mots utilisés dans les posts, puis représentation graphique sur les mots les plus utilisés du dataframe. Mise en évidence des stopwords et suppression de ces derniers. Test de notre modèle LSTM.


## Partie II : Entraînement et test d'un modèle de prédiction du cours de l'action de Apple (StockPredictions)

Dans cette deuxième grande partie, nous cherchons à entraîner un modèle à prédire le prix de l'action AAPL. L'objectif final étant, grâce au meilleure modèle possible, prédire la valeur du lendemain de l'action (par rapport à la date d'execution du programme), et ainsi voir si elle va augmenter ou diminuer.
Si la valeur augmente, alors il s'agit du deuxième feu vert pour l'achat d'action, en plus des "bonnes nouvelles" dans la presse et sur Reddit concernant Apple.

Pour ce faire, nous récupérons des données du site de Yahoo Finance et nous comparons les performances de 3 modèles : un modèle de **Naive Forecasting**, un modèle de **Moving Average** (deux modèles très simples) et un modèle plus compliqué à base de **LSTMs**. À nouveau, l'objectif n'est pas d'utiliser un modèle complexe à tout prix mais bien de comparer les performances de ces trois modèles et de garder celui qui est le plus fiable.

**Récupération et traitement des données** : Récupération des données financières du cours de l’action Apple via Yahoo Finance, nettoyage si nécessaire, organisation des données récupérées pour créer des ensembles d’entraînement, de validation et de test pour entraîner un modèle LSTM.

**Analyse descriptive et représentation graphique** : Représentation des fluctuations du cours de l'action AAPL (à l'ouverture par rapport à la fermeture du marché, volume de transactions ou encore en comparant les valeurs les plus hautes et les plus basses atteintes dans la journée). 
Comparaison des performances des différents modèles utilisés entre eux et par rapport aux variations réelles du cours de l'action. Analyse graphiques des performances de notre modèle LSTM.

**Modélisation** : Comparaison des performances de trois modèles différents : moving average, naive forecasting et LSTM.

**Pistes d'amélioration :** Dans cette partie, il aurait pu être intéressant de travailler un peu plus en profondeur sur un modèle de machine learning, en évitant de tomber dans de la redite de la première partie, afin que le modèle à utiliser à la fin soit autre chose que le naive forecasting. Bien que notre travail cherche aussi à montrer que parfois les modèles simples sont les plus efficaces. On aurait pu également chercher à diversifier nos sources sur les actions, ou sinon chercher à les obtenir de manière plus complexe avec du webscraping à la place de l'utilisation de la bibliothèque toute faite yfinance. Enfin, nous aurions pu complexifier notre algorithme de trading final en incluant à notre prédiction du cours de l'action Apple d'autres paramètres tels que les variations des actions de concurrrents directs ou encore des variations du cours du NASDAQ par exemple.

## Partie III : Création d'un algorithme de trading reprenant les deux parties précédentes (Trading_Algorithm)

Cette ultime étape vise à regrouper le travail effectué dans les deux parties précédentes pour offrir à l'utilisateur une "interface" dédiée à notre algorithme de trading. Nous demandons à l'utilisateur de rentrer la date du jour puis d'exécuter notre fichier. À la fin de ce dernier, après avoir récupéré les articles de presse et les posts Reddit correspondants aux 48 heures avant cette date et avoir fait tourné notre modèle de prédiction du cours de l'action AAPL, nous indiquons à notre utilisateur s'il doit ou non acheter une action AAPL en ce jour.
