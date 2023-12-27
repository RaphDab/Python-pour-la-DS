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

### B - Test du modèle sur des articles de presse récupérés via NewsAPI (Analyse_de_sentiments.ipynb)

  Dans cette partie, nous récupérons, nettoyons et analysons des articles de presse tirés du site Forbes.com et en relation avec l'actualité de Apple. Nous réalisons une série de statistiques descriptives sur ces articles afin d'en comprendre les spécificités. Enfin, nous testons notre modèle entraîné en A sur ces articles et nous comparons ses prédictions à celles d'un modèle Transformers extrêmement performant.

### C - Analyse de sentiments sur des posts extraits de Reddit (Post_Reddit.ipynb)

  Enfin, nous récupérons des articles Reddit en relation avec l'actualité de l'entreprise Apple et procédons à des étapes similaires à celles réalisées en B. Nous nettoyons ces articles et procédons à nouveau à des statistiques descriptives poussées. L'objectif est de diversifier les sources d'informations de notre algorithme de trading pour obtenir les résultats les plus justes possibles.


## Partie II : Entraînement et test d'un modèle de prédiction du cours de l'action de Apple

Dans cette deuxième grande partie, nous cherchons à entraîner un modèle à prédire le prix de l'action APPL. Pour ce faire, nous récupérons des données du site de Yahoo Finance et nous comparons les performances de 3 modèles : un modèle de **Naive Forecasting**, un modèle de **Moving Average** (deux modèles très simples) et un modèle plus compliqué à base de **LSTMs**. À nouveau, l'objectif n'est pas d'utiliser un modèle complexe à tout prix mais bien de comparer les performances de ces trois modèles et de garder celui qui est le plus fiable.


### Partie III : Création d'un algorithme de trading reprenant les deux parties précédentes

Cette ultime étape vise à regrouper le travail effectué dans les deux parties précédentes pour offrir à l'utilisateur une "interface" dédiée à notre algorithme de trading. Nous demandons à l'utilisateur de rentrer la date du jour puis d'exécuter notre fichier. À la fin de ce dernier, après avoir récupéré les articles de presse et les posts Reddit correspondants aux 48 heures avant cette date et avoir fait tourné notre modèle de prédiction du cours de l'action AAPL, nous indiquons à notre utilisateur s'il doit ou non acheter une action AAPL en ce jour.
