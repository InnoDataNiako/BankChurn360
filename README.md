# Churn_Client_Banque

**Churn_Client_Banque** est une application **Streamlit** conçue pour fournir une **vue 360° sur la clientèle d'une institution financière**. L'application permet de détecter et de prédire le **churn** (désabonnement) des clients, d'analyser les facteurs influençant ce churn et de segmenter la clientèle selon différents critères : comportementale, valeur client et risque crédit.

---

## Fonctionnalités principales

1. **Dashboard global**
   - Visualisation des KPIs clés : nombre de clients, taux de churn, CLV médian, exposition au risque.
   - Analyse des facteurs d’influence sur le churn via un graphique de type barres horizontales.
   - Graphiques interactifs avec Plotly pour explorer les données.

2. **Segmentation client**
   - Segmentation basée sur différents types : comportementale, valeur client, risque crédit.
   - Visualisation des segments avec des graphiques interactifs.

3. **Prédiction du churn**
   - Modèle prédictif basé sur **Random Forest**.
   - Prédiction des probabilités de churn pour chaque client.
   - Tableau interactif avec mise en évidence des clients à risque élevé.
   - Filtres avancés pour explorer les données selon la probabilité de churn et le CLV.

---

## Technologies et librairies utilisées

- **Python 3.12**
- **Streamlit** pour le front-end interactif
- **Pandas** et **NumPy** pour la manipulation des données
- **Scikit-learn** pour le machine learning (Random Forest, KMeans)
- **Plotly** pour les visualisations interactives
- **Matplotlib** (pour le dégradé de couleurs dans les tables avec `background_gradient`)

---

## Installation

1. Cloner le dépôt GitHub :

```bash
git clone https://github.com/InnoDataNiako/BankChurn360.git
cd Churn_Client_Banque
````

2. Créer un environnement virtuel et l’activer :

```bash
python -m venv env
# Windows
env\Scripts\activate
# macOS / Linux
source env/bin/activate
```

3. Installer les dépendances :

```bash
pip install -r requirements.txt
```

---

## Lancement de l'application

```bash
streamlit run app.py
```

Ouvrir le navigateur à l'adresse affichée (généralement `http://localhost:8501`).

---

## Structure du projet

```
Churn_Client_Banque/
│
├─ app.py                 # Script principal Streamlit
├─ BankChurners.csv       # Jeu de données client
├─ requirements.txt       # Dépendances Python
├─ README.md              # Documentation
└─ env/                   # Environnement virtuel
```

---

## Modèle prédictif

* **Algorithme utilisé** : Random Forest Classifier
* **Caractéristiques principales** : âge du client, limite de crédit, solde récurrent, utilisation, nombre de transactions, CLV
* **Sortie** : probabilité de churn pour chaque client

---

## Notes importantes

* Assurez-vous que toutes les colonnes nécessaires sont présentes dans le fichier `BankChurners.csv`.
* Le dégradé de couleurs dans les tableaux nécessite **matplotlib** (`pip install matplotlib`).

---

## Auteurs

**NIAKO Analytics**

* Développé par Niako Kebe
* Contact : [drivenindata@gmail.com](mailto:drivenindata@gmail.com)

---

## Licence

Ce projet est open source.

```


