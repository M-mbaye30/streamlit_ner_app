# Streamlit NER App

Une application web interactive de reconnaissance d'entités nommées (NER) construite avec Streamlit.

## 📋 Description

Cette application permet d'identifier et d'extraire automatiquement des entités nommées (personnes, lieux, organisations, etc.) à partir de textes en utilisant des modèles de traitement automatique du langage naturel.

## ✨ Fonctionnalités

- Interface utilisateur intuitive avec Streamlit
- Reconnaissance d'entités nommées en temps réel
- Visualisation des entités extraites
- Support de différents types d'entités (PERSON, ORG, LOC, etc.)
- Possibilité de traiter du texte personnalisé

## 🛠️ Technologies utilisées

- **Python 3.x**
- **Streamlit** - Interface web
- **spaCy** ou **Transformers** - Modèles NLP
- **Pandas** - Manipulation de données
- **Plotly/Matplotlib** - Visualisations (si applicable)

## 📦 Installation

1. Clonez le dépôt :
```bash
git clone https://github.com/M-mbaye30/streamlit_ner_app.git
cd streamlit_ner_app
```

2. Créez un environnement virtuel :
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
```

3. Installez les dépendances :
```bash
pip install -r requirements.txt
```

4. Téléchargez le modèle spaCy (si utilisé) :
```bash
python -m spacy download fr_core_news_sm  # Pour le français
python -m spacy download en_core_web_sm   # Pour l'anglais
```

## 🚀 Utilisation

1. Lancez l'application :
```bash
streamlit run app.py
```

2. Ouvrez votre navigateur à l'adresse `http://localhost:8501`

3. Saisissez votre texte dans la zone de texte

4. Visualisez les entités extraites avec leurs types et scores de confiance

## 📁 Structure du projet

```
streamlit_ner_app/
│
├── app.py                 # Application principale Streamlit
├── requirements.txt       # Dépendances Python
├── models/               # Modèles NLP (si applicable)
├── utils/                # Fonctions utilitaires
├── data/                 # Données d'exemple
└── README.md            # Ce fichier
```




