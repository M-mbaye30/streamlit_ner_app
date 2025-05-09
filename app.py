# Importer les bibliothèques nécessaires
import streamlit as st
import tensorflow as tf
import numpy as np

# Importer les fonctions des autres modules
from model_loader import load_mappings, load_ner_model, get_available_models
from ner_processor import get_processor, GLiNER_Processor  # Assurez-vous que GLiNER_Processor est correctement importé
from html_generator import generate_html

st.set_page_config(
    page_title="NER Models Inference",
    page_icon="👀",
    layout="wide"
)
st.title("👀 Named Entity Recognition App")

# Récupérer la liste des modèles disponibles
available_models = get_available_models()

# Sélection du modèle
selection_model = st.selectbox("Sélectionnez un modèle", options=available_models)

# Afficher des informations sur le modèle sélectionné
if selection_model == "BiLSTM-CRF":
    st.write("Modèle BiLSTM-CRF spécialisé dans la reconnaissance d'entités spécifiques aux anticorps monoclonaux.")
elif selection_model == "GLiNER":
    st.write("GLiNER est un modèle transformer basé sur BERT optimisé pour la reconnaissance d'entités nommées.")
elif selection_model == "SciSpaCy":
    st.write("SciSpaCy est un modèle spécialisé pour l'analyse de textes biomédicaux et scientifiques.")

st.write("Entrez une description de mAb pour identifier les entités pertinentes.")

@st.cache_resource
def load_resources(model_name):
    """Charge le modèle et les mappings."""
    try:
        mappings = load_mappings(model_name)
        model = load_ner_model(model_name)
        if mappings is None or model is None:
            error_message = f"Impossible de charger le modèle {model_name} ou ses mappings. Vérifiez les logs."
            st.error(error_message)
            return None, None
        return model, mappings
    except Exception as e:
        st.error(f"Erreur lors du chargement des ressources: {e}")
        return None, None

# Charger le modèle et les mappings en fonction de la sélection
model, mappings = load_resources(selection_model)
if model is None or mappings is None:
    st.stop()

# Initialiser le processeur spécifique pour GLiNER si le modèle sélectionné est GLiNER
if selection_model == "GLiNER":
    gliner_processor = GLiNER_Processor(tokenizer=model.data_processor.transformer_tokenizer)

# Interface utilisateur pour l'entrée de texte
input_text = st.text_area("Description à analyser:", height=150, placeholder="Collez ou écrivez votre description ici...")

if st.button("Identifier les Entités"):
    if not input_text.strip():
        st.warning("Veuillez entrer du texte à analyser.")
    else:
        st.info("Analyse en cours...")

        try:
            # Obtenir le processeur approprié
            if selection_model == "GLiNER":
                processor = gliner_processor
            else:
                processor = get_processor(selection_model, mappings)

            # Prétraitement
            preprocessed_input, words = processor.preprocess_text(input_text)

            predictions = None

            # Prédiction
            if selection_model == "BiLSTM-CRF":
                if hasattr(model, 'predict'):
                    predictions = model.predict(preprocessed_input)
                elif callable(model):
                    predictions_output = model(preprocessed_input)
                    if isinstance(predictions_output, dict):
                        possible_keys = list(predictions_output.keys())
                        if len(possible_keys) == 1:
                            predictions = predictions_output[possible_keys[0]]
                        elif 'output_1' in predictions_output:
                            predictions = predictions_output['output_1']
                        elif 'tags' in predictions_output:
                            predictions = predictions_output['tags']
                        else:
                            predictions = predictions_output[possible_keys[0]]
                    else:
                        predictions = predictions_output
                    if tf.is_tensor(predictions):
                        predictions = predictions.numpy()
                else:
                    raise TypeError("Type de modèle non reconnu pour BiLSTM-CRF.")
            
            elif selection_model == "GLiNER":

                if mappings and "entity_types" in mappings and mappings["entity_types"]:
                    target_entity_types = mappings["entity_types"]
                    
                    try:
                        predictions = model.predict_entities(text=input_text, labels=target_entity_types)
                    except TypeError as e_predict:
                        # Si predict_entities a aussi un problème d'argument, on essaie predict positionnellement
                        st.warning(f"model.predict_entities a échoué ({e_predict}), tentative avec model.predict positionnel.")
                        try:
                            predictions = model.predict([input_text], target_entity_types) # arguments positionnels
                        except Exception as e_pos:
                            st.error(f"Les deux tentatives de prédiction GLiNER ont échoué: {e_pos}")
                            predictions = []
                    except Exception as e_other:
                        st.error(f"Erreur lors de model.predict_entities: {e_other}")
                        predictions = []

                else:
                    st.error("Les types d'entités cibles pour GLiNER n'ont pas été trouvés dans les mappings ou sont vides. Impossible de prédire.")
                    predictions = [] # Ou st.stop() pour arrêter l'exécution, ou gérer autrement
            
            elif selection_model == "en_core_sci_lg":

                try:
                    # `preprocessed_input` est le texte brut pour SciSpaCy
                    # `model` est l'objet nlp de spaCy
                    predictions = model(preprocessed_input) 
                    if predictions is None: # Au cas où le modèle retournerait None
                        st.error("La prédiction SciSpaCy a retourné None.")
                        predictions = [] # Assigner une liste vide par défaut
                except Exception as scispacy_err:
                    st.error(f"Erreur lors de la prédiction avec SciSpaCy : {scispacy_err}")
                    predictions = [] # Assigner une liste vide en cas d'erreur

            # Vérifier si `predictions` a été défini; sinon, assigner une liste vide
            if predictions is None:
                st.error(f"La prédiction pour le modèle {selection_model} n'a pas abouti. Vérifiez les étapes précédentes.")
                predictions = [] # Assurer que `predictions` est une liste vide pour le post-traitement

            # Post-traitement
            entities, word_label_pairs = processor.postprocess_predictions(predictions, words)

            # Génération HTML
            html_output = generate_html(word_label_pairs)

            # Affichage des résultats
            st.subheader("Entités identifiées 👾")
            st.markdown(html_output, unsafe_allow_html=True)

            st.subheader("Liste des entités extraites :")
            if entities:
                st.table(entities)
            else:
                st.write("Aucune entité trouvée.")

        except Exception as e:
            st.error(f"Une erreur est survenue : {e}")
            st.exception(e)

# Sidebar
st.sidebar.info("Application d'inférence pour modèles NER")

if selection_model == "BiLSTM-CRF" and mappings:
    st.sidebar.write("**Mappings chargés :**")
    st.sidebar.write(f"- Vocabulaire : {len(mappings.get('word2index', {}))} mots")
    st.sidebar.write(f"- Étiquettes : {len(mappings.get('label2index', {}))}")
elif selection_model == "GLiNER":
    st.sidebar.write("**Types d'entités supportés :**")
    if mappings.get("entity_types"):
        for entity in mappings["entity_types"]:
            st.sidebar.write(f"- {entity}")
    else:
        st.sidebar.write("- Non spécifiés")
elif selection_model == "SciSpaCy" and model:
    st.sidebar.write("**Informations SciSpaCy :**")
    st.sidebar.write(f"- Version : {model.meta['version']}")
    st.sidebar.write(f"- Pipeline : {', '.join(model.pipe_names)}")

if model:
    st.sidebar.success(f"Modèle {selection_model} chargé.")
else:
    st.sidebar.error(f"Modèle {selection_model} non chargé.")
