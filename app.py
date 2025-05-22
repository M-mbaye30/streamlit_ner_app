# Importer les bibliothèques nécessaires
import streamlit as st
import tensorflow as tf
import numpy as np
import time 
import pandas as pd 
import io 
from collections import Counter 
import streamlit as st
st.set_option('server.runOnSave', False)


# Importer les fonctions des autres modules
from model_loader import load_mappings, load_ner_model, get_available_models
from ner_processor import get_processor, GLiNER_Processor
# Mise à jour de l'import pour la nouvelle fonction de génération HTML et ENTITY_COLORS
from html_generator import generate_html_for_entities, ENTITY_COLORS, DEFAULT_COLOR

st.set_page_config(
    page_title="NER Models Inference",
    page_icon="👀",
    layout="wide"
)
#st.title("👀 Named Entity Recognition App")
st.markdown("<h1 style='text-align: center; color: #4A90E2;'>👀 Named Entity Recognition App</h1>", unsafe_allow_html=True)

# --- Exemples de Textes ---
EXAMPLE_TEXTS = {
    "Example : foravirumab Description": "immunoglobulin G1-kappa, anti-[rabies virus glycoprotein], Homo sapiens monoclonal antibody; gamma1 heavy chain (1-448) [Homo sapiens VH (IGHV3-33*03 (95.90%) -(IGHD)-IGHJ4*01) [8.8.12] (1-119) -IGHG1*03, CH3 K130>del (120-448)], (222-214')-disulfide with kappa light chain (1'-214') [Homo sapiens V-KAPPA (IGKV1-17*01 (95.80%) -IGKJ4*01) [6.3.9] (1'-107') -IGKC*01 (108'-214')]; (228-228'':231-231'')-bisdisulfide dimer"
    
}

# --- Fonctions Utilitaires ---
@st.cache_resource
def load_resources(model_name):
    """Charge le modèle et les mappings."""
    try:
        mappings = load_mappings(model_name)
        model = load_ner_model(model_name)
        if model_name != "GLiNER" and (mappings is None or model is None):
            error_message = f"Impossible de charger le modèle {model_name} ou ses mappings. Vérifiez les logs."
            st.error(error_message)
            return None, None
        elif model_name == "GLiNER" and model is None:
            error_message = f"Impossible de charger le modèle {model_name}. Vérifiez les logs."
            st.error(error_message)
            return None, None
        return model, mappings
    except Exception as e:
        st.error(f"Erreur lors du chargement des ressources pour {model_name}: {e}")
        return None, None

# --- Initialisation de Session State --- 
if 'current_text_input' not in st.session_state:
    st.session_state.current_text_input = ""
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'last_entities' not in st.session_state:
    st.session_state.last_entities = []
if 'last_input_text' not in st.session_state:
    st.session_state.last_input_text = ""
if 'processing_time' not in st.session_state:
    st.session_state.processing_time = 0.0

# --- Configuration de la Sidebar ---
st.sidebar.info("Inference application for NER models")

available_models = get_available_models()
selection_model = st.sidebar.selectbox("Select a model", options=available_models)

model, mappings = load_resources(selection_model)

if model is None:
    st.error(f"Le chargement du modèle {selection_model} a échoué. L'application ne peut pas continuer.")
    st.stop()
if selection_model != "GLiNER" and mappings is None:
    st.error(f"Le chargement des mappings pour {selection_model} a échoué. L'application ne peut pas continuer.")
    st.stop()
if selection_model == "GLiNER" and mappings is None:
    st.warning("Les mappings pour GLiNER n'ont pas pu être chargés. Les types d'entités cibles pourraient ne pas être disponibles.")
    mappings = {} 

if model:
    st.sidebar.success(f"Model {selection_model} loaded.")
else:
    st.sidebar.error(f"Model {selection_model} not loaded.")

st.sidebar.subheader("Model Information")
if selection_model == "BiLSTM-CRF":
    st.sidebar.write("BiLSTM-CRF is a popular model used in NLP tasks such asNER. The model combines Bidirectional Long Short-Term Memory (BiLSTM) networks with Conditional Random Fields (CRF) to leverage the strengths of both architectures.")
    if mappings:
        st.sidebar.write("**Mappings loaded:**")
        st.sidebar.write(f"- Vocabulary: {len(mappings.get('word2index', {}))} words")
        st.sidebar.write(f"- Labels: {len(mappings.get('label2index', {}))}")
elif selection_model == "GLiNER":
    st.sidebar.write("GLiNER is a BERT-based transform model tuned for named entity recognition.")
    # if mappings:
    #     st.sidebar.write("**Target entity types for GLiNER:**")
    #     entity_types_gliner = mappings.get("entity_types")
    #     if entity_types_gliner:
    #         for entity_type_name in entity_types_gliner:
    #             st.sidebar.write(f"- {entity_type_name}")
    #     else:
    #         st.sidebar.write("- Not specified (model might use defaults or find none)")
elif selection_model == "SciSpaCy" or selection_model == "en_core_sci_lg":
    st.sidebar.write("is a specialized NER model for biomedical and scientific texts.")
    if model and hasattr(model, 'meta'):
        st.sidebar.write(f"- Model: {model.meta.get('name', 'N/A')}")
        st.sidebar.write(f"- Version: {model.meta.get('version', 'N/A')}")
        st.sidebar.write(f"- Pipeline: {', '.join(model.pipe_names)}")

st.sidebar.subheader("Color Legend")
legend_html = "<ul style='padding-left: 20px; margin-bottom: 10px;'>"
defined_entity_types = sorted([k for k in ENTITY_COLORS.keys() if k != "DEFAULT_ENTITY_TYPE"])

for entity_type in defined_entity_types:
    color = ENTITY_COLORS[entity_type]
    legend_html += f"<li><span style='background-color:{color}; padding: 0.1em 0.4em; margin-right: 5px; border-radius: 0.3em;'>&nbsp;&nbsp;&nbsp;</span> {entity_type}</li>"
legend_html += f"<li><span style='background-color:{DEFAULT_COLOR}; padding: 0.1em 0.4em; margin-right: 5px; border-radius: 0.3em;'>&nbsp;&nbsp;&nbsp;</span> Other / Default</li>"
legend_html += "</ul>"
st.sidebar.markdown(legend_html, unsafe_allow_html=True)

st.sidebar.subheader("Filter Entity Types")
selected_types_for_display = st.sidebar.multiselect(
    label="Select entity types to display:",
    options=defined_entity_types,
    default=defined_entity_types
)

if selection_model == "GLiNER":
    try:
        if hasattr(model, 'tokenizer'):
            gliner_tokenizer = model.tokenizer
        elif hasattr(model, 'data_processor') and hasattr(model.data_processor, 'transformer_tokenizer'):
            gliner_tokenizer = model.data_processor.transformer_tokenizer
        else:
            st.error("Impossible de trouver le tokenizer pour le modèle GLiNER.")
            gliner_tokenizer = None
        gliner_processor = GLiNER_Processor(tokenizer=gliner_tokenizer, mappings=mappings)
    except Exception as e:
        st.error(f"Erreur lors de l'initialisation du processeur GLiNER: {e}")
        st.stop()

# --- Zone Principale de l'Application ---
st.write("Enter a mAb description to identify the relevant entities.")

example_options = ["Enter your own text"] + list(EXAMPLE_TEXTS.keys())
chosen_example_key = st.selectbox("Or choose an example:", options=example_options, key="example_selector")

if chosen_example_key != "Enter your own text":
    st.session_state.current_text_input = EXAMPLE_TEXTS[chosen_example_key]

input_text = st.text_area("Text to analyze:", value=st.session_state.current_text_input, height=150, placeholder="Paste or write your description here...", key="main_text_area")
st.session_state.current_text_input = input_text 

if st.button("Identify Entities"):
    if not input_text.strip():
        st.warning("Please enter text to analyze.")
        st.session_state.analysis_done = False
    else:
        st.session_state.analysis_done = False 
        start_time = time.time() 
        st.info("Analysis in progress...")
        try:
            if selection_model == "GLiNER":
                processor = gliner_processor
            else:
                processor = get_processor(selection_model, mappings)
            
            preprocessed_input_for_model, processed_input_data_for_postproc = processor.preprocess_text(input_text)
            predictions = None

            if selection_model == "BiLSTM-CRF":
                if hasattr(model, 'predict'):
                    predictions = model.predict(preprocessed_input_for_model)
                elif callable(model):
                    predictions_output = model(preprocessed_input_for_model)
                    if isinstance(predictions_output, dict):
                        possible_keys = list(predictions_output.keys())
                        if 'tags' in predictions_output: predictions = predictions_output['tags']
                        elif 'output_1' in predictions_output: predictions = predictions_output['output_1']
                        elif len(possible_keys) == 1: predictions = predictions_output[possible_keys[0]]
                        else: predictions = predictions_output[possible_keys[0]]
                    else:
                        predictions = predictions_output
                    if tf.is_tensor(predictions):
                        predictions = predictions.numpy()
                else:
                    raise TypeError("Unrecognized model type for BiLSTM-CRF.")
            
            elif selection_model == "GLiNER":
                target_entity_types_from_mappings = mappings.get("entity_types", [])
                if not target_entity_types_from_mappings:
                    target_entity_types = defined_entity_types
                    st.info(f"No specific entity types in mappings for GLiNER. Using general types: {', '.join(target_entity_types)}")
                else:
                    target_entity_types = target_entity_types_from_mappings
                
                try:
                    predictions = model.predict_entities(text=input_text, labels=target_entity_types)
                except TypeError:
                    try:
                        predictions = model.predict([input_text], target_entity_types)
                    except Exception as e_pos:
                        st.error(f"GLiNER prediction attempts failed: {e_pos}")
                        predictions = []
                except Exception as e_other:
                    st.error(f"Error during model.predict_entities: {e_other}")
                    predictions = []
            
            elif selection_model == "SciSpaCy" or selection_model == "en_core_sci_lg":
                try:
                    predictions = model(preprocessed_input_for_model)
                    if predictions is None:
                        st.error("SciSpaCy prediction returned None.")
                        predictions = []
                except Exception as scispacy_err:
                    st.error(f"Error during prediction with SciSpaCy: {scispacy_err}")
                    predictions = []

            if predictions is None:
                st.error(f"Prediction for model {selection_model} did not succeed.")
                predictions = []

            all_entities, _ = processor.postprocess_predictions(predictions, processed_input_data_for_postproc, input_text)
            st.session_state.last_entities = all_entities
            st.session_state.last_input_text = input_text
            st.session_state.processing_time = time.time() - start_time
            st.session_state.analysis_done = True

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
            st.exception(e)
            st.session_state.analysis_done = False

# Affichage des résultats
if st.session_state.analysis_done:
    st.subheader("Analysis Results")
    if st.session_state.last_entities:
        entities_to_display = [entity for entity in st.session_state.last_entities if entity['type'] in selected_types_for_display]
        
        st.markdown("#### Entity Statistics (based on current filters)")
        total_displayed_entities = len(entities_to_display)
        col1, col2 = st.columns(2)
        col1.metric(label="Total Displayed Entities", value=total_displayed_entities)
        if entities_to_display:
            entity_type_counts = Counter(entity['type'] for entity in entities_to_display)
            most_common_type, most_common_count = entity_type_counts.most_common(1)[0]
            col2.metric(label=f"Most Frequent: {most_common_type}", value=most_common_count)
            with st.expander("View detailed counts per type"):
                df_counts = pd.DataFrame(entity_type_counts.items(), columns=['Entity Type', 'Count']).sort_values(by='Count', ascending=False)
                st.dataframe(df_counts, use_container_width=True)
        else:
            col2.metric(label="Most Frequent", value="N/A")
        st.markdown("---")

        html_output = generate_html_for_entities(st.session_state.last_input_text, entities_to_display)
        st.subheader("Identified Entities in Text 👾")
        st.markdown(html_output, unsafe_allow_html=True)
        
        st.subheader("Extracted Entities List (filtered)")
        if entities_to_display:
            # Préparer les données pour le tableau et l'export CSV
            # Pour l'export, nous voulons toutes les infos: text, type, start, end
            # Pour le tableau, seulement text et type.
            entities_for_export_df = pd.DataFrame(entities_to_display)
            entities_for_table_display = entities_for_export_df[['text', 'type']]
            st.table(entities_for_table_display)

            # --- Export CSV --- 
            csv_buffer = io.StringIO()
            entities_for_export_df.to_csv(csv_buffer, index=False)
            csv_string = csv_buffer.getvalue()
            st.download_button(
                label="Download Filtered Entities as CSV",
                data=csv_string,
                file_name=f"filtered_entities_{selection_model}.csv",
                mime="text/csv",
            )
            # --- Fin Export CSV ---
        else:
            st.write("No entities to display based on current filters.")
    else:
        st.write("No entities found in the provided text.")
    
    st.caption(f"Processing time: {st.session_state.processing_time:.2f} seconds")

