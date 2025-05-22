# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import json
import spacy
from spacy.tokens import Doc
import re 

# Constantes globales
MAX_SEQ_LEN = 512
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

class NERProcessor:
    """Classe de base pour tous les processeurs NER."""
    
    def __init__(self, mappings):
        self.mappings = mappings
    
    def preprocess_text(self, text):
        """Méthode à implémenter dans les sous-classes."""
        raise NotImplementedError
    
    def postprocess_predictions(self, predictions, processed_input_data, original_text):
        """Méthode à implémenter dans les sous-classes.
        Doit retourner (entities, word_label_pairs)
        où entities est une liste de dicts: {"text": str, "type": str, "start": int, "end": int}
        """
        raise NotImplementedError

class BiLSTM_CRF_Processor(NERProcessor):
    """Processeur pour le modèle BiLSTM-CRF."""
    
    def preprocess_text(self, text):
        """Prétraite le texte brut pour l'entrée du modèle BiLSTM-CRF."""
        word2index = self.mappings.get("word2index")
        if not word2index:
            raise ValueError("Mapping word2index manquant.")

        pad_index = word2index.get("<PAD>", 0)
        unk_index = word2index.get("<UNK>", 1)

        # Tokenisation avec conservation des spans de caractères
        word_spans = []
        for match in re.finditer(r'\S+', text):
            word_spans.append({'text': match.group(0), 'start': match.start(), 'end': match.end()})
        
        if not word_spans:
            # Gérer le cas du texte vide ou ne contenant que des espaces
            return np.array([]).reshape(1,0), []

        words_for_model = [ws['text'] for ws in word_spans]
        word_indices = [word2index.get(w, unk_index) for w in words_for_model]

        word_indices_padded = tf.keras.preprocessing.sequence.pad_sequences(
            [word_indices], maxlen=MAX_SEQ_LEN, padding='post', 
            truncating='post', value=pad_index
        )

        return word_indices_padded, word_spans # Retourne les spans des mots
    
    def postprocess_predictions(self, predictions, word_spans, original_text):
        """Post-traite les indices prédits pour extraire les entités avec leurs offsets."""
        index2label = self.mappings.get("index2label")
        if not index2label:
            raise ValueError("Mapping index2label manquant.")

        if not word_spans: # Si word_spans est vide (texte d'entrée vide/espaces uniquement)
            return [], []

        if len(predictions.shape) == 2:
            pred_indices = predictions[0]  # Premier élément du batch
        else:
            raise ValueError(f"Format de prédiction inattendu: {predictions.shape}")

        pred_indices = pred_indices.astype(int)
        pred_indices = pred_indices[:len(word_spans)] # Tronquer à la longueur des mots réels

        entities = []
        word_label_pairs = [] # Pour la compatibilité, si toujours utilisé ailleurs
        current_entity_text_parts = []
        current_entity_type = None
        current_entity_start_char = -1
        current_entity_end_char = -1

        for i, word_span_info in enumerate(word_spans):
            word = word_span_info['text']
            label_index = pred_indices[i]
            label = index2label.get(label_index, 'O')
            word_label_pairs.append((word, label)) # Construction de word_label_pairs

            if label.startswith('B-'):
                if current_entity_text_parts: # Sauvegarder l'entité précédente
                    entities.append({
                        "text": original_text[current_entity_start_char : current_entity_end_char],
                        "type": current_entity_type,
                        "start": current_entity_start_char,
                        "end": current_entity_end_char
                    })
                current_entity_text_parts = [word]
                current_entity_type = label[2:]
                current_entity_start_char = word_span_info['start']
                current_entity_end_char = word_span_info['end']
            elif label.startswith('I-'):
                if current_entity_text_parts and label[2:] == current_entity_type:
                    current_entity_text_parts.append(word)
                    current_entity_end_char = word_span_info['end'] # Mettre à jour la fin de l'entité
                else: # Erreur de séquençage ou I- sans B-
                    if current_entity_text_parts: # Sauvegarder l'entité précédente si elle existe
                        entities.append({
                            "text": original_text[current_entity_start_char : current_entity_end_char],
                            "type": current_entity_type,
                            "start": current_entity_start_char,
                            "end": current_entity_end_char
                        })
                    # Commencer une nouvelle entité (même si c'est un I- isolé, traité comme B-)
                    current_entity_text_parts = [word]
                    current_entity_type = label[2:] if len(label) > 2 else 'UNKNOWN'
                    current_entity_start_char = word_span_info['start']
                    current_entity_end_char = word_span_info['end']
            else:  # label == 'O'
                if current_entity_text_parts: # Sauvegarder l'entité précédente
                    entities.append({
                        "text": original_text[current_entity_start_char : current_entity_end_char],
                        "type": current_entity_type,
                        "start": current_entity_start_char,
                        "end": current_entity_end_char
                    })
                    current_entity_text_parts = []
                    current_entity_type = None
                    current_entity_start_char = -1

        if current_entity_text_parts: # Gérer la dernière entité 
            entities.append({
                "text": original_text[current_entity_start_char : current_entity_end_char],
                "type": current_entity_type,
                "start": current_entity_start_char,
                "end": current_entity_end_char
            })

        return entities, word_label_pairs

class GLiNER_Processor(NERProcessor):
    """Processeur pour le modèle GLiNER."""
    def __init__(self, tokenizer, mappings=None):
        if mappings:
            super().__init__(mappings)
        self.tokenizer = tokenizer # Le tokenizer de GLiNER 

    def preprocess_text(self, text):
       # GLiNER prend le texte brut pour sa propre tokenisation interne lors de la prédiction.
       # La tokenisation ici est pour `words` qui est utilisé dans `postprocess_predictions` pour `word_label_pairs`.
       # Si `word_label_pairs` n'est plus nécessaire, `words` peut être simplifié ou supprimé.
       words = text.split() # Conserver pour l'instant pour word_label_pairs
       return text, words # GLiNER model.predict_entities prend le texte brut
    
    def postprocess_predictions(self, predictions, words_for_pairing, original_text):
        """Post-traite les prédictions de GLiNER. `predictions` est la sortie directe du modèle GLiNER."""
        entities = []
        for entity_pred in predictions:
            # Chaque `entity_pred` est un dict avec 'text', 'label', 'start', 'end'
            entities.append({
                "text": entity_pred['text'], # Utiliser le texte fourni par GLiNER
                "type": entity_pred['label'],
                "start": entity_pred['start'],
                "end": entity_pred['end']
            })
        
        word_label_pairs = [(word, 'O') for word in words_for_pairing]
        
        
        return entities, word_label_pairs

class SciSpaCy_Processor(NERProcessor):
    """Processeur pour le modèle SciSpaCy."""
    
    def preprocess_text(self, text):
        """Prétraite le texte pour SciSpaCy."""
        # SciSpaCy traite directement le texte brut, qui sera passé au modèle spaCy.
        return text, text.split() 
    
    def postprocess_predictions(self, doc, words_for_pairing, original_text):
        """Post-traite les prédictions de SciSpaCy. `doc` est l'objet Doc de spaCy."""
        
        if not isinstance(doc, Doc):
             # Gérer le cas où `doc` n'est pas un objet Doc (par exemple, une liste vide de prédictions)
            if isinstance(doc, list) and not doc: # Si c'est une liste vide
                return [], [(word, 'O') for word in words_for_pairing]
            # Si `doc` est None ou autre chose inattendue
            print(f"Avertissement SciSpaCy: 'predictions' n'est pas un objet Doc spaCy. Reçu: {type(doc)}")
            return [], [(word, 'O') for word in words_for_pairing]

        all_entities = []
        for ent in doc.ents:
            all_entities.append({
                "text": ent.text,
                "type": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })

        # Construction de word_label_pairs (basé sur la tokenisation de spaCy)
        spacy_tokens_texts = [token.text for token in doc]
        word_label_pairs = [(token_text, 'O') for token_text in spacy_tokens_texts]
        for ent in doc.ents:
            # Les indices ent.start et ent.end sont des indices de tokens spaCy
            for i in range(ent.start, ent.end):
                if i < len(word_label_pairs):
                    token_text = doc[i].text # ou spacy_tokens_texts[i]
                    if i == ent.start:
                        word_label_pairs[i] = (token_text, f'B-{ent.label_}')
                    else:
                        word_label_pairs[i] = (token_text, f'I-{ent.label_}')

        return all_entities, word_label_pairs

def get_processor(model_name, mappings):
    """Renvoie le processeur approprié pour le modèle spécifié."""
    from model_loader import MODELS_CONFIG # Importation locale pour éviter les dépendances circulaires
    
    if model_name not in MODELS_CONFIG:
        raise ValueError(f"Modèle non reconnu: {model_name}")
    
    processor_config = MODELS_CONFIG[model_name]
    processor_type = processor_config["processor_type"]
    
    if processor_type == "bilstm_crf":
        return BiLSTM_CRF_Processor(mappings)
    elif processor_type == "gliner":
        
        return GLiNER_Processor(tokenizer=mappings.get('tokenizer'), mappings=mappings) 
    elif processor_type == "scispacy":
        return SciSpaCy_Processor(mappings) # Mappings peut être utilisé pour des configurations spécifiques si nécessaire
    else:
        raise ValueError(f"Type de processeur non pris en charge: {processor_type}")