import numpy as np
import tensorflow as tf
import json
import spacy
from spacy.tokens import Doc
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
    
    def postprocess_predictions(self, predictions, words):
        """Méthode à implémenter dans les sous-classes."""
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

        words = text.split()

        # Conversion en indices 
        word_indices = [word2index.get(w, unk_index) for w in words]

        # Padding avec les mêmes paramètres que dans l'entraînement
        word_indices_padded = tf.keras.preprocessing.sequence.pad_sequences(
            [word_indices], maxlen=MAX_SEQ_LEN, padding='post', 
            truncating='post', value=pad_index
        )

        return word_indices_padded, words
    
    def postprocess_predictions(self, predictions, words):
        """Post-traite les indices prédits pour extraire les entités et les paires mot/label."""
        index2label = self.mappings.get("index2label")
        if not index2label:
            raise ValueError("Mapping index2label manquant.")

        if len(predictions.shape) == 2:
            pred_indices = predictions[0]  # Premier élément du batch
        else:
            raise ValueError(f"Format de prédiction inattendu: {predictions.shape}")

        # Assurer que les indices sont des entiers
        pred_indices = pred_indices.astype(int)

        # Tronquer les prédictions à la longueur des mots réels
        pred_indices = pred_indices[:len(words)]

        entities = []
        word_label_pairs = []
        current_entity_text = []
        current_entity_type = None

        for i, word in enumerate(words):
            label_index = pred_indices[i]
            label = index2label.get(label_index, 'O')
            word_label_pairs.append((word, label))

            if label.startswith('B-'):
                if current_entity_text:
                    entities.append({
                        "text": " ".join(current_entity_text),
                        "type": current_entity_type
                    })
                current_entity_text = [word]
                current_entity_type = label[2:]
            elif label.startswith('I-'):
                if current_entity_text and label[2:] == current_entity_type:
                    current_entity_text.append(word)
                else:
                    if current_entity_text:
                        entities.append({
                            "text": " ".join(current_entity_text),
                            "type": current_entity_type
                        })
                    current_entity_text = [word]
                    current_entity_type = label[2:]
            else:  # label == 'O' ou autre
                if current_entity_text:
                    entities.append({
                        "text": " ".join(current_entity_text),
                        "type": current_entity_type
                    })
                    current_entity_text = []
                    current_entity_type = None

        # Gérer la dernière entité potentielle
        if current_entity_text:
            entities.append({
                "text": " ".join(current_entity_text),
                "type": current_entity_type
            })

        return entities, word_label_pairs

class GLiNER_Processor(NERProcessor):
    """Processeur pour le modèle GLiNER."""
    def __init__(self, tokenizer, mappings=None):
        if mappings:
            super().__init__(mappings)
        self.tokenizer = tokenizer

    def preprocess_text(self, text):
       fixed_max_length = 128
       encoding = self.tokenizer(text, 
                                 return_tensors="pt", 
                                 add_special_tokens=True, 
                                 truncation=True, 
                                 padding="max_length", 
                                 max_length=fixed_max_length)
       words = text.split() # À ajuster si besoin pour le post-processing
       return encoding, words # 'encoding' est le dictionnaire avec input_ids, attention_mask
    
    def postprocess_predictions(self, predictions, words):
        """Post-traite les prédictions de GLiNER."""
        # GLiNER renvoie directement un dict d'entités
        entities = []
        word_label_pairs = [(word, 'O') for word in words]  # Par défaut, tout est 'O'
        
        # Convertir le format des entités et mettre à jour les paires (mot, label)
        for entity in predictions:
            start_idx = entity['start']
            end_idx = entity['end']
            entity_type = entity['label']
            entity_text = entity['text']
            
            # Ajouter l'entité formatée
            entities.append({
                "text": entity_text,
                "type": entity_type
            })
            
            # Mettre à jour les word_label_pairs pour l'affichage
            if start_idx < len(words) and end_idx < len(words):
                for i in range(start_idx, end_idx + 1):
                    if i == start_idx:
                        word_label_pairs[i] = (words[i], f'B-{entity_type}')
                    else:
                        word_label_pairs[i] = (words[i], f'I-{entity_type}')
        
        return entities, word_label_pairs

class SciSpaCy_Processor(NERProcessor):
    """Processeur pour le modèle SciSpaCy."""
    
    def preprocess_text(self, text):
        """Prétraite le texte pour SciSpaCy."""
        # SciSpaCy traite directement le texte brut
        return text, text.split()
    
    def postprocess_predictions(self, predictions, words):
        """Post-traite les prédictions de SciSpaCy."""
        """Assume predictions is a list of Doc objects"""
        
        # Gérer le cas où predictions est un seul Doc (et pas une liste)
        if isinstance(predictions, Doc):
            doc = predictions
        elif isinstance(predictions, list) and predictions:
            doc = predictions[0]  # liste de Doc(s)
        else:
        # predictions est vide ou None
            return [], [(word, 'O') for word in words]

        all_entities = []
        spacy_tokens_texts = [token.text for token in doc]
        word_label_pairs = [(token_text, 'O') for token_text in spacy_tokens_texts]

        for ent in doc.ents:
            all_entities.append({
            "text": ent.text,
            "type": ent.label_
        })

            for i in range(ent.start, ent.end):
                if i < len(word_label_pairs):
                    token_text = doc[i].text
                    if i == ent.start:
                        word_label_pairs[i] = (token_text, f'B-{ent.label_}')
                    else:
                        word_label_pairs[i] = (token_text, f'I-{ent.label_}')

        return all_entities, word_label_pairs

def get_processor(model_name, mappings):
    """Renvoie le processeur approprié pour le modèle spécifié."""
    from model_loader import MODELS_CONFIG
    
    if model_name not in MODELS_CONFIG:
        raise ValueError(f"Modèle non reconnu: {model_name}")
    
    processor_type = MODELS_CONFIG[model_name]["processor_type"]
    
    if processor_type == "bilstm_crf":
        return BiLSTM_CRF_Processor(mappings)
    elif processor_type == "gliner":
        return GLiNER_Processor(mappings)
    elif processor_type == "scispacy":
        return SciSpaCy_Processor(mappings)
    else:
        raise ValueError(f"Type de processeur non pris en charge: {processor_type}")