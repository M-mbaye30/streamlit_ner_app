import os
import pickle
import tensorflow as tf
from tf2crf import CRF, ModelWithCRFLoss
import json
import sys

# Structure de configuration pour les modèles disponibles
MODELS_CONFIG = {
    "BiLSTM-CRF": {
        "model_dir": r"./BiLSTM-CRF_Model",
        "mappings_file": "bilstm_crf_mappings.pkl",
        "custom_objects": {"CRF": CRF, "ModelWithCRFLoss": ModelWithCRFLoss},
        "processor_type": "bilstm_crf"
    },
    "GLiNER": {
        "model_dir": r"./Gliner_Small_Model",
        "mappings_file": None,  # GLiNER n'utilise pas de mappings classiques
        "custom_objects": {},  # Pas d'objets personnalisés TensorFlow
        "processor_type": "gliner"
    },
    "en_core_sci_lg": {
        "model_dir": r"./sci_lg_ner_model",  # Nom du modèle SciSpaCy 
        "mappings_file": None,  # SciSpaCy n'utilise pas de mappings personnalisés
        "custom_objects": {},
        "processor_type": "scispacy"
    }
}

def get_available_models():
    """Renvoie la liste des modèles disponibles."""
    return list(MODELS_CONFIG.keys())

def load_mappings(model_name):
    """Charge les mappings pour un modèle spécifique si nécessaire."""
    if model_name not in MODELS_CONFIG:
        raise ValueError(f"Modèle non reconnu: {model_name}")
    
    config = MODELS_CONFIG[model_name]
    
    # Si le modèle n'utilise pas de mappings
    if config["mappings_file"] is None:
        if config["processor_type"] == "gliner":
            # Pour GLiNER, on retourne les types d'entités supportés
            return {
                "entity_types": ["RECEPT_ID", "SPECIFITY", "MAB_SPECIES", "CHAIN_TYPE", "CHAIN_SPECIES",
                                "DOMAIN_TYPE", "GENE", "ALLELE", "MUTATION", "ALLOTYPE", "CDR_IMGT", 
                                "HINGE_REGION", "SPECIFITY_CLASS", "BRIDGE", "CONJUGATE", 
                                "FUSED", "PRODUCTION_SYSTEME", "MOA"]
            }
        elif config["processor_type"] == "scispacy":
            # Pour SciSpaCy, retourner un mappage vide
            return {}
        else:
            return {}
    
    # Pour les modèles avec mappings classiques (comme BiLSTM-CRF)
    mappings_path = os.path.join(config["model_dir"], config["mappings_file"])
    
    try:
        with open(mappings_path, "rb") as f:
            mappings = pickle.load(f)
        print(f"Mappings pour {model_name} chargés avec succès.")
        
        # Vérifier la présence des clés essentielles pour les modèles qui en ont besoin
        if config["processor_type"] == "bilstm_crf":
            required_keys = ["word2index", "index2word", "label2index", "index2label"]
            for key in required_keys:
                if key not in mappings:
                    raise KeyError(f"Clé manquante dans les mappings: {key}")
        
        return mappings
    except FileNotFoundError:
        print(f"Erreur: Fichier de mappings non trouvé à {mappings_path}")
        return None
    except Exception as e:
        print(f"Erreur lors du chargement des mappings: {e}")
        return None

def load_ner_model(model_name):
    """Charge un modèle NER spécifique."""
    if model_name not in MODELS_CONFIG:
        raise ValueError(f"Modèle non reconnu: {model_name}")
    
    config = MODELS_CONFIG[model_name]
    model_path = config["model_dir"]
    
    try:
        # Cas spécial pour GLiNER
        if config["processor_type"] == "gliner":
            from gliner import GLiNER
            # Modification pour éviter l'erreur torch::class_
            os.environ['TORCH_CLASSES_PATH'] = '.'  # Définir un chemin explicite
            try:
                # Essayer d'abord avec l'option use_torchscript=False
                model = GLiNER.from_pretrained(model_path, use_torchscript=False)
            except TypeError:
                # Si l'option n'est pas supportée, utiliser l'appel standard
                model = GLiNER.from_pretrained(model_path)
            print(f"Modèle GLiNER chargé avec succès depuis {model_path}")
            return model
            
        # Cas spécial pour SciSpaCy
        elif config["processor_type"] == "scispacy":
            import spacy
            try:
                model = spacy.load(model_path)
                print(f"Modèle SciSpaCy chargé avec succès: {model_path}")
            except OSError:
                print(f"Le modèle SciSpaCy {model_path} n'est pas installé. Installation en cours...")
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "scispacy"])
                subprocess.check_call([sys.executable, "-m", "pip", "install", f"https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/{model_path}-0.5.1.tar.gz"])
                model = spacy.load(model_path)
                print(f"Modèle SciSpaCy {model_path} installé et chargé avec succès.")
            return model
            
        # Pour les modèles Keras/TensorFlow (comme BiLSTM-CRF)
        custom_objects = config["custom_objects"]
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        print(f"Modèle {model_name} chargé avec succès.")
        return model
    except OSError as e:
        print(f"Erreur lors du chargement du modèle (vérifiez le chemin et le format): {e}")
        return None
    except Exception as e:
        print(f"Erreur inattendue lors du chargement du modèle: {e}")
        return None

# Si exécuté directement, tester le chargement
if __name__ == "__main__":
    import sys
    print("Modèles disponibles:")
    for model in get_available_models():
        print(f" - {model}")
        
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        print(f"\nTest de chargement du modèle {model_name}...")
        mappings = load_mappings(model_name)
        model = load_ner_model(model_name)
        
        if model:
            print(f"Chargement réussi pour {model_name}")
        else:
            print(f"Échec du chargement pour {model_name}")
