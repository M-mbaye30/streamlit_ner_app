# -*- coding: utf-8 -*-

import html

# --- Palette de couleurs pour les types d'entités ---

ENTITY_COLORS = {
    "RECEPT_ID": "#FFADAD",
    "SPECIFITY": "#FFD6A5",
    "CHAIN_TYPE": "#FDFFB6",
    "CHAIN_SPECIES": "#CAFFBF",
    "DOMAIN_TYPE": "#9BF6FF",
    "ALLELE": "#A0C4FF",
    "MUTATION": "#BDB2FF",
    "BRIDGE": "#FFC6FF",
    "MOA": "#FFFFFC",
    "MAB_SPECIES": "#E4C1F9",
    "GENE": "#EDE7B1",
    "FUSED": "#A9DEF9",
    "HINGE_REGION": "#FF99C8",
    "SPECIFITY_CLASS": "#FCF6BD",
    "CONJUGATE": "#D0F4DE",
    "ALLOTYPE": "#C1AEB3",
    "PRODUCTION_SYSTEME": "#E2E2DF",
    "CDR_IMGT": "#F5CDBE",
}
DEFAULT_COLOR = "#E0E0E0"

def generate_html(word_label_pairs):
    """Génère une chaîne HTML avec des spans colorés pour les entités NER."""
    html_parts = []
    
    for word, label in word_label_pairs:
        escaped_word = html.escape(word)
        
        if label == 'O':
            html_parts.append(escaped_word)
        else:
            if label.startswith("B-") or label.startswith("I-"):
                entity_type = label[2:]
            else:
                entity_type = label
            
            color = ENTITY_COLORS.get(entity_type, DEFAULT_COLOR)
            
            span = (
                f'<span style="background-color: {color}; ' 
                f'padding: 0.2em 0.4em; margin: 0.1em; ' 
                f'border-radius: 0.3em; line-height: 1.8; ' 
                f'border: 1px solid {color};">' 
                f'{escaped_word}' 
                f'<span style="font-size: 0.8em; margin-left: 0.4em; color: #555;">' 
                f'({html.escape(entity_type)})</span>' 
                f'</span>'
            )
            html_parts.append(span)
            
    return " ".join(html_parts)