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
    
    "DEFAULT_ENTITY_TYPE": "#E0E0E0" 
}
DEFAULT_COLOR = "#E0E0E0" # Utilisé si un type n'est pas dans ENTITY_COLORS

def generate_html_for_entities(original_text, entities_with_positions):
    """
    Génère une chaîne HTML avec des spans colorés pour les entités NER complètes.
    'original_text' est le texte brut d'origine.
    'entities_with_positions' est une liste de dictionnaires, où chaque dictionnaire est:
        {'text': 'entity_text', 'type': 'ENTITY_TYPE', 'start': start_char_offset, 'end': end_char_offset}
    Les entités dans la liste DOIVENT être triées par leur 'start' offset et ne doivent pas se chevaucher de manière conflictuelle.
    """
    html_parts = []
    current_char_index = 0

    
    for entity in entities_with_positions:
        entity_text = entity['text']
        entity_type = entity['type']
        start_char = entity.get('start')
        end_char = entity.get('end')

        # Vérification de base des indices et du texte de l'entité
        if start_char is None or end_char is None or start_char < current_char_index or end_char < start_char or original_text[start_char:end_char] != entity_text:
            # Si les indices sont manquants, invalides, ou si le texte de l'entité ne correspond pas,
            # on pourrait logguer une alerte et sauter cette entité pour éviter des erreurs HTML.
            print(f"Alerte: Entité incohérente ou mal positionnée ignorée: {entity}")
            continue

        # 1. Ajouter le segment de texte AVANT l'entité actuelle
        if start_char > current_char_index:
            html_parts.append(html.escape(original_text[current_char_index:start_char]))

        # 2. Ajouter l'entité colorée
        color = ENTITY_COLORS.get(entity_type, DEFAULT_COLOR)
        
        escaped_entity_text = html.escape(entity_text)

        span_html = (
            f'<span style="background-color: {color}; '
            f'padding: 0.2em 0.4em; margin: 0.1em; '
            f'border-radius: 0.3em; line-height: 1.8; '
            f'border: 1px solid {color}; display: inline-block;">'
            f'{escaped_entity_text}'
            f'<span style="font-size: 0.8em; margin-left: 0.4em; color: #555;">'
            f'({html.escape(entity_type)})</span>'
            f'</span>'
        )
        html_parts.append(span_html)

        current_char_index = end_char

    # 3. Ajouter le segment de texte restant APRÈS la dernière entité
    if current_char_index < len(original_text):
        html_parts.append(html.escape(original_text[current_char_index:]))

    return "".join(html_parts)


