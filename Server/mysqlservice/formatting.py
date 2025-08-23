from typing import Dict, Any

def format_results(result: Dict[str, Any]) -> str:
    """Formate les résultats de manière naturelle et conversationnelle pour les assistants vocaux français"""
    
    # Gérer les erreurs simplement
    if not result.get('success', False):
        return "Désolé, je n'ai pas pu obtenir cette information pour le moment. Pourriez-vous reformuler votre question ?"

    data = result.get('data', [])
    columns = result.get('columns', [])
    
    # Aucun résultat trouvé
    if not data:
        return "Je n'ai trouvé aucune donnée pour cette question."
    
    # Résultats à valeur unique (comme moyennes, totaux, comptages)
    if len(data) == 1 and len(data[0]) == 1:
        value = data[0][0]
        
        # Formater les nombres pour la voix
        if isinstance(value, (int, float)):
            if value == int(value):
                value = int(value)  # Supprimer .0 des nombres entiers
            else:
                value = round(float(value), 2)  # Arrondir à 2 décimales
        
        # Format français avec séparateur de milliers
        if isinstance(value, (int, float)) and value >= 1000:
            value_str = f"{value:,.2f}".replace(",", " ").replace(".", ",") if isinstance(value, float) else f"{value:,}".replace(",", " ")
            return f"La réponse est {value_str}."
        else:
            return f"La réponse est {value}."
    
    # Ligne unique avec plusieurs colonnes
    elif len(data) == 1:
        row = data[0]
        if len(columns) == 2:
            # Modèle commun : étiquette et valeur
            column_french = translate_column_name(columns[0])
            value = format_french_number(row[1])
            return f"Le {column_french} est de {value}."
        else:
            # Plusieurs valeurs
            parts = []
            for i, col in enumerate(columns):
                value = format_french_number(row[i])
                col_french = translate_column_name(col)
                parts.append(f"le {col_french} est de {value}")
            return f"Voici ce que j'ai trouvé : {', '.join(parts)}."
    
    # Plusieurs lignes - fournir un résumé
    else:
        row_count = len(data)
        
        # Si c'est une liste simple (une colonne)
        if len(columns) == 1:
            if row_count <= 5:
                values = [str(row[0]) for row in data]
                return f"J'ai trouvé {row_count} résultats : {', '.join(values)}."
            else:
                return f"J'ai trouvé {row_count} résultats. Les principaux sont {data[0][0]}, {data[1][0]}, et {data[2][0]}."
        
        # Plusieurs colonnes - décrire le modèle
        else:
            if row_count <= 3:
                # Lister quelques résultats
                descriptions = []
                for row in data:
                    if len(columns) == 2:
                        value = format_french_number(row[1])
                        descriptions.append(f"{row[0]} avec {value}")
                    else:
                        descriptions.append(f"{row[0]}")
                return f"Voici les résultats : {', '.join(descriptions)}."
            else:
                # Résumer de nombreux résultats
                first_col = columns[0]
                if 'date' in first_col.lower() or 'month' in first_col.lower():
                    recent_value = format_french_number(data[0][1])
                    earliest_value = format_french_number(data[-1][1])
                    return f"J'ai trouvé des données pour {row_count} périodes. La plus récente montre {recent_value} et la plus ancienne montre {earliest_value}."
                else:
                    top_value = format_french_number(data[0][1])
                    return f"J'ai trouvé {row_count} résultats. Le premier résultat est {data[0][0]} avec {top_value}."

def translate_column_name(column_name: str) -> str:
    """Traduit les noms de colonnes en français"""
    translations = {
        'date': 'date',
        'nb_new_customers': 'nombre de nouveaux clients',
        'revenue': 'chiffre d\'affaires',
        'profit': 'bénéfice',
        'avg_order_value': 'valeur moyenne de commande',
        'nb_operations': 'nombre d\'opérations',
        'month': 'mois',
        'total_revenue': 'chiffre d\'affaires total',
        'avg_revenue': 'chiffre d\'affaires moyen',
        'total_profit': 'bénéfice total',
        'avg_profit': 'bénéfice moyen'
    }
    return translations.get(column_name.lower(), column_name)

def format_french_number(value) -> str:
    """Formate les nombres selon les conventions françaises"""
    if value is None:
        return "non disponible"
    
    if isinstance(value, (int, float)):
        if value == int(value):
            value = int(value)
        else:
            value = round(float(value), 2)
        
        # Format français : espace pour milliers, virgule pour décimales
        if isinstance(value, float):
            return f"{value:,.2f}".replace(",", " ").replace(".", ",")
        else:
            return f"{value:,}".replace(",", " ")
    
    return str(value)
