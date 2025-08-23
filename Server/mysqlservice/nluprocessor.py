import re
from typing import Optional, Tuple, List
from langchain_ollama import OllamaLLM
from mysqlservice.databasecontext import DatabaseContext
from mysqlservice.llmanalyzer import ImprovedMySQLLLMAnalyzer
from mysqlservice.config import MYSQL_CONFIG

try:
    import mysql.connector
    from mysql.connector import Error as MySQLError
    HAS_MYSQL = True
except Exception:
    mysql = None
    MySQLError = Exception
    HAS_MYSQL = False

class ImprovedMySQLNLUProcessor:
    def __init__(self, db_context: DatabaseContext, analyzer: ImprovedMySQLLLMAnalyzer, model: str = "mistral:latest", mysql_config: dict = None):
        self.context = db_context
        self.analyzer = analyzer
        self.model = model
        self.mysql_config = mysql_config or MYSQL_CONFIG
        self.llm = None
        
        if not HAS_MYSQL:
            raise ImportError("mysql-connector-python is required. Install with: pip install mysql-connector-python")
            
        try:
            self.llm = OllamaLLM(model=self.model, temperature=0.0)
        except Exception as e:
            print(f"⚠️ Could not init OllamaLLM: {e}")

    def get_connection(self):
        """Create MySQL connection"""
        try:
            return mysql.connector.connect(**self.mysql_config)
        except MySQLError as e:
            print(f"❌ MySQL connection failed: {e}")
            raise

    def natural_language_to_sql(self, question: str, max_retries: int = 3) -> str:
        """Enhanced SQL generation with better reasoning"""
        
        # Get relevant schema snippets
        snippets = self.analyzer.retrieve_relevant_snippets(question, top_k=4)
        
        # Build the enhanced prompt
        prompt = self._build_enhanced_sql_prompt(snippets, question)
        
        valid_sql = None
        last_sql = ""
        for attempt in range(max_retries + 1):
            print(f"🧠 LLM attempt {attempt + 1}/{max_retries + 1}")
            
            try:
                # Get LLM response
                raw_response = self.llm.invoke(prompt)
                sql_query = self._extract_sql_from_response(raw_response)
                
                if not sql_query:
                    print(f"⚠️ No SQL extracted from response: {raw_response[:200]}...")
                    continue
                
                print(f"📝 Generated SQL: {sql_query}")
                last_sql = sql_query
                
                # Validate the SQL - FIXED: Changed from _validate_sql to _validate_mysql_sql
                is_valid, error_msg = self._validate_mysql_sql(sql_query)
                
                if is_valid:
                    print("✅ SQL validation passed!")
                    valid_sql = sql_query
                    return valid_sql
                else:
                    print(f"❌ SQL validation failed: {error_msg}")
                    if attempt < max_retries:
                        # Add feedback to prompt for next attempt
                        prompt += f"\n\nPREVIOUS ATTEMPT FAILED: {error_msg}\nPlease fix this issue in your next attempt.\n"
                        continue
                
            except Exception as e:
                print(f"⚠️ Error during SQL generation: {e}")
                continue
        
        return valid_sql or f"ERROR: Could not generate valid SQL after {max_retries + 1} attempts"
    def _build_enhanced_sql_prompt(self, schema_snippets: List[str], question: str) -> str:
        """Build a comprehensive prompt in French that helps the LLM reason better"""
    
        schema_text = "\n\n".join(schema_snippets)
        
        prompt = f"""Tu es un analyste SQL expert spécialisé dans les données de KPI de séries temporelles. Ton travail est de convertir les questions en langage naturel français en requêtes SQL précises pour une base de données MySQL contenant des métriques commerciales quotidiennes.

    DESCRIPTION DE LA BASE DE DONNÉES :
    Cette base de données suit les indicateurs clés de performance (KPI) quotidiens d'une entreprise de mars 2025 à août 2025. Elle a une seule table 'daily_kpis_wide' avec la structure suivante :
    - date : La date des métriques (clé primaire, format 'YYYY-MM-DD').
    - nb_new_customers : Nombre de nouveaux clients acquis ce jour-là (entier).
- revenue : Chiffre d'affaires total du jour (décimal/réel).
- profit : Bénéfice total du jour (décimal/réel).
- avg_order_value : Valeur moyenne des commandes ce jour-là (décimal/réel).
- nb_operations : Nombre d'opérations effectuées ce jour-là (entier).

Toutes les données sont quotidiennes, donc les requêtes impliquent souvent le filtrage par plages de dates, le regroupement par mois/année (en utilisant DATE_FORMAT), ou l'agrégation (SUM, AVG, MAX, MIN) sur des périodes de temps.

SCHÉMA DE LA BASE DE DONNÉES :
{schema_text}

RÈGLES CRITIQUES :
1. Toujours commencer par SELECT et inclure une clause FROM appropriée (ex: FROM daily_kpis_wide).
2. Aucun JOIN nécessaire puisqu'il n'y a qu'une seule table.
3. Pour les questions sur "par mois", "chaque mois", ou "mensuel", utiliser GROUP BY DATE_FORMAT(date, '%Y-%m') et alias comme 'mois'.
4. Pour "le plus haut/le plus bas" (ex: jour de revenus max), utiliser ORDER BY DESC/ASC avec LIMIT 1, ou MAX/MIN avec sous-requêtes si groupé.
5. Utiliser l'agrégation appropriée : SUM pour les totaux, AVG pour les moyennes (arrondir à 2 décimales avec ROUND(AVG(...), 2)), COUNT pour les comptages, MAX/MIN pour les extrêmes.
6. Pour les filtres de date, utiliser WHERE date BETWEEN 'YYYY-MM-DD' AND 'YYYY-MM-DD', ou date >= 'YYYY-MM-DD' pour "depuis". Utiliser DATE_FORMAT(date, '%Y-%m') = 'YYYY-MM' pour les mois.
7. Inclure ORDER BY pour un tri logique (ex: par date DESC pour les plus récents en premier).
8. Générer uniquement des requêtes SELECT, jamais INSERT/UPDATE/DELETE.
9. Si la question ne peut pas être répondue avec le schéma (ex: colonnes inexistantes), ne pas inventer de données.

PROCESSUS DE RAISONNEMENT :
1. Identifier les métriques demandées (ex: chiffre d'affaires, bénéfice).
2. Déterminer si un filtrage temporel est nécessaire (ex: mois spécifique, plage).
3. Vérifier le regroupement (ex: par mois) ou l'agrégation (ex: total, moyenne).
4. Décider du tri ou des limites (ex: top 5 jours par chiffre d'affaires).
5. S'assurer que le SQL est valide pour MySQL.

EXEMPLES DE BONNE STRUCTURE SQL :

Pour "chiffre d'affaires total en mars 2025" :
SELECT SUM(revenue) AS total_revenue
FROM daily_kpis_wide
WHERE date BETWEEN '2025-03-01' AND '2025-03-31';

Pour "bénéfice moyen par mois, trié par le plus élevé" :
SELECT DATE_FORMAT(date, '%Y-%m') AS mois, ROUND(AVG(profit), 2) AS benefice_moyen
FROM daily_kpis_wide
GROUP BY mois
ORDER BY benefice_moyen DESC;

Maintenant analyse cette question et génère le SQL approprié :

QUESTION : {question}

Pense étape par étape :
1. Quelles métriques et informations sont demandées ?
2. Quelle plage de dates ou regroupement est impliqué ?
3. Quelle agrégation ou filtrage est nécessaire ?
4. Comment les résultats doivent-ils être triés ou limités ?

Génère uniquement la requête SQL (sans explications) :"""

        return prompt

    def _extract_sql_from_response(self, response: str) -> str:
        """Extract clean SQL from LLM response"""
        if not response:
            return ""
        
        # Remove code blocks
        response = re.sub(r'```sql\n?', '', response)
        response = re.sub(r'```\n?', '', response)
        
        # Find SQL query starting with SELECT
        sql_match = re.search(r'(SELECT\b.*?)(?:\n\n|$)', response, re.IGNORECASE | re.DOTALL)
        if sql_match:
            sql = sql_match.group(1).strip()
            # Remove trailing semicolon if present
            sql = sql.rstrip(';').strip()
            return sql
        
        # Fallback: look for any line starting with SELECT
        lines = response.split('\n')
        for line in lines:
            if line.strip().upper().startswith('SELECT'):
                return line.strip().rstrip(';')
        
        return ""

    def _validate_mysql_sql(self, sql: str) -> Tuple[bool, str]:
        """Comprehensive MySQL SQL validation with better FROM clause detection"""
        if not sql:
            return False, "Empty SQL query"
        
        # Clean the SQL and normalize whitespace
        sql_clean = ' '.join(sql.strip().split())  # This removes extra whitespace and normalizes
        sql_upper = sql_clean.upper()
        
        print(f"🔍 Validating SQL: '{sql_clean}'")  # Debug print to see what we're validating
        print(f"🔍 Uppercase SQL: '{sql_upper}'")  # Debug print
        
        # Must be a SELECT query
        if not sql_upper.startswith('SELECT'):
            return False, "Query must start with SELECT"
        
        # Better FROM clause detection - check for FROM followed by word boundary
        import re
        if not re.search(r'\bFROM\b', sql_upper):
            return False, "Query must include FROM clause"
        
        # Check for dangerous operations
        dangerous_keywords = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE', 'TRUNCATE', 'GRANT', 'REVOKE']
        for keyword in dangerous_keywords:
            if re.search(r'\b' + keyword + r'\b', sql_upper):
                return False, f"Query contains forbidden keyword: {keyword}"
        
        # Validate against database schema
        try:
            # Test query syntax by using EXPLAIN
            conn = self.get_connection()
            cur = conn.cursor()
            
            # Use the cleaned SQL for execution
            explain_query = f"EXPLAIN {sql_clean}"
            print(f"🔍 Running EXPLAIN: {explain_query}")  # Debug print
            
            cur.execute(explain_query)
            cur.fetchall()  # Fetch results to ensure query completes
            conn.close()
            
            print("✅ SQL passed EXPLAIN test")  # Debug print
            return True, "Valid SQL"
            
        except MySQLError as e:
            error_msg = f"MySQL syntax error: {str(e)}"
            print(f"❌ MySQL Error: {error_msg}")  # Debug print
            return False, error_msg
        except Exception as e:
            error_msg = f"Validation error: {str(e)}"
            print(f"❌ General Error: {error_msg}")  # Debug print
            return False, error_msg     
            