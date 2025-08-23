# conversational_responder.py
from typing import Dict, Any, List
from datetime import datetime
import json
from llmanalyzer import ImprovedLLMAnalyzer
from nluprocessor import ImprovedNLUProcessor
from sqlexecutor import SQLExecutor

class ConversationalResponder:
    """Converts raw SQL results into human-like conversational responses"""
    
    def __init__(self, llm_model=None):
        self.llm_model = llm_model
        
    def generate_conversational_response(self, 
                                       question: str, 
                                       sql_result: Dict[str, Any],
                                       context: str = "") -> str:
        """
        Transform SQL results into natural, conversational responses
        """
        if not sql_result.get('success'):
            return self._handle_error_conversationally(question, sql_result.get('error', ''))
        
        # Extract key information
        data = sql_result.get('data', [])
        columns = sql_result.get('columns', [])
        row_count = sql_result.get('row_count', 0)
        
        if row_count == 0:
            return self._handle_no_results(question)
        
        # Use LLM to generate conversational response
        if self.llm_model:
            return self._llm_conversational_response(question, data, columns, context)
        else:
            return self._template_based_response(question, data, columns)
    
    def _llm_conversational_response(self, question: str, data: List, columns: List, context: str) -> str:
        """Use LLM to generate natural conversational response"""
        
        # Prepare data summary for LLM
        data_summary = self._prepare_data_summary(data, columns, max_rows=10)
        
        prompt = f"""You are a helpful AI assistant analyzing company data. Your job is to provide natural, conversational responses to user questions about employee data.

            IMPORTANT RULES:
            1. Speak naturally like a human colleague would
            2. Use conversational phrases like "Let me tell you about...", "Here's what I found...", "Interesting..."
            3. Provide insights and observations, not just raw data
            4. Keep it engaging and easy to understand
            5. If there are many results, summarize the key findings and mention there are more
            6. Use natural language for numbers (e.g., "about 100" instead of "exactly 99")
            7. Make it sound like you're having a conversation, not reading a report
            8. Add personality but stay professional

            CONTEXT: {context}

            USER QUESTION: {question}

            DATA FOUND:
            Columns: {', '.join(columns)}
            Number of records: {len(data)}

            Sample results:
            {data_summary}

            Generate a conversational response (2-4 sentences) that answers their question naturally:"""

        try:
            response = self.llm_model.invoke(prompt)
            return self._clean_response(response)
        except Exception as e:
            print(f"⚠️ LLM response generation failed: {e}")
            return self._template_based_response(question, data, columns)
    
    def _template_based_response(self, question: str, data: List, columns: List) -> str:
        """Fallback template-based conversational responses"""
        
        row_count = len(data)
        
        # Detect question type and respond appropriately
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['highest', 'top', 'maximum', 'best']):
            return self._handle_ranking_question(data, columns, 'highest')
        
        elif any(word in question_lower for word in ['lowest', 'minimum', 'worst', 'bottom']):
            return self._handle_ranking_question(data, columns, 'lowest')
        
        elif any(word in question_lower for word in ['average', 'mean']):
            return self._handle_average_question(data, columns)
        
        elif any(word in question_lower for word in ['count', 'how many', 'number']):
            return self._handle_count_question(data, columns, row_count)
        
        elif any(word in question_lower for word in ['each', 'per', 'by department', 'by']):
            return self._handle_grouping_question(data, columns, row_count)
        
        else:
            return self._handle_general_question(data, columns, row_count)
    
    def _handle_ranking_question(self, data: List, columns: List, rank_type: str) -> str:
        """Handle highest/lowest type questions"""
        if not data:
            return "I couldn't find any data to rank for you."
        
        top_result = data[0]  # Assuming data is already sorted
        
        if len(columns) >= 4:  # dept, first_name, last_name, salary
            dept, first_name, last_name, value = top_result[:4]
            return f"Great question! The {rank_type} paid employee is {first_name} {last_name} from {dept}, earning ${value:,.2f}. I found {len(data)} employees in total, so there's quite a range in salaries across the company."
        
        return f"Here's what I found - the {rank_type} result shows {', '.join(str(x) for x in top_result[:3])}. There are {len(data)} records total if you'd like to see more details."
    
    def _handle_count_question(self, data: List, columns: List, row_count: int) -> str:
        """Handle counting questions"""
        if row_count == 1:
            return f"I found exactly one result for your question. Here it is: {', '.join(str(x) for x in data[0])}."
        else:
            return f"Let me count that for you - I found {row_count} results. The breakdown shows some interesting patterns across different categories."
    
    def _handle_grouping_question(self, data: List, columns: List, row_count: int) -> str:
        """Handle 'per department' or grouping questions"""
        if row_count <= 5:
            return f"Here's the breakdown you asked for - I found {row_count} different groups. The results show some interesting variations across the categories."
        else:
            sample_groups = data[:3]
            return f"I've analyzed the data by groups and found {row_count} different categories. The top few results show {', '.join(str(x[0]) for x in sample_groups)} leading the way, with quite a bit of variation across all groups."
    
    def _handle_average_question(self, data: List, columns: List) -> str:
        """Handle average/mean questions"""
        if data and len(data[0]) >= 2:
            avg_value = data[0][-1]  # Assuming average is last column
            return f"Looking at the averages across the data, I calculated approximately ${avg_value:,.2f} as the typical value. The data shows {len(data)} different categories, so there's definitely some variation worth exploring."
        return f"I calculated the averages you requested and found {len(data)} different groups with varying results."
    
    def _handle_general_question(self, data: List, columns: List, row_count: int) -> str:
        """Handle general listing questions"""
        if row_count <= 10:
            return f"I found {row_count} results that match your question. The data includes information about {', '.join(columns)} and shows some interesting patterns."
        else:
            return f"Your question returned quite a lot of data - {row_count} records in total! The results include {', '.join(columns[:3])} and many other details. Would you like me to focus on any specific aspect?"
    
    def _handle_no_results(self, question: str) -> str:
        """Handle when no results are found"""
        responses = [
            f"Hmm, I couldn't find any data that matches your question about '{question}'. The database might not have that specific information, or we might need to phrase the question differently.",
            f"That's interesting - your question didn't return any results. It's possible the data you're looking for isn't in our current database, or maybe we need to approach it from a different angle.",
            f"I searched thoroughly but couldn't find anything matching '{question}'. This could mean the data isn't available, or perhaps we need to adjust how we're looking for it."
        ]
        return responses[0]  # Could randomize for variety
    
    def _handle_error_conversationally(self, question: str, error: str) -> str:
        """Handle errors in a conversational way"""
        return f"I ran into a problem trying to answer your question about '{question}'. It seems there was a technical issue with how I interpreted your request. Could you try rephrasing it, or let me know if you'd like to try something else?"
    
    def _prepare_data_summary(self, data: List, columns: List, max_rows: int = 10) -> str:
        """Prepare a clean summary of data for LLM"""
        if not data:
            return "No data available"
        
        summary_lines = []
        for i, row in enumerate(data[:max_rows]):
            row_dict = dict(zip(columns, row))
            summary_lines.append(f"{i+1}. {json.dumps(row_dict, default=str)}")
        
        if len(data) > max_rows:
            summary_lines.append(f"... and {len(data) - max_rows} more records")
        
        return "\n".join(summary_lines)
    
    def _clean_response(self, response: str) -> str:
        """Clean up LLM response for TTS"""
        # Remove any markdown or formatting
        response = response.replace('**', '').replace('*', '')
        response = response.replace('```', '').replace('`', '')
        
        # Remove extra whitespace
        response = ' '.join(response.split())
        
        # Ensure it ends properly
        if not response.endswith(('.', '!', '?')):
            response += '.'
        
        return response


# Enhanced NLU System with Conversational Response
class EnhancedConversationalNLUSystem:
    """Enhanced system that provides human-like conversational responses"""
    
    def __init__(self, db_path: str = "company.db", model: str = "mistral:latest"):
        # Import your existing classes

        
        self.db_path = db_path
        self.model = model
        self.analyzer = ImprovedLLMAnalyzer(db_path=db_path, model=model)
        self.processor = None
        self.executor = None
        self.responder = None
        self.initialized = False
        self.query_history = []
        
    def initialize(self) -> bool:
        """Initialize the enhanced conversational system"""
        print(f"🚀 Initializing Enhanced Conversational NLU System...")
        
        try:
            # Analyze database structure
            ctx = self.analyzer.analyze_database()
            
            # Initialize components
            self.processor = ImprovedNLUProcessor(ctx, self.analyzer, model=self.model)
            self.executor = SQLExecutor(db_path=self.db_path)
            
            # Initialize conversational responder with LLM
            self.responder = ConversationalResponder(llm_model=self.processor.llm)
            
            self.initialized = True
            print("✅ Enhanced conversational system ready!")
            print("🎯 Now providing human-like responses perfect for text-to-speech!")
            return True
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            return False
    
    def query(self, natural_language: str, include_technical_details: bool = False) -> str:
        """Process query and return conversational response"""
        if not self.initialized:
            return "I'm sorry, but I'm not ready to help you yet. Let me get initialized first."
        
        print(f"\n🎯 Processing: {natural_language}")
        
        # Generate SQL
        sql = self.processor.natural_language_to_sql(natural_language)
        
        if sql.startswith("ERROR:"):
            return "I'm having trouble understanding your question. Could you try rephrasing it in a different way?"
        
        # Execute SQL
        sql_result = self.executor.execute_sql(sql)
        
        # Generate conversational response
        conversational_response = self.responder.generate_conversational_response(
            question=natural_language,
            sql_result=sql_result,
            context="employee database with departments, salaries, and job information"
        )
        
        # Store in history
        self.query_history.append({
            "question": natural_language,
            "sql": sql,
            "response": conversational_response,
            "success": sql_result.get("success", False),
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Show technical details if requested (for debugging)
        if include_technical_details:
            print(f"🔧 SQL: {sql}")
            print(f"📊 Results: {sql_result.get('row_count', 0)} rows")
        
        print(f"💬 Response: {conversational_response}")
        return conversational_response
    
    def interactive_mode(self):
        """Enhanced interactive mode with conversational responses"""
        if not self.initialize():
            print("❌ Failed to initialize system")
            return
        
        print("\n🤖 Enhanced Voice Assistant Ready!")
        print("💡 I can answer questions about employees, departments, and salaries.")
        print("🎙️ My responses are optimized for text-to-speech conversion!")
        print("\nTry asking things like:")
        print("   - Who makes the most money in each department?")
        print("   - How many people work in sales?")
        print("   - What's the average salary across the company?")
        print("\nType 'exit' to quit, 'debug' for technical details\n")
        
        debug_mode = False
        
        while True:
            try:
                question = input("🗣️  Ask me: ").strip()
                
                if not question:
                    continue
                    
                if question.lower() in ["exit", "quit", "bye"]:
                    print("👋 Thanks for chatting! Have a great day!")
                    break
                    
                if question.lower() == "debug":
                    debug_mode = not debug_mode
                    print(f"🔧 Debug mode {'ON' if debug_mode else 'OFF'}")
                    continue
                
                # Get conversational response
                response = self.query(question, include_technical_details=debug_mode)
                
                # This is your TTS-ready response!
                print(f"\n🎤 TTS Ready: {response}\n")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye! Thanks for using the voice assistant!")
                break
            except Exception as e:
                print(f"❌ Something went wrong: {e}")


# Simple TTS Integration Example
def add_tts_support(text: str, engine='pyttsx3'):
    """
    Example of how to add TTS to your responses
    Install: pip install pyttsx3
    """
    try:
        if engine == 'pyttsx3':
            import pyttsx3
            tts_engine = pyttsx3.init()
            tts_engine.say(text)
            tts_engine.runAndWait()
        # You could also use other TTS engines like gTTS, Azure Speech, etc.
    except ImportError:
        print("💡 Install pyttsx3 for TTS support: pip install pyttsx3")
    except Exception as e:
        print(f"TTS Error: {e}")


# Usage example
if __name__ == "__main__":
    system = EnhancedConversationalNLUSystem(db_path="company.db", model="mistral:latest")
    system.interactive_mode()