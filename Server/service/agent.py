from langchain_community.llms import Ollama
from langchain.agents import initialize_agent, Tool, AgentType
from .tools import run_sql, get_database_schema, simple_query

def create_agent():
    """Create and return a LangChain agent with database tools."""
    
    # 1. Set up the local LLM
    llm = Ollama(
        model="phi3",
        temperature=0.0,  # Very deterministic for SQL generation
    )
    
    # 2. Create tools for the agent
    sql_tool = Tool(
        name="query_database",
        func=run_sql,
        description="Execute SQL queries on company database. Input must be a valid SQL SELECT statement."
    )
    
    schema_tool = Tool(
        name="get_schema",
        func=get_database_schema,
        description="Get the complete database schema with table structures and sample data."
    )
    
    # 3. Create agent with both tools
    agent = initialize_agent(
        tools=[schema_tool, sql_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False, 
        handle_parsing_errors=True,
        max_iterations=3,  
        early_stopping_method="generate"
    )
    
    return agent

def ask_agent(question: str):
    """Ask a question using the LangChain agent."""
    try:
        agent = create_agent()
        response = agent.invoke({"input": question})
        return response.get('output', 'No output received')
    except Exception as e:
        # Fallback to simple query if agent fails
        return simple_query(question)

def ask_simple(question: str):
    """Ask a question using the simple LLM-to-SQL approach."""
    return simple_query(question)