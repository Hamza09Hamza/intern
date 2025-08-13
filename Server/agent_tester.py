from service.agent import ask_agent, ask_simple
from service.tools import run_sql, get_database_schema

def test_tools_directly():
    """Test the database tools directly without any agent."""
    print("TESTING TOOLS DIRECTLY:")
    print("=" * 50)

    # 1. Schema
    print("1. Database Schema:")
    schema = get_database_schema.invoke("")  
    print(schema[:500] + "...\n")

    # 2. Direct SQL Queries
    direct_queries = [
        "SELECT COUNT(*) FROM employees",
        "SELECT name FROM departments",
        "SELECT name, salary FROM employees ORDER BY salary DESC LIMIT 1"
    ]
    print("2. Direct SQL Queries:")
    for i, sql in enumerate(direct_queries, 1):
        print(f"{i}. {sql}")
        result = run_sql.invoke(sql)  
        print(f"   Result: {result.strip()}\n")

def test_simple_approach():
    """Test the simple LLM-to-SQL approach."""
    print("SIMPLE APPROACH (LLM generates SQL):")
    print("=" * 50)
    questions = [
        "How many employees do we have?",
        "List all departments",
        "Who has the highest salary?",
        "How many sales were made?",
        "Which projects are in progress?",
        "What's the average salary?",
        "Show me total sales amount"
    ]
    for q in questions:
        a = ask_simple(q)
        print(f"Q: {q}\nA: {a}\n")

def test_agent_approach():
    """Test the LangChain agent approach."""
    print("AGENT APPROACH (LangChain ReAct):")
    print("=" * 50)
    questions = [
        "How many employees do we have?",
        "List all departments",
        "Who has the highest salary?",
        "How many sales were made?",
        "Which projects are in progress?",
        "What's the average salary by department?",
        "Show me employees hired recently"
    ]
    for q in questions:
        a = ask_agent(q)
        print(f"Q: {q}\nA: {a}\n")

def compare_approaches():
    """Compare both approaches."""
    print("COMPARISON TEST:")
    print("=" * 50)
    for q in [
        "How many employees do we have?",
        "Who has the highest salary?",
        "What's the total sales amount?"
    ]:
        print(f"Question: {q}")
        print("-" * 30)
        print("Simple approach:")
        print(ask_simple(q))
        print("\nAgent approach:")
        print(ask_agent(q))
        print("\n" + "=" * 50 + "\n")

def main():
    print("DATABASE QUERY SYSTEM TESTS")
    print("=" * 60)
    test_simple_approach()
    test_agent_approach()

if __name__ == "__main__":
    main()
