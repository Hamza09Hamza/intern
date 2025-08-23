from nlusystem import NLUSystem
from mysqlservice.config import MYSQL_CONFIG

def main():
    """Main function to run the MySQL NLU system"""
    
    # You can customize MySQL configuration here
    mysql_config=MYSQL_CONFIG
    # Initialize the system
    system = NLUSystem(
        mysql_config=mysql_config, 
        model="mistral:latest"  # Change to your preferred model
    )
    
    # Start interactive mode
    system.interactive_mode()

def test_connection():
    """Test MySQL connection with your configuration"""
    from sqlexecutor import MySQLExecutor
    
    mysql_config = MYSQL_CONFIG
    
    print("🔍 Testing MySQL connection...")
    executor = MySQLExecutor(mysql_config=mysql_config)
    
    if executor.test_connection():
        print("✅ MySQL connection successful!")
        
        # Show available tables
        result = executor.execute_sql("SHOW TABLES")
        if result['success']:
            tables = [row[0] for row in result['data']]
            print(f"📊 Found {len(tables)} tables: {', '.join(tables)}")
        else:
            print(f"❌ Could not list tables: {result['error']}")
    else:
        print("❌ MySQL connection failed!")

if __name__ == "__main__":
    # Uncomment the line below to test connection first
    # test_connection()
    
    # Run the main application
    main()