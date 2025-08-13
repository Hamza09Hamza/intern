# database.py - Database creation and setup
import sqlite3
import random
from datetime import datetime, timedelta

def create_company_database():
    """Create a simple company database with basic tables."""
    
    print("Creating company.db...")
    
    # Connect to database (creates file if it doesn't exist)
    conn = sqlite3.connect("company.db")
    cursor = conn.cursor()
    
    # Delete existing tables if they exist
    cursor.execute("DROP TABLE IF EXISTS sales")
    cursor.execute("DROP TABLE IF EXISTS performance_reviews") 
    cursor.execute("DROP TABLE IF EXISTS projects")
    cursor.execute("DROP TABLE IF EXISTS employees")
    cursor.execute("DROP TABLE IF EXISTS departments")
    
    # Create departments table
    cursor.execute("""
        CREATE TABLE departments (
            id INTEGER PRIMARY KEY,
            name TEXT
        )
    """)
    
    # Create employees table
    cursor.execute("""
        CREATE TABLE employees (
            id INTEGER PRIMARY KEY,
            name TEXT,
            department_id INTEGER,
            hire_date TEXT,
            salary REAL,
            FOREIGN KEY(department_id) REFERENCES departments(id)
        )
    """)
    
    # Create sales table
    cursor.execute("""
        CREATE TABLE sales (
            id INTEGER PRIMARY KEY,
            employee_id INTEGER,
            amount REAL,
            sale_date TEXT,
            FOREIGN KEY(employee_id) REFERENCES employees(id)
        )
    """)
    
    # Create projects table
    cursor.execute("""
        CREATE TABLE projects (
            id INTEGER PRIMARY KEY,
            name TEXT,
            department_id INTEGER,
            status TEXT,
            deadline TEXT,
            FOREIGN KEY(department_id) REFERENCES departments(id)
        )
    """)
    
    # Create performance reviews table
    cursor.execute("""
        CREATE TABLE performance_reviews (
            id INTEGER PRIMARY KEY,
            employee_id INTEGER,
            review_date TEXT,
            score INTEGER,
            reviewer TEXT,
            FOREIGN KEY(employee_id) REFERENCES employees(id)
        )
    """)
    
    print("Adding sample data...")
    
    # Add departments
    departments = ["Engineering", "Marketing", "Sales", "HR", "Finance"]
    for i, dept in enumerate(departments, 1):
        cursor.execute("INSERT INTO departments VALUES (?, ?)", (i, dept))
    
    # Add employees
    employees = [
        "Alice Johnson", "Bob Smith", "Charlie Brown", "Diana Prince", "Eve Wilson",
        "Frank Miller", "Grace Lee", "Henry Davis", "Ivy Chen", "Jack Wilson",
        "Kate Taylor", "Liam Jones", "Mia Garcia", "Noah Martinez", "Olivia Lopez"
    ]
    
    for i, name in enumerate(employees, 1):
        dept_id = random.randint(1, 5)  # Random department
        hire_date = datetime.now() - timedelta(days=random.randint(100, 1000))
        salary = random.randint(40000, 100000)
        
        cursor.execute("""
            INSERT INTO employees VALUES (?, ?, ?, ?, ?)
        """, (i, name, dept_id, hire_date.strftime("%Y-%m-%d"), salary))
    
    # Add sales data
    for i in range(1, 101):  # 100 sales records
        employee_id = random.randint(1, len(employees))
        amount = round(random.uniform(500, 5000), 2)
        sale_date = datetime.now() - timedelta(days=random.randint(0, 365))
        
        cursor.execute("""
            INSERT INTO sales VALUES (?, ?, ?, ?)
        """, (i, employee_id, amount, sale_date.strftime("%Y-%m-%d")))
    
    # Add projects
    project_names = [
        "Website Redesign", "Mobile App", "Database Upgrade", "Marketing Campaign",
        "Training Program", "Security Audit", "Cost Reduction", "New Product Launch"
    ]
    
    statuses = ["completed", "in_progress", "cancelled"]
    
    for i, project_name in enumerate(project_names, 1):
        dept_id = random.randint(1, 5)
        status = random.choice(statuses)
        deadline = datetime.now() + timedelta(days=random.randint(30, 200))
        
        cursor.execute("""
            INSERT INTO projects VALUES (?, ?, ?, ?, ?)
        """, (i, project_name, dept_id, status, deadline.strftime("%Y-%m-%d")))
    
    # Add performance reviews
    for i in range(1, 51):  # 50 reviews
        employee_id = random.randint(1, len(employees))
        review_date = datetime.now() - timedelta(days=random.randint(0, 365))
        score = random.randint(1, 10)
        reviewer = random.choice(["Manager A", "Manager B", "HR Director"])
        
        cursor.execute("""
            INSERT INTO performance_reviews VALUES (?, ?, ?, ?, ?)
        """, (i, employee_id, review_date.strftime("%Y-%m-%d"), score, reviewer))
    
    # Save changes and close
    conn.commit()
    conn.close()
    
    print("✅ Database created successfully!")
    print("Tables created: departments, employees, sales, projects, performance_reviews")

if __name__ == "__main__":
    create_company_database()