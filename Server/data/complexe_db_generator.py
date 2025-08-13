# Fixing asset_history insert placeholder mismatch and rerunning generation (fresh DB)
import sqlite3, os
from pathlib import Path

OUT_PATH = "./company.db"
if os.path.exists(OUT_PATH):
    os.remove(OUT_PATH)

import random, uuid, json
from datetime import datetime, timedelta
from faker import Faker
import numpy as np

fake = Faker()
random.seed(42)
np.random.seed(42)
SCALE_EMPLOYEES = 10000
BATCH = 2000

def pragmas(conn):
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode = WAL;")
    cur.execute("PRAGMA synchronous = NORMAL;")
    cur.execute("PRAGMA temp_store = MEMORY;")
    cur.execute("PRAGMA foreign_keys = ON;")
    conn.commit()

def create_schema(conn):
    cur = conn.cursor()
    cur.executescript("""
    PRAGMA foreign_keys = OFF;
    DROP TABLE IF EXISTS audit_log;
    DROP TABLE IF EXISTS emp_projects;
    DROP TABLE IF EXISTS employee_skills;
    DROP TABLE IF EXISTS salaries;
    DROP TABLE IF EXISTS events;
    DROP TABLE IF EXISTS addresses;
    DROP TABLE IF EXISTS asset_history;
    DROP TABLE IF EXISTS assets;
    DROP TABLE IF EXISTS contracts;
    DROP TABLE IF EXISTS clients;
    DROP TABLE IF EXISTS tasks;
    DROP TABLE IF EXISTS projects;
    DROP TABLE IF EXISTS skills;
    DROP TABLE IF EXISTS employees;
    DROP TABLE IF EXISTS departments;
    PRAGMA foreign_keys = ON;
    """)
    cur.executescript("""
    CREATE TABLE departments (
        dept_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL UNIQUE,
        location TEXT,
        budget REAL,
        created_at TEXT
    );

    CREATE TABLE employees (
        emp_id INTEGER PRIMARY KEY,
        dept_id INTEGER,
        emp_uuid TEXT UNIQUE,
        first_name TEXT,
        last_name TEXT,
        email TEXT UNIQUE,
        hire_date TEXT,
        role TEXT,
        salary REAL,
        manager_id INTEGER,
        profile_json TEXT,
        active INTEGER DEFAULT 1,
        created_at TEXT,
        updated_at TEXT,
        FOREIGN KEY(dept_id) REFERENCES departments(dept_id) ON DELETE SET NULL,
        FOREIGN KEY(manager_id) REFERENCES employees(emp_id) ON DELETE SET NULL
    );

    CREATE TABLE projects (
        proj_id INTEGER PRIMARY KEY,
        dept_id INTEGER,
        name TEXT,
        description TEXT,
        start_date TEXT,
        end_date TEXT,
        status TEXT,
        budget REAL,
        config_json TEXT,
        created_at TEXT,
        FOREIGN KEY(dept_id) REFERENCES departments(dept_id) ON DELETE SET NULL
    );

    CREATE TABLE tasks (
        task_id INTEGER PRIMARY KEY,
        proj_id INTEGER,
        assigned_to INTEGER,
        title TEXT,
        description TEXT,
        status TEXT,
        priority INTEGER,
        due_date TEXT,
        created_at TEXT,
        FOREIGN KEY(proj_id) REFERENCES projects(proj_id) ON DELETE CASCADE,
        FOREIGN KEY(assigned_to) REFERENCES employees(emp_id) ON DELETE SET NULL
    );

    CREATE TABLE clients (
        client_id INTEGER PRIMARY KEY,
        name TEXT,
        industry TEXT,
        contact_email TEXT,
        created_at TEXT
    );

    CREATE TABLE contracts (
        contract_id INTEGER PRIMARY KEY,
        client_id INTEGER,
        proj_id INTEGER,
        amount REAL,
        signed_date TEXT,
        terms_json TEXT,
        created_at TEXT,
        FOREIGN KEY(client_id) REFERENCES clients(client_id) ON DELETE CASCADE,
        FOREIGN KEY(proj_id) REFERENCES projects(proj_id) ON DELETE CASCADE
    );

    CREATE TABLE skills (
        skill_id INTEGER PRIMARY KEY,
        name TEXT UNIQUE
    );

    CREATE TABLE employee_skills (
        emp_id INTEGER,
        skill_id INTEGER,
        level INTEGER,
        PRIMARY KEY(emp_id, skill_id),
        FOREIGN KEY(emp_id) REFERENCES employees(emp_id) ON DELETE CASCADE,
        FOREIGN KEY(skill_id) REFERENCES skills(skill_id) ON DELETE CASCADE
    );

    CREATE TABLE emp_projects (
        emp_id INTEGER,
        proj_id INTEGER,
        assigned_at TEXT,
        role_on_project TEXT,
        allocation_percent INTEGER,
        PRIMARY KEY(emp_id, proj_id),
        FOREIGN KEY(emp_id) REFERENCES employees(emp_id) ON DELETE CASCADE,
        FOREIGN KEY(proj_id) REFERENCES projects(proj_id) ON DELETE CASCADE
    );

    CREATE TABLE assets (
        asset_id INTEGER PRIMARY KEY,
        name TEXT,
        type TEXT,
        owner_emp INTEGER,
        purchase_date TEXT,
        value REAL,
        metadata_json TEXT,
        FOREIGN KEY(owner_emp) REFERENCES employees(emp_id) ON DELETE SET NULL
    );

    CREATE TABLE asset_history (
        hist_id INTEGER PRIMARY KEY,
        asset_id INTEGER,
        change_date TEXT,
        note TEXT,
        FOREIGN KEY(asset_id) REFERENCES assets(asset_id) ON DELETE CASCADE
    );

    CREATE TABLE addresses (
        addr_id INTEGER PRIMARY KEY,
        emp_id INTEGER,
        line1 TEXT,
        city TEXT,
        state TEXT,
        postal_code TEXT,
        country TEXT,
        geo TEXT,
        is_primary INTEGER DEFAULT 0,
        FOREIGN KEY(emp_id) REFERENCES employees(emp_id) ON DELETE CASCADE
    );

    CREATE TABLE events (
        event_id INTEGER PRIMARY KEY,
        emp_id INTEGER,
        event_time TEXT,
        event_type TEXT,
        details_json TEXT,
        created_at TEXT,
        FOREIGN KEY(emp_id) REFERENCES employees(emp_id) ON DELETE CASCADE
    );

    CREATE TABLE salaries (
        sal_id INTEGER PRIMARY KEY,
        emp_id INTEGER,
        effective_date TEXT,
        salary REAL,
        reason TEXT,
        created_at TEXT,
        FOREIGN KEY(emp_id) REFERENCES employees(emp_id) ON DELETE CASCADE
    );

    CREATE TABLE audit_log (
        audit_id INTEGER PRIMARY KEY,
        table_name TEXT,
        action TEXT,
        row_id TEXT,
        diff TEXT,
        created_at TEXT
    );
    """)
    cur.executescript("""
    CREATE INDEX idx_employees_dept ON employees(dept_id);
    CREATE INDEX idx_employees_manager ON employees(manager_id);
    CREATE INDEX idx_projects_dept ON projects(dept_id);
    CREATE INDEX idx_tasks_proj ON tasks(proj_id);
    CREATE INDEX idx_emp_projects_proj ON emp_projects(proj_id);
    CREATE INDEX idx_salaries_emp ON salaries(emp_id);
    CREATE INDEX idx_events_emp_time ON events(emp_id, event_time);
    CREATE INDEX idx_assets_owner ON assets(owner_emp);
    """)
    conn.commit()

def batched_insert(conn, sql, rows):
    cur = conn.cursor()
    for i in range(0, len(rows), BATCH):
        cur.executemany(sql, rows[i:i+BATCH])
        conn.commit()

def generate_departments(n=25):
    rows = []
    now = datetime.utcnow().isoformat()
    for i in range(n):
        rows.append((f"Dept {i+1}", random.choice(['NY','CA','TX','DE','MA','Remote']), round(random.uniform(100000, 5000000),2), now))
    return rows

def generate_employees(n, dept_count):
    rows = []
    now = datetime.utcnow().isoformat()
    roles = ['Engineer','Senior Engineer','Manager','Director','HR','Sales','Analyst','QA','DevOps','Product']
    for _ in range(n):
        eid = str(uuid.uuid4())
        first = fake.first_name()
        last = fake.last_name()
        email = f"{first.lower()}.{last.lower()}.{random.randint(1,999999)}@example.com"
        hire = fake.date_between(start_date='-12y', end_date='today').isoformat()
        role = random.choices(roles, weights=[6,4,1.5,0.8,1,2,2,1,1,1])[0]
        salary = max(25000, float(np.random.lognormal(mean=10, sigma=0.5)))
        dept = random.randint(1, dept_count)
        manager = None
        profile = json.dumps({"bio": fake.sentence(nb_words=12), "skills_guess": random.sample(["python","sql","ml","k8s","aws","react","go"], k=random.randint(1,4))})
        rows.append((dept, eid, first, last, email, hire, role, salary, manager, profile, 1, now, now))
    return rows

def generate_projects(num_projects, dept_count):
    rows = []
    now = datetime.utcnow().isoformat()
    statuses = ['planned','active','completed','cancelled']
    for i in range(num_projects):
        name = f"Project {fake.word().title()} {i+1}"
        dept = random.randint(1, dept_count)
        start = fake.date_between(start_date='-6y', end_date='today').isoformat()
        end = (datetime.fromisoformat(start) + timedelta(days=random.randint(30, 1000))).isoformat()
        status = random.choices(statuses, weights=[1,4,3,0.5])[0]
        budget = round(abs(np.random.normal(loc=200000, scale=100000)),2)
        cfg = json.dumps({"stack": random.sample(["python","node","java","go","rust"], k=2), "priority": random.randint(1,5)})
        rows.append((dept, name, fake.sentence(nb_words=6), start, end, status, budget, cfg, now))
    return rows

def generate_tasks(num_projects, employees_count):
    rows = []
    now = datetime.utcnow().isoformat()
    statuses = ['Pending','In Progress','Done','Blocked']
    for proj in range(1, num_projects+1):
        num_tasks = random.randint(5, 20)
        for _ in range(num_tasks):
            assigned = random.randint(1, employees_count)
            title = fake.sentence(nb_words=6)
            desc = fake.paragraph(nb_sentences=2)
            status = random.choice(statuses)
            priority = random.randint(1,5)
            due = fake.date_between(start_date='-1y', end_date='+1y').isoformat()
            rows.append((proj, assigned, title, desc, status, priority, due, now))
    return rows

def generate_clients(num_clients):
    rows = []
    now = datetime.utcnow().isoformat()
    industries = ['Tech','Finance','Retail','Healthcare','Education','Manufacturing']
    for _ in range(num_clients):
        rows.append((fake.company(), random.choice(industries), fake.company_email(), now))
    return rows

def generate_contracts(num_contracts, num_clients, num_projects):
    rows = []
    now = datetime.utcnow().isoformat()
    for _ in range(num_contracts):
        client = random.randint(1, num_clients)
        proj = random.randint(1, num_projects)
        amount = round(random.uniform(10000, 2000000),2)
        signed = fake.date_between(start_date='-5y', end_date='today').isoformat()
        terms = json.dumps({"duration_months": random.randint(1,60), "sla": random.choice(["standard","premium","none"])})
        rows.append((client, proj, amount, signed, terms, now))
    return rows

def generate_skills(num_skills=60):
    base = ["Python","SQL","Java","C++","Go","Rust","Management","Design","Data Analysis","ML","NLP","DevOps","Kubernetes","AWS","GCP","Azure"]
    rows = []
    seen = set()
    for i in range(num_skills):
        name = base[i] if i < len(base) else fake.unique.word().title()
        if name in seen:
            name = fake.unique.word().title()
        seen.add(name)
        rows.append((name,))
    return rows

def generate_employee_skills(num_employees, num_skills):
    rows = []
    for emp in range(1, num_employees+1):
        count = random.randint(1, 6)
        skills = random.sample(range(1, num_skills+1), k=count)
        for s in skills:
            rows.append((emp, s, random.randint(1,5)))
    return rows

def generate_emp_projects(num_employees, num_projects, target_assignments):
    rows = []
    seen = set()
    for _ in range(target_assignments):
        emp = random.randint(1, num_employees)
        proj = random.randint(1, num_projects)
        key = (emp, proj)
        if key in seen:
            continue
        seen.add(key)
        assigned = fake.date_between(start_date='-3y', end_date='today').isoformat()
        role = random.choice(['contributor','lead','architect','tester','pm'])
        alloc = random.randint(5,100)
        rows.append((emp, proj, assigned, role, alloc))
    return rows

def generate_assets(num_assets, num_employees):
    rows = []
    now = datetime.utcnow().isoformat()
    types = ['Laptop','Server','Vehicle','Furniture','Phone','Monitor']
    for _ in range(num_assets):
        owner = random.randint(1, num_employees)
        name = random.choice(types) + " " + fake.word().title()
        atype = random.choice(['IT','Transport','Furniture'])
        purchase = fake.date_between(start_date='-6y', end_date='today').isoformat()
        value = round(random.uniform(100, 50000),2)
        meta = json.dumps({"warranty_years": random.randint(0,5), "serial": str(uuid.uuid4())})
        rows.append((name, atype, owner, purchase, value, meta))
    return rows

def generate_asset_history(num_assets):
    rows = []
    for aid in range(1, num_assets+1):
        for _ in range(random.randint(1,5)):
            change = fake.date_between(start_date='-5y', end_date='today').isoformat()
            note = fake.sentence(nb_words=8)
            rows.append((aid, change, note))
    return rows

def generate_addresses(num_employees):
    rows = []
    for emp in range(1, num_employees+1):
        addr_count = random.choices([0,1,2], weights=[0.05,0.8,0.15])[0]
        for i in range(addr_count):
            line1 = fake.street_address()
            city = fake.city()
            state = fake.state_abbr()
            pc = fake.postcode()
            country = fake.country()
            geo = f"{round(random.uniform(-90,90),6)},{round(random.uniform(-180,180),6)}"
            is_primary = 1 if i == 0 else 0
            rows.append((emp, line1, city, state, pc, country, geo, is_primary))
    return rows

def generate_events(num_employees, avg_per_emp=3):
    rows = []
    kinds = ['login','logout','promotion','policy_ack','access_request','incident','password_change']
    for emp in range(1, num_employees+1):
        count = np.random.poisson(lam=avg_per_emp)
        count = max(0, count)
        for _ in range(count):
            t = (datetime.utcnow() - timedelta(days=random.randint(0, 365*3), seconds=random.randint(0,86400))).isoformat()
            kind = random.choice(kinds)
            details = json.dumps({"ip": fake.ipv4_public(), "meta": fake.sentence(nb_words=6)})
            rows.append((emp, t, kind, details, datetime.utcnow().isoformat()))
    return rows

def generate_salaries_history(num_employees):
    rows = []
    for emp in range(1, num_employees+1):
        count = random.randint(1,4)
        for _ in range(count):
            eff = (datetime.utcnow() - timedelta(days=random.randint(0, 365*5))).date().isoformat()
            salary = round(max(20000, np.random.lognormal(mean=10, sigma=0.45)), 2)
            reason = random.choice(['hire','raise','promotion','correction'])
            rows.append((emp, eff, salary, reason, datetime.utcnow().isoformat()))
    return rows

def batched_insert(conn, sql, rows):
    cur = conn.cursor()
    for i in range(0, len(rows), BATCH):
        cur.executemany(sql, rows[i:i+BATCH])
        conn.commit()

def fill_db(conn, scale_emp):
    cur = conn.cursor()
    now = datetime.utcnow().isoformat()

    print("[*] Generating departments...")
    depts = generate_departments(25)
    batched_insert(conn, "INSERT OR IGNORE INTO departments (name, location, budget, created_at) VALUES (?, ?, ?, ?);", depts)

    dept_count = 25

    print(f"[*] Generating {scale_emp} employees (batches of {BATCH}) ...")
    employees = generate_employees(scale_emp, dept_count)
    batched_insert(conn, """INSERT OR IGNORE INTO employees (dept_id, emp_uuid, first_name, last_name, email, hire_date, role, salary, manager_id, profile_json, active, created_at, updated_at)
                             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);""", employees)

    print("[*] Assigning managers...")
    cur.execute("SELECT emp_id FROM employees;")
    emp_ids = [r[0] for r in cur.fetchall()]
    managers = random.sample(emp_ids, max(1, scale_emp // 20))
    updates = []
    for i, eid in enumerate(emp_ids):
        mgr = random.choice(managers) if random.random() < 0.9 else None
        updates.append((mgr, eid))
    for i in range(0, len(updates), BATCH):
        cur.executemany("UPDATE employees SET manager_id = ? WHERE emp_id = ?;", updates[i:i+BATCH])
        conn.commit()

    num_projects = max(1000, scale_emp // 10)
    print(f"[*] Generating {num_projects} projects...")
    projects = generate_projects(num_projects, dept_count)
    batched_insert(conn, """INSERT OR IGNORE INTO projects (dept_id, name, description, start_date, end_date, status, budget, config_json, created_at)
                             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);""", projects)

    print("[*] Generating tasks per project...")
    tasks = generate_tasks(num_projects, scale_emp)
    batched_insert(conn, """INSERT OR IGNORE INTO tasks (proj_id, assigned_to, title, description, status, priority, due_date, created_at)
                             VALUES (?, ?, ?, ?, ?, ?, ?, ?);""", tasks)

    num_clients = max(400, num_projects // 2)
    print(f"[*] Generating {num_clients} clients...")
    clients = generate_clients(num_clients)
    batched_insert(conn, "INSERT OR IGNORE INTO clients (name, industry, contact_email, created_at) VALUES (?, ?, ?, ?);", clients)

    num_contracts = max(800, num_projects // 1)
    print(f"[*] Generating {num_contracts} contracts...")
    contracts = generate_contracts(num_contracts, num_clients, num_projects)
    batched_insert(conn, "INSERT OR IGNORE INTO contracts (client_id, proj_id, amount, signed_date, terms_json, created_at) VALUES (?, ?, ?, ?, ?, ?);", contracts)

    num_skills = 60
    print(f"[*] Generating {num_skills} skills...")
    skills = generate_skills(num_skills)
    batched_insert(conn, "INSERT OR IGNORE INTO skills (name) VALUES (?);", skills)

    print("[*] Generating employee skills (many-to-many)...")
    emp_skills = generate_employee_skills(scale_emp, num_skills)
    batched_insert(conn, "INSERT OR IGNORE INTO employee_skills (emp_id, skill_id, level) VALUES (?, ?, ?);", emp_skills)

    print("[*] Assigning employees to projects (~2 assignments per employee)...")
    target_assignments = scale_emp * 2
    emp_projects = generate_emp_projects(scale_emp, num_projects, target_assignments)
    batched_insert(conn, "INSERT OR IGNORE INTO emp_projects (emp_id, proj_id, assigned_at, role_on_project, allocation_percent) VALUES (?, ?, ?, ?, ?);", emp_projects)

    num_assets = max(2000, num_projects // 2)
    print(f"[*] Generating {num_assets} assets...")
    assets = generate_assets(num_assets, scale_emp)
    batched_insert(conn, "INSERT OR IGNORE INTO assets (name, type, owner_emp, purchase_date, value, metadata_json) VALUES (?, ?, ?, ?, ?, ?);", assets)

    print("[*] Generating asset history...")
    asset_hist = generate_asset_history(num_assets)
    batched_insert(conn, "INSERT OR IGNORE INTO asset_history (asset_id, change_date, note) VALUES (?, ?, ?);", asset_hist)

    print("[*] Generating addresses...")
    addresses = generate_addresses(scale_emp)
    batched_insert(conn, "INSERT OR IGNORE INTO addresses (emp_id, line1, city, state, postal_code, country, geo, is_primary) VALUES (?, ?, ?, ?, ?, ?, ?, ?);", addresses)

    print("[*] Generating events (~3 per employee average)...")
    events = generate_events(scale_emp, avg_per_emp=3)
    batched_insert(conn, "INSERT OR IGNORE INTO events (emp_id, event_time, event_type, details_json, created_at) VALUES (?, ?, ?, ?, ?);", events)

    print("[*] Generating salaries history...")
    salaries = generate_salaries_history(scale_emp)
    batched_insert(conn, "INSERT OR IGNORE INTO salaries (emp_id, effective_date, salary, reason, created_at) VALUES (?, ?, ?, ?, ?);", salaries)

    conn.commit()
    print("[+] Data population complete.")

def main():
    Path(os.path.dirname(OUT_PATH)).mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(OUT_PATH, timeout=120)
    pragmas(conn)
    create_schema(conn)
    fill_db(conn, SCALE_EMPLOYEES)
    conn.close()
    print(f"[+] Database created at: {OUT_PATH}")

if __name__ == "__main__":
    main()


