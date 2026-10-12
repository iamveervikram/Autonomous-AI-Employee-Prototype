import os
import sqlite3
from datetime import datetime, timezone
from typing import Dict, List, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "hr_system.db")


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(reset: bool = False) -> None:
    conn = get_db_connection()
    cursor = conn.cursor()

    if reset:
        cursor.execute("DROP TABLE IF EXISTS employees")
        cursor.execute("DROP TABLE IF EXISTS audit_logs")

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS employees (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            role TEXT NOT NULL,
            department TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            vault_status TEXT NOT NULL DEFAULT 'intact',
            updated_at TEXT NOT NULL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id TEXT NOT NULL,
            action TEXT NOT NULL,
            details TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
        """
    )

    cursor.execute("SELECT COUNT(*) as count FROM employees")
    count = cursor.fetchone()["count"]

    if count == 0:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        seed_employees = [
            (
                "EMP-001",
                "John Doe",
                "john.doe@centralign.ai",
                "Senior Backend Engineer",
                "Platform Engineering",
                "active",
                "intact",
                now,
            ),
            (
                "EMP-002",
                "Radha K Murthy",
                "radha.murthy@centralign.ai",
                "Staff Product Designer",
                "Design Systems",
                "active",
                "intact",
                now,
            ),
            (
                "EMP-003",
                "Jane Smith",
                "jane.smith@centralign.ai",
                "DevOps Specialist",
                "Cloud Infrastructure",
                "active",
                "intact",
                now,
            ),
            (
                "EMP-004",
                "Arjun Mehta",
                "arjun.mehta@centralign.ai",
                "Data Scientist",
                "AI Research",
                "inactive",
                "purged",
                now,
            ),
        ]
        cursor.executemany(
            """
            INSERT INTO employees (id, name, email, role, department, status, vault_status, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            seed_employees,
        )

        cursor.execute(
            """
            INSERT INTO audit_logs (employee_id, action, details, timestamp)
            VALUES (?, ?, ?, ?)
            """,
            ("SYSTEM", "DB_SEEDED", "Initialized HR database with 4 employees.", now),
        )

    conn.commit()
    conn.close()


def log_audit_event(employee_id: str, action: str, details: str) -> None:
    conn = get_db_connection()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    conn.execute(
        """
        INSERT INTO audit_logs (employee_id, action, details, timestamp)
        VALUES (?, ?, ?, ?)
        """,
        (employee_id, action, details, now),
    )
    conn.commit()
    conn.close()


def get_all_employees(search_query: Optional[str] = None) -> List[Dict]:
    conn = get_db_connection()
    cursor = conn.cursor()
    if search_query and search_query.strip():
        q = f"%{search_query.strip()}%"
        cursor.execute(
            """
            SELECT * FROM employees
            WHERE name LIKE ? OR id LIKE ? OR email LIKE ? OR department LIKE ?
            ORDER BY id ASC
            """,
            (q, q, q, q),
        )
    else:
        cursor.execute("SELECT * FROM employees ORDER BY id ASC")
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_employee_by_id(employee_id: str) -> Optional[Dict]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM employees WHERE UPPER(id) = UPPER(?)", (employee_id.strip(),))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def update_employee_status(employee_id: str, new_status: str) -> bool:
    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    cursor.execute(
        """
        UPDATE employees
        SET status = ?, updated_at = ?
        WHERE UPPER(id) = UPPER(?)
        """,
        (new_status, now, employee_id.strip()),
    )
    updated = cursor.rowcount > 0
    conn.commit()
    conn.close()
    if updated:
        log_audit_event(
            employee_id.upper(),
            f"STATUS_CHANGED_TO_{new_status.upper()}",
            f"Employee status updated to '{new_status}' via HR Portal.",
        )
    return updated


def update_vault_status(employee_id: str, new_vault_status: str) -> bool:
    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    cursor.execute(
        """
        UPDATE employees
        SET vault_status = ?, updated_at = ?
        WHERE UPPER(id) = UPPER(?)
        """,
        (new_vault_status, now, employee_id.strip()),
    )
    updated = cursor.rowcount > 0
    conn.commit()
    conn.close()
    if updated:
        log_audit_event(
            employee_id.upper(),
            f"VAULT_{new_vault_status.upper()}",
            f"Employee data vault status updated to '{new_vault_status}'.",
        )
    return updated


def get_recent_audit_logs(limit: int = 10) -> List[Dict]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM audit_logs ORDER BY id DESC LIMIT ?", (limit,))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Initialize or reset the HR SQLite database.")
    parser.add_argument("--reset", action="store_true", help="Drop existing tables and re-seed data.")
    args = parser.parse_args()

    init_db(reset=args.reset)
    print(f"Database initialized at: {DB_PATH}")
    for emp in get_all_employees():
        print(f" - {emp['id']}: {emp['name']} ({emp['role']}) | status={emp['status']} | vault={emp['vault_status']}")
