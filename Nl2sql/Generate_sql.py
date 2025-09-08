from __future__ import annotations
import re, sqlite3, os
import pandas as pd
from typing import Tuple, Optional

# We assume you created a global OpenAI client in config.py
# config.py contains:
#   from openai import OpenAI
#   client = OpenAI(api_key=OPENAI_API_KEY)
from Config import client

ALLOWED_TABLE = "transactions"
ALLOWED_COLUMNS = {"id","ts","amount","ccy","counterparty","book"}

SCHEMA_DESC = (
    "SQLite database with a single table 'transactions' (read-only).\n"
    "Columns:\n"
    "- id (INTEGER primary key)\n"
    "- ts (TIMESTAMP as TEXT 'YYYY-MM-DD HH:MM:SS')\n"
    "- amount (FLOAT)\n"
    "- ccy (TEXT currency code: 'USD','EUR','GBP',...)\n"
    "- counterparty (TEXT)\n"
    "- book (TEXT)\n"
)

SYSTEM_PROMPT = f"""You are an assistant that translates user requests into **SQL SELECT** queries for SQLite.

Rules:
- Use ONLY the table `{ALLOWED_TABLE}` with columns {sorted(list(ALLOWED_COLUMNS))}.
- Output **only** a valid SQL SELECT statement. No explanations, no backticks.
- Do NOT modify data. Disallow any DDL/DML (DROP, DELETE, UPDATE, INSERT, ALTER, CREATE, ATTACH, PRAGMA).
- Prefer explicit filters (e.g., ccy='USD', counterparty='Acme Corp').
- If the query could return many rows, include a LIMIT (e.g., LIMIT 100).
- Timestamps are in column `ts` as 'YYYY-MM-DD HH:MM:SS'. You may use LIKE 'YYYY-MM-DD%' to filter by date.

Examples:
User: How many EUR trades with Acme Corp?
SQL: SELECT COUNT(*) FROM transactions WHERE ccy = 'EUR' AND counterparty = 'Acme Corp';

User: Show total amount per counterparty in USD.
SQL: SELECT counterparty, SUM(amount) AS total_amount FROM transactions WHERE ccy = 'USD' GROUP BY counterparty;

User: List the 5 largest USD trades with Beta Bank.
SQL: SELECT * FROM transactions WHERE ccy = 'USD' AND counterparty = 'Beta Bank' ORDER BY amount DESC LIMIT 5;
"""

# ---------- LLM call ----------

def llm_sql(user_query: str, model: str = "gpt-4o-mini", max_tokens: int = 200) -> str:
    """Ask the LLM for a SQL SELECT statement. Returns raw SQL string."""
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_query.strip()},
    ]
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        max_tokens=max_tokens,
        messages=msgs,
    )
    sql = resp.choices[0].message.content.strip()
    # Remove code fences if the model added them
    sql = re.sub(r"^```(?:sql)?\s*|\s*```$", "", sql, flags=re.IGNORECASE|re.DOTALL).strip()
    # Normalize quotes (avoid “ ”)
    sql = sql.replace("’","'").replace("`","")
    return sql

# ---------- Guardrails ----------

DDL_DML_PATTERN = re.compile(
    r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|CREATE|TRUNCATE|ATTACH|PRAGMA|VACUUM)\b",
    re.IGNORECASE
)

IDENT_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

AGG_FUNCS = {"COUNT","SUM","AVG","MIN","MAX"}

def is_select(sql: str) -> bool:
    return bool(re.match(r"^\s*SELECT\b", sql, flags=re.IGNORECASE))

# --- replace mentions_only_allowed_identifiers() with these helpers ---

TABLE_NAME = ALLOWED_TABLE.lower()
ALLOWED_COLS_LC = {c.lower() for c in ALLOWED_COLUMNS}

TABLE_FROM_PATTERN = re.compile(r"\bFROM\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)
TABLE_JOIN_PATTERN = re.compile(r"\bJOIN\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)
DOT_COL_PATTERN   = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)")

def only_allowed_tables(sql: str) -> bool:
    """Ensure FROM/JOIN tables are exactly 'transactions'."""
    for m in TABLE_FROM_PATTERN.finditer(sql):
        if m.group(1).lower() != TABLE_NAME:
            return False
    for m in TABLE_JOIN_PATTERN.finditer(sql):
        if m.group(1).lower() != TABLE_NAME:
            return False
    # Require at least one FROM transactions
    return bool(TABLE_FROM_PATTERN.search(sql))

def dot_columns_are_allowed(sql: str) -> bool:
    """If dot notation is used, ensure table is 'transactions' and column is allowed."""
    for m in DOT_COL_PATTERN.finditer(sql):
        t, c = m.group(1).lower(), m.group(2).lower()
        if t != TABLE_NAME or c not in ALLOWED_COLS_LC:
            return False
    return True



def has_limit(sql: str) -> bool:
    return bool(re.search(r"\bLIMIT\s+\d+\b", sql, flags=re.IGNORECASE))

def is_aggregate(sql: str) -> bool:
    return any(func in sql.upper() for func in AGG_FUNCS)

def add_default_limit(sql: str, default_limit: int = 100) -> str:
    if not has_limit(sql) and not is_aggregate(sql):
        return sql.rstrip(";") + f" LIMIT {default_limit};"
    return sql

def sanitize_sql(sql: str) -> Tuple[bool, str]:
    sql_clean = sql.strip().rstrip(";") + ";"

    if not is_select(sql_clean):
        return False, "Only SELECT queries are allowed."
    if DDL_DML_PATTERN.search(sql_clean):
        return False, "Destructive or unsafe SQL detected (DDL/DML not allowed)."
    if not only_allowed_tables(sql_clean):
        return False, "Query references unknown or disallowed tables."
    if not dot_columns_are_allowed(sql_clean):
        return False, "Query references unknown columns via dot notation."

    sql_final = add_default_limit(sql_clean)
    return True, sql_final


# ---------- Execution ----------

def execute_sql(db_path: str, sql: str, limit_rows: int = 100) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(sql, con)
        # Safety cap in case LIMIT missing (extra guard)
        if len(df) > limit_rows:
            df = df.head(limit_rows)
        return df
    finally:
        con.close()

# ---------- Main entrypoint for the app ----------

def nl2sql_run(user_query: str,
               db_path: str = os.path.join("data","transactions.db"),
               model: str = "gpt-4o-mini") -> Tuple[Optional[str], Optional[pd.DataFrame], Optional[str]]:
    """
    Returns (generated_sql, dataframe, error_message).
    If error_message is not None, generated_sql/dataframe may be None.
    """
    if not os.path.exists(db_path):
        return None, None, f"Database not found at {db_path}. Run data/make_transactions_db.py first."

    try:
        raw_sql = llm_sql(user_query, model=model)
    except Exception as e:
        return None, None, f"LLM error: {e}"

    ok, safe_or_msg = sanitize_sql(raw_sql)
    if not ok:
        return raw_sql, None, f"Blocked query: {safe_or_msg}"

    try:
        df = execute_sql(db_path, safe_or_msg)
        return safe_or_msg, df, None
    except Exception as e:
        return safe_or_msg, None, f"SQL execution error: {e}"
