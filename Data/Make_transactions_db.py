#Run once python data/make_transactions_db.py

import os, sqlite3
import random, datetime

# Where the database will live
DB_PATH = os.path.join("data", "transactions.db")
os.makedirs("data", exist_ok=True)

counterparties = ["Acme Corp", "Beta Bank", "Gamma LLC", "Delta Ltd", "Omega Partners"]
currencies = ["USD", "EUR", "GBP", "JPY"]
books = ["Loans", "FX Desk", "Derivatives", "Equities"]

rows = []
id_counter = 1
base_date = datetime.datetime(2025, 9, 1, 9, 0, 0)

for day in range(10):  # 10 days of trades
    for _ in range(5):  # 5 trades per day
        ts = base_date + datetime.timedelta(days=day, hours=random.randint(0,8), minutes=random.randint(0,59))
        amount = random.randint(50_000, 10_000_000)
        ccy = random.choice(currencies)
        cp = random.choice(counterparties)
        book = random.choice(books)
        rows.append((id_counter, ts.strftime("%Y-%m-%d %H:%M:%S"), amount, ccy, cp, book))
        id_counter += 1


schema_sql = """
CREATE TABLE IF NOT EXISTS transactions (
  id INTEGER PRIMARY KEY,
  ts TEXT,
  amount REAL,
  ccy TEXT,
  counterparty TEXT,
  book TEXT
);
"""

def main():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.executescript(schema_sql)
    cur.execute("DELETE FROM transactions;")  # reset for idempotent runs
    cur.executemany(
        "INSERT INTO transactions (id, ts, amount, ccy, counterparty, book) VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    con.commit()
    con.close()
    print(f"Seeded {DB_PATH} with {len(rows)} rows.")

if __name__ == "__main__":
    main()
