import sqlite3
import datetime
import tiktoken

def create_table():
    conn = sqlite3.connect("token_usage.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS training_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            query TEXT,
            teacher_embedding BLOB,
            student_embedding BLOB,
            loss REAL
        )
    """)
    conn.commit()
    conn.close()

create_table()

def log_training_data(query, teacher_emb, student_emb, loss):
    conn = sqlite3.connect("token_usage.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO training_logs 
        (timestamp, query, teacher_embedding, student_embedding, loss)
        VALUES (?, ?, ?, ?, ?)
    """, (
        datetime.datetime.now().isoformat(),
        query,
        str(teacher_emb),
        str(student_emb),
        loss
    ))
    conn.commit()
    conn.close()
