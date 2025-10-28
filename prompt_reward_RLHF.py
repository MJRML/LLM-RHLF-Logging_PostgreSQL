import psycopg2
from sentence_transformers import SentenceTransformer, util
import torch

# Load embedding model (for semantic similarity)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Connect to PostgreSQL
conn = psycopg2.connect(
    host='localhost',
    database='llm_prompts_response',
    user='',
    password=''
)
cur = conn.cursor()

# Make sure reward_score column exists
cur.execute("""
ALTER TABLE llm_logs
ADD COLUMN IF NOT EXISTS reward_score FLOAT;
""")
conn.commit()

# Fetch all prompts and responses
cur.execute("SELECT id, prompt, response FROM llm_logs")
rows = cur.fetchall()

for row in rows:
    record_id, prompt, response = row

    # Here we create a simple "reference" by combining the prompt with a factual template
    # In production, you could use trusted sources or retrieval-augmented references
    reference = f"{prompt} [Expected correct answer]"

    # Compute semantic similarity as reward score (0 to 1)
    sim_score = util.cos_sim(
        embedder.encode(reference, convert_to_tensor=True),
        embedder.encode(response, convert_to_tensor=True)
    )[0][0].item()

    # Update the table with the reward score
    cur.execute(
        "UPDATE llm_logs SET reward_score = %s WHERE id = %s",
        (sim_score, record_id)
    )

# Commit changes and close connection
conn.commit()
cur.close()
conn.close()

print(f"Updated {len(rows)} rows with reward scores.")
