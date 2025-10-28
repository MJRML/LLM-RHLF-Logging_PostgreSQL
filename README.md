# LLM-RLHF-Logging-DB

### Automating Reward Scoring and Data Logging for Reinforcement Learning from Human Feedback (RLHF)

---

## Overview

**LLM-RLHF-Logging-DB** is a Python-based project that logs prompts and responses from large language models (LLMs) into a PostgreSQL database while calculating key metrics like token counts, generation times, and reward scores.

The goal of this project is to explore **automating the reward signal process used in RLHF**, potentially reducing the need for human annotators. By leveraging automated metrics such as **semantic similarity**, **factuality**, and **relevance**, this system evaluates LLM responses to generate structured datasets suitable for RLHF fine-tuning.

---

## Features

- Logs prompts, responses, and metrics (token count, generation time) to PostgreSQL  
- Computes **automated reward scores** using semantic similarity and NLP metrics  
- Supports **batch processing** of multiple prompts  
- Tracks model configuration and metadata for auditability  
- Generates **RLHF-ready datasets** for fine-tuning or evaluation  

---

## Project Architecture

### llm_logger.py

- Sends prompts to LLM  
- Generates responses  
- Logs data to PostgreSQL  
- Captures metrics: token_count, generation_time  

### reward_scorer.py

- Fetches logged responses  
- Embeds prompt/response pairs  
- Calculates semantic similarity  
- Updates reward_score in DB  

### Database Table: llm_logs

CREATE TABLE llm_logs (
    id SERIAL PRIMARY KEY,
    prompt TEXT,
    response TEXT,
    model_name TEXT,
    token_count INT,
    generation_time FLOAT,
    reward_score FLOAT
);

---

## Columns Description

- prompt – The input question or instruction  
- response – The model-generated output  
- model_name – Name of the model used (e.g., microsoft/Phi-3-mini-4k-instruct)  
- token_count – Total tokens used (input + output)  
- generation_time – Time taken for generation (seconds)  
- reward_score – Semantic similarity score (automated RLHF signal)  

---

## Requirements

- Python 3.10+  
- PostgreSQL instance  
- Python libraries:

pip install psycopg2-binary transformers sentence-transformers torch

---

## Usage

- Run the Logger:

python prompt_response_db.py.py

- Run the Reward Scorer:

python prompt_reward_RLHF.py

- View Results in PostgreSQL:

SELECT * FROM llm_logs ORDER BY id DESC;

---

## Future Improvements

- Add additional reward metrics (factuality, coherence, style)  
- Integrate multiple LLM models for comparative analysis  
- Build a dashboard to visualize performance trends over time  
- Extend to distributed processing for large-scale prompt evaluation
