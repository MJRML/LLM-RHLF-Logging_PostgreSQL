#  LLM-RLHF-Logging-DB

### Automating Reward Scoring and Data Logging for Reinforcement Learning from Human Feedback (RLHF)

---

##  Overview

**LLM-RLHF-Logging-DB** is a Python-based project that logs prompts and responses from large language models (LLMs) into a PostgreSQL database while calculating key metrics like token counts, generation times, and reward scores.

The goal of this project is to explore **automating the reward signal process used in RLHF**, potentially reducing the need for human annotators. By leveraging automated metrics such as **semantic similarity**, **factuality**, and **relevance**, this system evaluates LLM responses to generate structured datasets suitable for RLHF fine-tuning.

---

##  Why This Project

- Traditional RLHF pipelines depend heavily on **manual human feedback**.  
- This project tests whether **automated scoring methods** can produce high-quality reward signals.  
- The result is a **scalable, self-improving feedback loop** where LLMs can be evaluated and fine-tuned automatically.  
- Demonstrates hands-on skills with **transformer-based LLMs, RLHF data workflows, database design, and applied AI engineering**.  

---

##  Features

- Logs prompts, responses, and metrics (token count, generation time) to PostgreSQL  
- Computes **automated reward scores** using semantic similarity and NLP metrics  
- Supports **batch processing** of multiple prompts  
- Tracks model configuration and metadata for auditability  
- Generates **RLHF-ready datasets** for fine-tuning or evaluation  

---

##  Technology Stack

| Component | Purpose |
|------------|----------|
| **Python 3.10+** | Core programming language |
| **PyTorch** | Backend for transformer inference |
| **Hugging Face Transformers** | LLM and tokenization pipeline |
| **SentenceTransformers** | Semantic similarity & automated reward metrics |
| **PostgreSQL** | Structured storage for prompt-response pairs |
| **psycopg2** | Python PostgreSQL adapter |
| **Git & GitHub** | Version control and deployment |

---

## Project Architecture

The project consists of two main scripts that form a lightweight data pipeline for prompt logging and automated evaluation:

+----------------------------+

Script 1: llm_logger.py
- Sends prompts to LLM
- Generates responses
- Logs to PostgreSQL DB
- Captures metrics:
• token_count
• generation_time
+----------------------------+

        │
        ▼

+----------------------------------+

Script 2: reward_scorer.py
- Fetches logged responses
- Embeds prompt/response pairs
- Calculates semantic similarity
- Updates DB with reward_score
+----------------------------------+


**Database Table:** `llm_logs`  
Stores:
- `prompt` – The input question or instruction  
- `response` – The model-generated output  
- `model_name` – Name of the model used (e.g., `microsoft/Phi-3-mini-4k-instruct`)  
- `token_count` – Total tokens used (input + output)  
- `generation_time` – Time taken for generation (seconds)  
- `reward_score` – Semantic similarity score (automated RLHF signal)

---

##  Script Summaries

### **1️prompt_response_db.py**

This script handles **prompt generation, LLM inference, and database logging**.

**Key Steps:**
- Connects to a PostgreSQL database (`llm_prompts_response`).
- Loads a Hugging Face transformer model (`microsoft/Phi-3-mini-4k-instruct`).
- Iterates over a list of predefined prompts.
- Generates responses and computes:
  - Token counts using `AutoTokenizer`
  - Response generation times
- Inserts the results into the database for later evaluation.

**Core Libraries Used:**
- `psycopg2` for PostgreSQL connections  
- `transformers` for LLM inference  
- `time` for generation timing  

---

### **2️ prompt_reward_RLHF.py**

This script performs **automated evaluation** by assigning a **reward score** to each LLM response.

**Key Steps:**
- Loads a `SentenceTransformer` model (`all-MiniLM-L6-v2`) for text embeddings.
- Fetches all `(prompt, response)` pairs from the database.
- Calculates semantic similarity between each response and a reference prompt.
- Writes the resulting **reward_score** back into the database.

**Core Libraries Used:**
- `psycopg2` for database operations  
- `sentence-transformers` for embedding models  
- `torch` for tensor computation  

---

## System Workflow

1. **Prompt & Response Logging**
   - `llm_logger.py` sends prompts to an LLM and logs outputs in PostgreSQL.

2. **Automated Evaluation**
   - `reward_scorer.py` reads the database, embeds text pairs, and computes semantic similarity.

3. **RLHF Dataset Generation**
   - The final table provides structured RLHF-compatible data with automated reward signals.

---

## Architecture Diagram

┌──────────────────────────┐
│ Prompts (User Inputs) │
└────────────┬─────────────┘
│
▼
┌──────────────────────────┐
│ LLM (Phi-3-mini-4k) │
│ → Generates Responses │
└────────────┬─────────────┘
│
▼
┌──────────────────────────┐
│ PostgreSQL: llm_logs │
│ • prompt │
│ • response │
│ • token_count │
│ • generation_time │
└────────────┬─────────────┘
│
▼
┌──────────────────────────┐
│ reward_scorer.py │
│ → Embedding Similarity │
│ → reward_score │
└────────────┬─────────────┘
▼
┌──────────────────────────┐
│ RLHF-Ready Dataset │
└──────────────────────────┘


---

## Requirements

Ensure the following Python libraries are installed:

```bash
pip install psycopg2-binary transformers sentence-transformers torch


## PostgreSQL instance and table configuration

CREATE TABLE llm_logs (
    id SERIAL PRIMARY KEY,
    prompt TEXT,
    response TEXT,
    model_name TEXT,
    token_count INT,
    generation_time FLOAT,
    reward_score FLOAT
);


### Usage

- Run the logger:
- python llm_logger.py  

- Run the reward scorer:
- python reward_scorer.py  

- View results:
- SELECT * FROM llm_logs ORDER BY id DESC;


### Future Improvements

- Add additional reward metrics (factuality, coherence, style).
- Integrate multiple LLM models for comparative analysis.
- Build a dashboard to visualize performance trends over time.
- Extend to distributed processing for large-scale prompt evaluation.