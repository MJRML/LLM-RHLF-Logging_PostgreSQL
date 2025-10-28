import psycopg2
import os
from transformers import pipeline, AutoTokenizer
import time

#connect to postgreSQL

conn = psycopg2.connect(
    host='localhost',
    database='llm_prompts_response',
    user='postgres',
    password='P@ssW0rd!'
)

cur = conn.cursor()

# Enable progress bars and logs
os.environ["HF_HUB_ENABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "info"

#Load the model
print('Loading model.......')
generator = pipeline(
    'text-generation',
    model='microsoft/Phi-3-mini-4k-instruct',
    device_map='auto',
    dtype='auto',
    return_full_text=False
)

tokenizer=AutoTokenizer.from_pretrained('microsoft/Phi-3-mini-4k-instruct')


prompts = ['What is cryptocurrency',
      'What is scientology',
      'When was the United Arab Emirates founded']
#Prompt
# prompts=[
#     'Is Irish real-estate a good investment now',
#     # 'What is cryptocurrency and what is the best cryptocurrency to invest in',
#     # 'Will there be a recession in 2026',
#     'How old is Elton John',
#     # 'Who is he heavtweight boxing world champion',
#     # 'What are the top 5 priced stocks in 2025',
#     # 'What is scientology',
#     # 'Who is Michael Collins, in Irish history',
#     # 'When was the United Arab Emirates founded',
#     # 'Will AI take my role as a marketing assistant'
# ]

#Gnerate a response -- loop through the prompts

for prompt in prompts:
    start_time = time.time()
    response = generator(prompt, max_new_tokens=200, do_sample=False)[0]['generated_text']
    end_time = time.time()
    generation_time = end_time - start_time

    #Count tokens
    tokens = tokenizer.encode(prompt + response)
    token_count = len(tokens)


    #Insert into database
    cur.execute(
        'INSERT INTO llm_logs (prompt, response, model_name, token_count, generation_time) VALUES (%s, %s, %s, %s, %s)',
        (prompt, response, 'microsoft/Phi-3-mini-4k-instruct', token_count, generation_time)
    )

conn.commit()

print(f"Saved! Tokens: {token_count}, Time: {generation_time:.2f} sec")

#Close connection
cur.close()
conn.close()