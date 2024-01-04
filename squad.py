import psycopg2
from vars import conn_params,get_strategy_by_name,make_strategy
from documents import create_in_memory_csv

import json
from os.path import join

from tqdm import tqdm

# def get_val():
#     with open(join('data','tdklab___hebrew_squad_v1','validation.json')) as f:
#     	return json.load(f)['data']

def get_val():
    from datasets import load_dataset
    return load_dataset("squad",split='validation')



def add_data(conn, data_list, strategy_id):
    with conn.cursor() as cursor:
        for data_item in tqdm(data_list):
            # Create a file record
            file_insert_query = "INSERT INTO files (file_name, contents, strategy_id) VALUES (%s, %s, %s) RETURNING file_id"
            cursor.execute(file_insert_query, (data_item['id'], data_item['context'], strategy_id))
            file_id = cursor.fetchone()[0]  # Fetch the file_id

            # Create snippet text
            a=data_item['answers']['text'][0]
            assert type(a)==str
            snippet_text = f"{data_item['title']} {data_item['context']} {data_item['question']} {a}"

            # Create a snippet record
            snippet_insert_query = "INSERT INTO gpt_snippets (snippet_text, num_tokens, file_id, strategy_id) VALUES (%s, %s, %s, %s) RETURNING snippet_id"
            cursor.execute(snippet_insert_query, (snippet_text, len(snippet_text.split()), file_id, strategy_id))
            snippet_id = cursor.fetchone()[0]  # Fetch the snippet_id

            # Create a question record
            question_insert_query = "INSERT INTO questions (contents, snippet_id, strategy_id) VALUES (%s, %s, %s)"
            cursor.execute(question_insert_query, (data_item['question'], snippet_id, strategy_id))

def add_hard_data(conn, data_list, strategy_id):
    with conn.cursor() as cursor:
        for data_item in tqdm(data_list):
            # Create a file record
            file_insert_query = "INSERT INTO files (file_name, contents, strategy_id) VALUES (%s, %s, %s) RETURNING file_id"
            cursor.execute(file_insert_query, (data_item['id'], data_item['context'], strategy_id))
            file_id = cursor.fetchone()[0]  # Fetch the file_id

            snippet_text = f"{data_item['title']}\n{data_item['context']}"

            # Create a snippet record
            snippet_insert_query = "INSERT INTO gpt_snippets (snippet_text, num_tokens, file_id, strategy_id) VALUES (%s, %s, %s, %s) RETURNING snippet_id"
            cursor.execute(snippet_insert_query, (snippet_text, len(snippet_text.split()), file_id, strategy_id))
            snippet_id = cursor.fetchone()[0]  # Fetch the snippet_id

            # Create a question record
            question_insert_query = "INSERT INTO questions (contents, snippet_id, strategy_id) VALUES (%s, %s, %s)"
            cursor.execute(question_insert_query, (data_item['question'], snippet_id, strategy_id))

if __name__=="__main__":
    data=get_val()
    #print(data[0].keys())
    with psycopg2.connect(**conn_params) as conn:  
        #write_id=make_strategy(conn,"hebrew squad (question->context) v2")
        write_id=make_strategy(conn,"ensglish squad (question->context) v2")
        #write_id=make_strategy(conn,"ensglish squad (question->context)")
        #add_data(conn,data,write_id)
        add_hard_data(conn,data,write_id)
		