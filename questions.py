from openai import OpenAI
import pandas as pd 
import psycopg2
from documents import create_in_memory_csv
from vars import conn_params,get_strategy_by_name,make_strategy

import random
from concurrent.futures import ThreadPoolExecutor


def get_gpt_snippets_by_strategy(conn, strategy_id):
    """
    Fetch all GPT snippets associated with a specific strategy ID.

    :param conn: psycopg2 connection object to the database
    :param strategy_id: The strategy ID to filter GPT snippets
    :return: List of dictionaries, each representing a GPT snippet
    """
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT * FROM GPT_Snippets 
            WHERE strategy_id = %s;
        """, (strategy_id,))

        # Fetch column names from the cursor
        column_names = [desc[0] for desc in cursor.description]

        # Fetch rows and convert each to a dictionary
        snippets = [dict(zip(column_names, row)) for row in cursor.fetchall()]
        return snippets



def get_question(client, text):
    
    completion = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": "You will be provided with a text. Your task is to formulate specific and insightful questions that delve into the details and information within this text. Including names and specific details is encouraged, as long as they are not serial numbers or unique identifiers specific to the case. Avoid direct references to the document such as 'according to the document' or 'based on the document'."},
            {"role": "user", "content": text},
            {"role": "system", "content": "Focus on creating questions that demonstrate a deep understanding of the text's content. Include names and specific details where relevant, but avoid mentioning the document directly or using unique identifiers like case or serial numbers."},
        ]
    )
    return completion.choices[0].message.content 


def bulk_move(csv_buffer,conn):
    with conn.cursor() as cursor:
        cursor.copy_expert("COPY questions (strategy_id,snippet_id, contents) FROM STDIN WITH CSV", csv_buffer)

def make_question_entry(client,snippet):
	try:
		snippet["question"]=get_question(client,snippet["snippet_text"])
	except Exception as e:
		print(e)
		snippet["question"]=None

	return snippet


def make_questions_for(conn,read_strategy_id,write_strategy_id,client,cup=None):
	snippets=get_gpt_snippets_by_strategy(conn,read_strategy_id)
	if cup:
		snippets=random.sample(snippets,cup)
	
	with ThreadPoolExecutor() as ex:
		snippets=ex.map(lambda s: make_question_entry(client,s),snippets)
	snippets=[s for s in snippets if s['question']]
	
	csv_buffer=create_in_memory_csv([(write_strategy_id,f["snippet_id"],f["question"]) for f in snippets])
	bulk_move(csv_buffer,conn)

if __name__=="__main__":

	client = OpenAI()

	with psycopg2.connect(**conn_params) as conn:  
		read_id=get_strategy_by_name(conn,"deafualt choped 1_000 10_000")['strategy_id']
		write_id=make_strategy(conn,"testing with 3 v2")
		make_questions_for(conn,read_id,write_id,client,3)

	
	