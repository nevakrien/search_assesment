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



def get_question(client, text,model):
    
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You will be provided with a text. Your task is to formulate a single question that is specific, detailed, and insightful, based on the information in the text. The question should center around the names of the people involved and the actions or events described. Avoid phrases like 'in the document', 'according to the document', or any unique identifiers like serial or case numbers. For example, do not ask 'What did the document say about Rachel Hali Miron's economic rights?' Instead, ask 'What specific allegations did Rachel Hali Miron make regarding the violation of her economic rights?'"},
            {"role": "user", "content": text},
            {"role": "system", "content": "Craft a single question that is clear and standalone, focusing on the actions, decisions, and roles of the individuals named in the text. The question should be understandable and answerable without referring back to the text."},
        ]
    )
    return completion.choices[0].message.content 


def bulk_move(csv_buffer,conn):
    with conn.cursor() as cursor:
        cursor.copy_expert("COPY questions (strategy_id,snippet_id, contents) FROM STDIN WITH CSV", csv_buffer)

def make_question_entry(client,snippet,model):
	try:
		snippet["question"]=get_question(client,snippet["snippet_text"],model)
	except Exception as e:
		print(e)
		snippet["question"]=None

	return snippet


def make_questions_for(conn,read_strategy_id,write_strategy_id,client,cup=None,model="gpt-3.5-turbo-1106"):
	snippets=get_gpt_snippets_by_strategy(conn,read_strategy_id)
	if cup:
		snippets=random.sample(snippets,cup)
	
	with ThreadPoolExecutor() as ex:
		snippets=ex.map(lambda s: make_question_entry(client,s,model),snippets)
	snippets=[s for s in snippets if s['question']]
	
	csv_buffer=create_in_memory_csv([(write_strategy_id,f["snippet_id"],f["question"]) for f in snippets])
	try:
		bulk_move(csv_buffer,conn)
	except Exception as e:
		with open('crash_question_data.csv','w') as f:
			f.write(csv_buffer.getvalue())
		raise e

if __name__=="__main__":

	client = OpenAI()

	with psycopg2.connect(**conn_params) as conn:  
		#read_id=get_strategy_by_name(conn,"deafualt choped 1_000 10_000")['strategy_id']
		read_id=get_strategy_by_name(conn,"10wikipedia choped  100_000")['strategy_id']
		
		#write_name="1000 gpt3.5"
		write_name="10 wikipedia gpt4"
		write_id=get_strategy_by_name(conn,write_name)
		if(write_id==None):
		    write_id=make_strategy(conn,write_name)
		else:
		    write_id=write_id['strategy_id']
		
		make_questions_for(conn,read_id,write_id,client,None,"gpt-4-1106-preview")#,1000)

	
	