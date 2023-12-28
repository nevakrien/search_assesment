import psycopg2
from vars import conn_params,make_strategy
from tqdm import tqdm
import re 
from documents import create_in_memory_csv

import os
from os.path import join

def document_iter(dir_name):
    
    for file in tqdm(tuple(os.listdir(dir_name))):
        if file[-4:]!='.txt':
            continue

        with open(join(dir_name,file)) as f:
            text=f.read()

        if not text:
            continue
        text=text.strip()

        yield (file,re.sub(r'\n\s*\n\s*\n+', '\n\n', text))
        

def bulk_move(csv_buffer,conn):
    with conn.cursor() as cursor:
        cursor.copy_expert("COPY files (strategy_id,file_name, contents) FROM STDIN WITH CSV", csv_buffer)

def make_docs(conn,write_id,folder_name):
    csv_buffer = create_in_memory_csv((strat_id,)+x for x in document_iter(folder_name))
    bulk_move(csv_buffer,conn)

if __name__ == "__main__":
    
    # Connect to the database
    with psycopg2.connect(**conn_params) as conn:
        print('making strategy')
        strat_id=make_strategy(conn,"10wikipedia","hand copied wikipedia pages")
        make_docs(conn,strat_id,"wikipedia_texts")
        
    print('done')
