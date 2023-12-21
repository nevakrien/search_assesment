import psycopg2
from vars import conn_params,make_strategy
import datasets
from tqdm import tqdm
import re 

import io

def document_iter(text_list):
    bads = (
            re.compile(r'<!--\s*\n\s*/\*\s*Font Definitions\s*\*/'),
            re.compile(r'\uFFFD'),
            re.compile(r'endstream.*?(?=\n|$)'),
            re.compile(r'/Author \(user\)\n|/Creator'),
            re.compile(r'\ue51d'),
            re.compile(r'[\u0080-\u00FF\u0100-\u017F]'), #weird latin dialects like ã
           )
            

    
    for i,text in enumerate(text_list):
        if not text:
            continue
        text=text.strip()
        
        if text in ("File not found",""):
            continue
            
        skip=False
        for b in bads:
            if b.search(text):
                skip=True
                break
        if skip:
            continue

        yield (i,re.sub(r'\n\s*\n\s*\n+', '\n\n', text))
        

def create_in_memory_csv(documents):
    # Create an in-memory bytes buffer
    csv_buffer = io.BytesIO()

    # Write data to the buffer
    for doc in documents:
        snippet = ','.join(['"' + str(field).replace('"', '""') + '"' for field in doc]) + '\n'
        csv_buffer.write(snippet.encode('utf-8'))

    # Reset buffer position to the beginning
    csv_buffer.seek(0)
    return csv_buffer

def bulk_move(csv_buffer,conn):
    with conn.cursor() as cursor:
        cursor.copy_expert("COPY files (strategy_id,file_name, contents) FROM STDIN WITH CSV", csv_buffer)



if __name__ == "__main__":
    print('loading dataset')
    data = datasets.load_dataset('LevMuchnik/SupremeCourtOfIsrael')
    texts = data['train']['text']

    
    # Connect to the database
    with psycopg2.connect(**conn_params) as conn:
        print('making strategy')
        strat_id=make_strategy(conn,"deafualt supreme court israel","a cleaned version of the hf database")
        print('making in memory csv')
        csv_buffer = create_in_memory_csv((strat_id,)+x for x in document_iter(tqdm(texts)))
        print('moving from memory to disk this may take a while...')
        bulk_move(csv_buffer,conn)
    print('done')
