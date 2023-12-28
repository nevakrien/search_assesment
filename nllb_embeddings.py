import psycopg2
#from documents import create_in_memory_csv
from vars import conn_params,get_strategy_by_name,make_strategy
from concurrent.futures import ThreadPoolExecutor
from questions import get_gpt_snippets_by_strategy

from transformers import NllbTokenizerFast,AutoModel
import torch

from tqdm import tqdm

def make_embeddings_table(conn, table_name, vector_size):
    """
    Create a dynamic table for embeddings with a specified vector size, if it does not already exist.

    :param conn: psycopg2 connection object to the database
    :param table_name: Name of the table to be created
    :param vector_size: Size of the vector for the 'embedding' column
    """
    with conn.cursor() as cursor:
        # Check if table already exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM pg_tables
                WHERE schemaname = 'public' AND tablename  = %s
            );
        """, (table_name,))
        table_exists = cursor.fetchone()[0]

        if not table_exists:
            # Create the table if it doesn't exist
            create_table_sql = f"""
                CREATE TABLE {table_name} (
                    embedding_id SERIAL PRIMARY KEY,
                    date_created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    embedding vector({vector_size}),
                    text TEXT,
                    tokens INT[],
                    snippet_id INT NOT NULL,
                    strategy_id INT NOT NULL,
                    FOREIGN KEY (snippet_id) REFERENCES GPT_Snippets (snippet_id),
                    FOREIGN KEY (strategy_id) REFERENCES Strategies (strategy_id)
                );
            """
            cursor.execute(create_table_sql)
            print(f"Table '{table_name}' created successfully.")
        else:
            print(f"Table '{table_name}' already exists.")

def chunk_iterator(snippets,tokenizer):
    for s in snippets:
        for out in s['snippet_text'].split('\n\n'):
            yield (s['snippet_id'],out)
           

@torch.no_grad
def run_mean(text,model,tokenizer):
    encoded_text = tokenizer(text, return_tensors="pt")
    encoded_text['input_ids'][:,1]=tokenizer.lang_code_to_id[tokenizer.src_lang]
    encoded_text={k:v.to(model.device) for k,v in encoded_text.items()}

    out=model(**encoded_text).last_hidden_state
    return out.mean(1).cpu()[0].tolist()


def update_embeddings(conn,table_name,l):
    #print([len(x) for x in l[0][-2:]])
    with conn.cursor() as cursor:
        cursor.execute(f"""INSERT INTO {table_name} 
                (snippet_id,strategy_id,text,embedding)
                VALUES (%s, %s, %s, %s)""",
                l)

def make_naive_embedding(conn,read_id,write_id,table_name,tokenizer,model):
    
    make_embeddings_table(conn,table_name,model.config.hidden_size)
    snippets=get_gpt_snippets_by_strategy(conn,read_id)
    
    
    for c in chunk_iterator(tqdm(snippets),tokenizer):
        out=run_mean(c[1],model,tokenizer)
        update_embeddings(conn,table_name,(c[0],write_id,c[1],out))

# Example usage
if __name__ == "__main__":
    #using avrage pooling because https://aclanthology.org/D19-1410.pdf
    
    model_name="facebook/nllb-200-3.3B"
    tokenizer=NllbTokenizerFast.from_pretrained(model_name,src_lang="heb_Hebr")
    model=AutoModel.from_pretrained(model_name)
    model=model.encoder
    model.to('cuda')
    embedding_table_name=f"{model_name.replace('/','_').replace('-','_').replace('.','_')}_avrage_pool"
    strat_name="naive"

    print("WARNING!!!: this code precommits unlike every other code in this codebase.\n it is your responsibelty to clean after it if it crashes\n this is because the transctions are so long I dont want to risk losing them.")
    #print(model(**tokenizer("שלום",return_tensors="pt")).last_hidden_state.shape)
    with psycopg2.connect(**conn_params) as conn:
        read_id=get_strategy_by_name(conn,"deafualt choped 1_000 10_000")['strategy_id']
        
        write_id=get_strategy_by_name(conn,f"{model_name}:{strat_name}")#['strategy_id']
        if(write_id==None):
            write_id=make_strategy(conn,f"{model_name}:{strat_name}")
        else:
            raise NotImplementedError("you are trying to return to loading the embeddings this is not yet implemented")
            #write_id=write_id['strategy_id']
        
        try:
            make_naive_embedding(conn,read_id,write_id,embedding_table_name,tokenizer,model)
        except Exception as e:
            print('writing transction in anyway')
            conn.commit()
            raise e