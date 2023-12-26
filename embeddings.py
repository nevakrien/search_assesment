import psycopg2
#from documents import create_in_memory_csv
from vars import conn_params,get_strategy_by_name,make_strategy
from concurrent.futures import ThreadPoolExecutor
from questions import get_gpt_snippets_by_strategy

from transformers import AutoTokenizer,AutoModel
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

def chunk_iterator(snippets,tokenizer,max_size,chunk_size):
    ans=[]
    for s in snippets:
        tokens=tokenizer.encode(s['snippet_text'])
        for i in range(0,len(tokens),max_size):
            ans.append((s['snippet_id'],tokens[i:i+max_size]))
            if(len(ans)==chunk_size):
                yield ans
                ans=[]
    if(ans):
        yield ans

@torch.no_grad
def run_mean(tokens,mask,model):
    mask=torch.IntTensor(mask).to(model.device)
    tokens=torch.IntTensor(tokens).to(model.device)
    #print(tokens.shape)

    out=model(tokens,mask).last_hidden_state
    
    mask=mask[:,:,None]
    out*=mask
    #print(out.sum(1).shape)
    return (out.sum(1)/mask.sum(1)).cpu().tolist()


# def update_embeddings(conn,table_name,l):
#     #print([len(x) for x in l[0][-2:]])
#     with conn.cursor() as cursor:
#         with ThreadPoolExecutor() as ex:
#             #ex.
#             list(map( lambda x: cursor.execute(f"""INSERT INTO {table_name} 
#                 (snippet_id,strategy_id,tokens,embedding)
#                 VALUES (%s, %s, %s, %s)""",
#                 x),l))

def update_embeddings(conn,table_name,l):
    #print([len(x) for x in l[0][-2:]])
    with conn.cursor() as cursor:
        cursor.executemany(f"""INSERT INTO {table_name} 
                (snippet_id,strategy_id,tokens,embedding)
                VALUES (%s, %s, %s, %s)""",
                l)

def make_naive_embedding(conn,read_id,write_id,table_name,tokenizer,model,chunk_size=500):
    
    make_embeddings_table(conn,table_name,model.config.hidden_size)
    max_size=model.config.max_position_embeddings
    #print(max_size)
    snippets=get_gpt_snippets_by_strategy(conn,read_id)
    
    
    for c in chunk_iterator(tqdm(snippets),tokenizer,max_size,chunk_size):
        mask=[[1]*len(x[1])+[0]*(max_size-len(x[1])) for x in c]
        tokens=[x[1]+[0]*(max_size-len(x[1])) for x in c]

        out=run_mean(tokens,mask,model)
        update_embeddings(conn,table_name,[(x[0],write_id,x[1],o) for x,o in zip(c,out)])

# Example usage
if __name__ == "__main__":
    #using avrage pooling because https://aclanthology.org/D19-1410.pdf
    
    model_name="facebook/nllb-200-3.3B"
    #model_name="bert-base-multilingual-cased"
    #model_name="avichr/Legal-heBERT"
    #model_name="avichr/heBERT"
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    model=AutoModel.from_pretrained(model_name)
    model.to('cuda')
    embedding_table_name=f"{model_name.replace('/','_').replace('-','_').replace('.','_')}_avrage_pool"
    strat_name="naive"


    #print(model(**tokenizer("שלום",return_tensors="pt")).last_hidden_state.shape)
    with psycopg2.connect(**conn_params) as conn:
        read_id=get_strategy_by_name(conn,"deafualt choped 1_000 10_000")['strategy_id']
        
        write_id=get_strategy_by_name(conn,f"{model_name}:{strat_name}")#['strategy_id']
        if(write_id==None):
            write_id=make_strategy(conn,f"{model_name}:{strat_name}")
        else:
            write_id=write_id['strategy_id']
        
        make_naive_embedding(conn,read_id,write_id,embedding_table_name,tokenizer,model,chunk_size=32)