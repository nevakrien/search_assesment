import psycopg2
#from documents import create_in_memory_csv
from vars import conn_params,get_strategy_by_name,make_strategy
from concurrent.futures import ThreadPoolExecutor
from questions import get_gpt_snippets_by_strategy

from deepsparse.sentence_transformers import DeepSparseSentenceTransformer
from tqdm import tqdm


def make_embeddings_table(conn, table_name, vector_size):
    """
    Create a dynamic table for embeddings with a specified vector size, if it does not already exist.

    :param conn: psycopg2 connection object to the database
    :param table_name: Name of the table to be created
    :param vector_size: Size of the vector for the 'embedding' column
    """
    with conn.cursor() as cursor:
        try:
            # Try creating the table
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
        except psycopg2.errors.DuplicateTable:
            # Handle duplicate table error
            print(f"Table '{table_name}' already exists.")
            conn.rollback()

def chunk_iterator(snippets,chunk_size):
    ans=[]
    for s in snippets:
        ans.append((s['snippet_id'],s['snippet_text']))
        if(len(ans)==chunk_size):
            yield ans
            ans=[]
    if(ans):
        yield ans


def update_embeddings(conn,table_name,l):
    #print([len(x) for x in l[0][-2:]])
    with conn.cursor() as cursor:
        cursor.executemany(f"""INSERT INTO {table_name} 
                (snippet_id,strategy_id,tokens,embedding)
                VALUES (%s, %s, %s, %s)""",
                l)

def make_naive_embedding(conn,read_id,write_id,table_name,model,chunk_size=32):
    
    make_embeddings_table(conn,table_name,model.config.hidden_size)
    snippets=get_gpt_snippets_by_strategy(conn,read_id)
    
    
    for c in chunk_iterator(tqdm(snippets),chunk_size):
        out=model.encode([x[1] for x in c])
        update_embeddings(conn,table_name,[(x[0],write_id,x[1],o) for x,o in zip(c,out)])



# Example usage
if __name__ == "__main__":

    model_name="neuralmagic/bge-large-en-v1.5-quant"


    model = DeepSparseSentenceTransformer(model_name, export=False)
    #model.to('cuda')
    #embedding_table_name=f"{model_name.replace('/','_').replace('-','_').replace('.','_')}_avrage_pool"
    strat_name="naive"
    #strat_name="test_based"
    table_extra="squad_ContextFromQuestion_"#"wiki_"

    #BREAKING CHANGE
    embedding_table_name=f"{table_extra}{model_name.replace('/','_').replace('-','_').replace('.','_')}_deep_sparse"


    #print(model(**tokenizer("שלום",return_tensors="pt")).last_hidden_state.shape)
    with psycopg2.connect(**conn_params) as conn:
        #read_id=get_strategy_by_name(conn,"deafualt choped 1_000 10_000")['strategy_id']
        #read_id=get_strategy_by_name(conn,"10wikipedia choped  100_000")['strategy_id']
        #read_id=get_strategy_by_name(conn,"hebrew squad (question->context)")['strategy_id']
        read_id=get_strategy_by_name(conn,"ensglish squad (question->context)")['strategy_id']
        #print(read_id)
        

        write_id=get_strategy_by_name(conn,f"{model_name}:{strat_name}")#['strategy_id']
        
        if(write_id==None):
            write_id=make_strategy(conn,f"{model_name}:{strat_name}")
        else:
            write_id=write_id['strategy_id']
        
        make_naive_embedding(conn,read_id,write_id,embedding_table_name,model,chunk_size=1)#,chunk_size=32)