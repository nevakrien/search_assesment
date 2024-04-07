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

# def tensor_iterator(snippets,tokenizer,chunk_size):
#     ans=[]
#     for s in snippets:
#         output=tokenizer(s['snippet_text'],padding=True, truncation=True, return_tensors='pt')
#         #print(output.keys())
#         ans.append((s['snippet_id'],output['input_ids'],output['attention_mask']))
#         if(len(ans)==chunk_size):
#             yield ans
#             ans=[]

#     if(ans):
#         yield ans

@torch.no_grad
def run_mean(tokens,mask,model):
    mask=torch.IntTensor(mask).to(model.device)
    tokens=torch.IntTensor(tokens).to(model.device)
    #print(tokens.shape)
    #print(mask.shape)

    out=model(tokens,mask).last_hidden_state
    
    mask=mask[:,:,None]
    out*=mask
    #print(out.sum(1).shape)
    return (out.sum(1)/mask.sum(1)).cpu().tolist()

@torch.no_grad
def run_mean_checked(tokens,mask,model):
    mask=torch.IntTensor(mask).to(model.device)
    tokens=torch.IntTensor(tokens).to(model.device)
    #print(tokens.shape)
    #print(mask.shape)

    out=model(input_ids=tokens,attention_mask=mask).last_hidden_state
    #print(out)
    
    mask=mask[:,:,None]
    out*=mask

    out=out.sum(1)
    out=senetize(out)
    #print(out.sum(1).shape)
    return (out/mask.sum(1)).cpu().tolist()

def senetize(tensor):
    # Replace +inf with the maximum float value
    tensor = torch.where(tensor == torch.inf, torch.full_like(tensor, torch.finfo(tensor.dtype).max), tensor)
    # Replace -inf with the minimum float value
    tensor = torch.where(tensor == -torch.inf, torch.full_like(tensor, -torch.finfo(tensor.dtype).max), tensor)

    tensor = torch.where(torch.isnan(tensor), torch.full_like(tensor, 0), tensor)
    return tensor

# @torch.no_grad
# def run_mean_tensors(tokens,mask,model):
#     print(tokens)
#     print(mask)
#     mask=mask.to(model.device)
#     tokens=tokens.to(model.device)
#     #print(tokens.shape)

#     out=model(tokens,mask).last_hidden_state
    
#     mask=mask[:,:,None]
#     out*=mask
#     #print(out.sum(1).shape)
#     return (out.sum(1)/mask.sum(1)).cpu().tolist()

@torch.no_grad
def run_pooler(tokens,mask,model):
    mask=torch.IntTensor(mask).to(model.device)
    tokens=torch.IntTensor(tokens).to(model.device)
    #print(tokens.shape)

    return model(tokens,mask).pooler_output.cpu().tolist()
    


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


def update_embeddings_relaxed(conn, table_name, l):
    try:
        with conn.cursor() as cursor:
            cursor.executemany(f"""INSERT INTO {table_name}
                    (snippet_id, strategy_id, tokens, embedding)
                    VALUES (%s, %s, %s, %s)""",
                    l)
            conn.commit()  # Commit changes only if all insertions are successful
    except psycopg2.Error as e:  # Catching PostgreSQL errors
        print("PostgreSQL error:", e)
        print("Input list that caused the error:", l)
        # Optionally, you could roll back the transaction if you want to undo any changes made before the error
        conn.rollback()

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

def make_avg_embedding(conn,read_id,write_id,table_name,tokenizer,model,chunk_size=500):
    
    make_embeddings_table(conn,table_name,model.config.hidden_size)
    max_size=model.config.max_position_embeddings
    #print(max_size)
    snippets=get_gpt_snippets_by_strategy(conn,read_id)
    
    
    for c in chunk_iterator(tqdm(snippets),tokenizer,max_size,chunk_size):
        mask=[[1]*len(x[1])+[0]*(max_size-len(x[1])) for x in c]
        tokens=[x[1]+[0]*(max_size-len(x[1])) for x in c]

        out=run_mean_checked(tokens,mask,model)
        update_embeddings(conn,table_name,[(x[0],write_id,x[1],o) for x,o in zip(c,out)])

def make_avg_embedding_relaxed(conn,read_id,write_id,table_name,tokenizer,model,chunk_size=500):
    
    make_embeddings_table(conn,table_name,model.config.hidden_size)
    max_size=model.config.max_position_embeddings
    #print(max_size)
    snippets=get_gpt_snippets_by_strategy(conn,read_id)
    
    
    for c in chunk_iterator(tqdm(snippets),tokenizer,max_size,chunk_size):
        mask=[[1]*len(x[1])+[0]*(max_size-len(x[1])) for x in c]
        tokens=[x[1]+[0]*(max_size-len(x[1])) for x in c]

        out=run_mean_checked(tokens,mask,model)
        update_embeddings_relaxed(conn,table_name,[(x[0],write_id,x[1],o) for x,o in zip(c,out)])

def make_pooler_embedding(conn,read_id,write_id,table_name,tokenizer,model,chunk_size=500):
    make_embeddings_table(conn,table_name,model.config.hidden_size)
    max_size=model.config.max_position_embeddings
    #print(max_size)
    snippets=get_gpt_snippets_by_strategy(conn,read_id)
    
    bar=tqdm(snippets)
    print('got to iterator tqdn should exist')
    for c in chunk_iterator(bar,tokenizer,max_size,chunk_size):
        mask=[[1]*len(x[1])+[0]*(max_size-len(x[1])) for x in c]
        tokens=[x[1]+[0]*(max_size-len(x[1])) for x in c]

        out=run_pooler(tokens,mask,model)
        update_embeddings(conn,table_name,[(x[0],write_id,x[1],o) for x,o in zip(c,out)])

# Example usage
if __name__ == "__main__":
    #using avrage pooling because https://aclanthology.org/D19-1410.pdf
    #model_name="bert-base-multilingual-cased"
    #model_name="avichr/Legal-heBERT"
    #model_name="avichr/heBERT"
    #model_name="bert-base-uncased"
    #model_name="models/bert-base-uncased_L2_v0"
    #model_name="sentence-transformers/all-MiniLM-L6-v2"#(sbert)
    #model_name="imvladikon/sentence-transformers-alephbert"
    
    #model_name="thenlper/gte-base"#"aws-neuron/bge-base-en-v1-5-seqlen-384-bs-1"
    #model_name="BAAI/bge-large-en-v1.5"
    #model_name="llmrails/ember-v1"

    #model_name="nomic-ai/nomic-embed-text-v1" #breaks the huggingface standard on argument order...
    #model_name="yam-peleg/Hebrew-Gemma-11B" #too slow gona need to run in a place with gpu (with my local machine db talking to it)
    #model_name="google/gemma-7b"
    
    model_name="my_model"
    model_path="/media/user/8a594cab-20d9-43ef-8d0e-b60b5cf43462/hebrew_search_stuff/results/checkpoint-2040000"
    tokenizer_path="avichr/heBERT"

    tokenizer=AutoTokenizer.from_pretrained(tokenizer_path)
    model=AutoModel.from_pretrained(model_path)

    #tokenizer=AutoTokenizer.from_pretrained(model_name)
    #model=AutoModel.from_pretrained(model_name,load_in_4bit=True,bnb_4bit_compute_dtype=torch.float16)
    print(model.config.max_position_embeddings)
    #model.to('cuda')
    #embedding_table_name=f"{model_name.replace('/','_').replace('-','_').replace('.','_')}_avrage_pool"
    strat_name="naive"
    #strat_name="test_based"
    table_extra="squad_ContextFromQuestion_v1_hebrew"#"squad_ContextFromQuestion_v2_hebrew"#"squad_ContextFromQuestion_v2_"#"squad_ContextFromQuestion_"#"wiki_"

    #BREAKING CHANGE
    embedding_table_name=f"{table_extra}{model_name.replace('/','_').replace('-','_').replace('.','_')}_avrage_pool"
    #embedding_table_name=f"{table_extra}{model_name.replace('/','_').replace('-','_').replace('.','_')}_pooler"


    #print(model(**tokenizer("שלום",return_tensors="pt")).last_hidden_state.shape)
    with psycopg2.connect(**conn_params) as conn:
        #read_id=get_strategy_by_name(conn,"deafualt choped 1_000 10_000")['strategy_id']
        #read_id=get_strategy_by_name(conn,"10wikipedia choped  100_000")['strategy_id']##
        read_id=get_strategy_by_name(conn,"hebrew squad (question->context)")['strategy_id']
        #read_id=get_strategy_by_name(conn,"ensglish squad (question->context)")['strategy_id']
        #read_id=get_strategy_by_name(conn,"ensglish squad (question->context) v2")['strategy_id']
        #read_id=get_strategy_by_name(conn,"hebrew squad (question->context) v2")['strategy_id']
        
        #print(read_id)
        

        write_id=get_strategy_by_name(conn,f"{model_name}:{strat_name}")#['strategy_id']
        
        if(write_id==None):
            write_id=make_strategy(conn,f"{model_name}:{strat_name}")
        else:
            write_id=write_id['strategy_id']
        
        #make_naive_embedding(conn,read_id,write_id,embedding_table_name,tokenizer,model,chunk_size=64)#,chunk_size=32)#,chunk_size=1)#,chunk_size=32)
        #make_pooler_embedding(conn,read_id,write_id,embedding_table_name,tokenizer,model)
        
        #make_avg_embedding(conn,read_id,write_id,embedding_table_name,tokenizer,model,chunk_size=2)
        make_avg_embedding_relaxed(conn,read_id,write_id,embedding_table_name,tokenizer,model,chunk_size=32)
