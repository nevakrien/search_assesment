import psycopg2
from documents import create_in_memory_csv
from vars import conn_params,get_strategy_by_name,make_strategy
from concurrent.futures import ThreadPoolExecutor
from questions import get_gpt_snippets_by_strategy

from transformers import AutoTokenizer,AutoModel

def create_embeddings_table(conn, table_name, vector_size):
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


# Example usage
if __name__ == "__main__":
    model_name="avichr/heBERT"
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    model=AutoModel.from_pretrained(model_name)
    embedding_table_name=f"{model_name} 'avrage' pool"

    print(model.config.max_position_embeddings)
    print(model(**tokenizer("שלום",return_tensors="pt")).last_hidden_state.shape)
    with psycopg2.connect(**conn_params) as conn:
        read_id=get_strategy_by_name(conn,"deafualt choped 1_000 10_000")['strategy_id']
        snippets=get_gpt_snippets_by_strategy(conn,read_id)
        make_strategy(conn,embedding_table_name+"cu")
        raise NotImplemented