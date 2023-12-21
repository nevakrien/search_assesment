import psycopg2
from vars import conn_params,make_strategy,get_strategy_by_name
from documents import create_in_memory_csv
import tiktoken
from tqdm import tqdm

def get_files_by_strategy_and_min_length(conn, strategy_id, min_length):
    """
    Fetch files associated with a specific strategy ID where the length of their contents is greater than min_length.

    :param conn: psycopg2 connection object to the database
    :param strategy_id: ID of the strategy to filter files
    :param min_length: Minimum length of file contents
    :return: List of files meeting the criteria
    """
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT file_id,contents FROM Files 
            WHERE strategy_id = %s 
            AND LENGTH(contents) >= %s;
        """, (strategy_id, min_length))

        files = cursor.fetchall()
        return files

def bulk_move(csv_buffer,conn):
    with conn.cursor() as cursor:
        cursor.copy_expert("COPY gpt_Snippets (strategy_id,file_id, snippet_text) FROM STDIN WITH CSV", csv_buffer)


def make_snipets_in_range(conn,read_strategy_id,write_strategy_id,min_tokens=1_000,max_tokens=10_000):
	enc = tiktoken.encoding_for_model("gpt-4")
	files=get_files_by_strategy_and_min_length(conn,read_strategy_id,min_tokens)
	files=[f for f in tqdm(files) if min_tokens<=len(enc.encode(f[1]))<=max_tokens]
	
	csv_buffer=create_in_memory_csv([(write_strategy_id,f[0],f[1]) for f in files])
	bulk_move(csv_buffer,conn)

if __name__=="__main__":
	
	with psycopg2.connect(**conn_params) as conn:
		read_id=get_strategy_by_name(conn,"deafualt supreme court israel")['strategy_id']
		write_id=make_strategy(conn,"deafualt choped 1_000 10_000")
		make_snipets_in_range(conn,read_id,write_id)

	print('done')