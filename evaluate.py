import psycopg2
from documents import create_in_memory_csv
from questions import get_gpt_snippets_by_strategy
from vars import conn_params,get_strategy_by_name,make_strategy

from concurrent.futures import ThreadPoolExecutor


def get_question_answer_pairs(conn, strategy_id):
    """
    Fetch the question text and the snippet ID for a given question ID.

    :param conn: psycopg2 connection object to the database
    :param question_id: The ID of the question
    :return: A tuple containing the question text and snippet ID
    """
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT questions.contents, questions.snippet_id
            FROM questions
            WHERE questions.strategy_id = %s;
        """, (strategy_id,))

        result = cursor.fetchall()
    return result

#overwrite this
def retrive(conn,question):
	return hack;

if __name__=="__main__":

	with psycopg2.connect(**conn_params) as conn:  
		strategy_id=get_strategy_by_name(conn,"testing with 3 gpt3.5_v3")['strategy_id']
		data=get_question_answer_pairs(conn,strategy_id)
		#print(data)
		#hack=[x[1] for x in data]
		#hack=list(range(100))
		hack=[data[0][1]]
		with ThreadPoolExecutor() as ex:
			ans=sum(ex.map(lambda x:x[1] in retrive(conn,x[0]),data))

		print(f"total corect: {ans}\naccuracy: {ans/len(data)}")

	