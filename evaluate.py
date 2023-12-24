import psycopg2
from documents import create_in_memory_csv
from questions import get_gpt_snippets_by_strategy
from vars import conn_params,get_strategy_by_name,make_strategy

from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer,AutoModel
import torch

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

def get_translated_question_answer_pairs(conn, strategy_id,translation_strategy_id):
    """
    Fetch the translated question text and the original snippet ID for a given translated question strategy ID.

    :param conn: psycopg2 connection object to the database
    :param strategy_id: The strategy ID of the translated questions
    :return: A list of tuples, each containing the translated question text and the original snippet ID
    """
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT translated_questions.contents, questions.snippet_id
            FROM translated_questions
            INNER JOIN questions ON translated_questions.question_id = questions.question_id
            WHERE questions.strategy_id = %s AND translated_questions.strategy_id =%s;
        """, (strategy_id,translation_strategy_id))

        result = cursor.fetchall()
    return result

def evaluate_retriver(conn,func,data):
	with ThreadPoolExecutor() as ex:
			ans=sum(ex.map(lambda x:x[1] in func(conn,x[0]),data))
	return ans

#overwrite this
def retrive(conn,question):
	return hack;

def get_naive_retriver(model_name,k=1):
	tokenizer=AutoTokenizer.from_pretrained(model_name)
	model=AutoModel.from_pretrained(model_name)
	model.to('cuda')
	embedding_table_name=f"{model_name.replace('/','_').replace('-','_')}_avrage_pool"

	@torch.no_grad
	def ans(conn,text):
		emb=model(tokenizer.encode(text,return_tensors='pt').to(model.device))
		emb=emb.last_hidden_state.mean(1).cpu().tolist()[0]
		with conn.cursor() as cursor:
			cursor.execute(f"""SELECT snippet_id
								FROM {embedding_table_name}
								WHERE embedding IS NOT NULL
								ORDER BY embedding <=> %s
								LIMIT %s;""",
								(str(emb),k)
								)
			return [x[0] for x in cursor.fetchall()]
	return ans

def targets_not_in_embeddings(conn,data,embedding_table_name):
	with conn.cursor() as cursor:
			cursor.execute(f"""SELECT snippet_id
								FROM {embedding_table_name}"""
								)
			ans=[x[0] for x in cursor.fetchall()]
	#print(len(ans))
	return [x for x in data if x[1] not in ans]


if __name__=="__main__":

	with psycopg2.connect(**conn_params) as conn:  
		strats=["testing with 3 gpt3.5_v3"]
		trans_strats=["basic: facebook/nllb-200-3.3B"]
		strategy_ids=[get_strategy_by_name(conn,s)['strategy_id'] for s in strats]
		translation_strategy_ids=[get_strategy_by_name(conn,s)['strategy_id'] for s in trans_strats]
		data=[get_translated_question_answer_pairs(conn,x1,x2) for x1,x2 in zip(strategy_ids,translation_strategy_ids)]
		data=sum(data,[])
		#print(data)
		#hack=[x[1] for x in data]
		#hack=list(range(100))
		#hack=[data[0][1]]

		model_name="avichr/heBERT"
		#model_name="avichr/Legal-heBERT"
		embedding_table_name=f"{model_name.replace('/','_').replace('-','_')}_avrage_pool"
		print(targets_not_in_embeddings(conn,data,embedding_table_name))
		retrive=get_naive_retriver(model_name,100)#327285
		
		ans=evaluate_retriver(conn,retrive,data)
		# with ThreadPoolExecutor() as ex:
		# 	ans=sum(ex.map(lambda x:x[1] in retrive(conn,x[0]),data))

		print(f"total corect: {ans}\naccuracy: {ans/len(data)}")

	