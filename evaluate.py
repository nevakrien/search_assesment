import psycopg2
from documents import create_in_memory_csv
from questions import get_gpt_snippets_by_strategy
from vars import conn_params,get_strategy_by_name,make_strategy

from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer,AutoModel
import torch

from tqdm import tqdm

def get_original_question_answer_pairs(conn, strategy_id):
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

def get_question_answer_pairs(conn, strategy_id,translation_strategy_id):
	if(translation_strategy_id==None):
		return get_original_question_answer_pairs(conn,strategy_id)
	return get_translated_question_answer_pairs(conn, strategy_id,translation_strategy_id)

def evaluate_retriver(conn,func,data,max_workers=None):
	with ThreadPoolExecutor(max_workers=max_workers) as ex:
			ans=sum(tqdm(ex.map(lambda x:x[1] in func(conn,x[0]),data),total=len(data)))
	return ans


#overwrite this
def retrive(conn,question):
	return hack;

def get_random_retriver(model_name,embedding_table_name,k=1):
	tokenizer=AutoTokenizer.from_pretrained(model_name)
	model=AutoModel.from_pretrained(model_name)
	model.to('cuda')
	#embedding_table_name=f"{model_name.replace('/','_').replace('-','_')}_avrage_pool"

	@torch.no_grad
	def ans(conn,text):
		emb=model(tokenizer.encode(text,return_tensors='pt').to(model.device))
		emb=emb.last_hidden_state.mean(1).cpu().tolist()[0]
		with conn.cursor() as cursor:
			cursor.execute(f"""SELECT snippet_id
								FROM {embedding_table_name}
								WHERE embedding IS NOT NULL
								LIMIT %s;""",
								(k,)
								)
			return [x[0] for x in cursor.fetchall()]
	return ans

def get_naive_retriver(model_name,embedding_table_name,k=1):
	tokenizer=AutoTokenizer.from_pretrained(model_name)
	model=AutoModel.from_pretrained(model_name)
	model.to('cuda')
	#embedding_table_name=f"{model_name.replace('/','_').replace('-','_')}_avrage_pool"

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

def get_naive_retriver_parts(embedding_table_name,tokenizer_path,model_path,k=1):
	tokenizer=AutoTokenizer.from_pretrained(tokenizer_path)
	model=AutoModel.from_pretrained(model_path)
	model.to('cuda')
	#embedding_table_name=f"{model_name.replace('/','_').replace('-','_')}_avrage_pool"

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


#this is needed because nomic is buggy
@torch.no_grad
def run_mean_nomic(tokens,mask,model):
    mask=torch.IntTensor(mask).to(model.device)
    tokens=torch.IntTensor(tokens).to(model.device)
    #print(tokens.shape)
    #print(mask.shape)

    out=model(input_ids=tokens,attention_mask=mask).last_hidden_state
    #print(out)
    
    mask=mask[:,:,None]
    out*=mask
    #print(out.sum(1).shape)
    #print(out.shape)
    return (out.sum(1)/mask.sum(1)).cpu().tolist()

#this is the buggist code I ever dealt with it seems like the phase of the moon has more effect whether or not it should run than anything I do
#DONT TOUCH THIS PLEASE I BEG YOU.
#literly identical code broke over details like where the print statment should exist (and I am not printing tensors...)
def get_nomic_retriver(model_name,embedding_table_name,k=1):
	tokenizer=AutoTokenizer.from_pretrained(model_name)
	model=AutoModel.from_pretrained(model_name)
	model.to('cuda')
	#embedding_table_name=f"{model_name.replace('/','_').replace('-','_')}_avrage_pool"

	@torch.no_grad
	def ans(conn,text):
		#inputs={k:torch.IntTensor([v]).to(model.device) for k,v in tokenizer(text).items()}
		#inputs={k:torch.IntTensor([v+[0]*(model.config.max_position_embeddings-len(v))]).to(model.device) for k,v in tokenizer([text,text]).items()}#, return_tensors='pt').items()}
		#inputs={k:[v+[0]*(model.config.max_position_embeddings-len(v))] for k,v in tokenizer(text).items()}#, return_tensors='pt').items()}
		#inputs={k:2*v for k,v in tokenizer(text).items()}
		# print({k:v.shape for k,v in emb.items()})
		#inputs.pop('token_type_ids') 
		assert(type(text)==str)
		inputs=tokenizer([text])
		try:
			emb=run_mean_nomic(inputs['input_ids'],inputs['attention_mask'],model)
			#emb=model(**inputs)
			#emb=model(input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask'])
		except Exception as e:
			print({k:v for k,v in inputs.items()})
			raise e
		#print(type(emb))
		#assert 1==2/4
		emb=emb[0]
		#emb=emb[0].last_hidden_state.mean(1).cpu().tolist()[0]
		#print('ok')
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

def get_quant_retriver(model_name,embedding_table_name,k=1):
	tokenizer=AutoTokenizer.from_pretrained(model_name)
	model=AutoModel.from_pretrained(model_name,load_in_4bit=True,bnb_4bit_compute_dtype=torch.float16)
	#embedding_table_name=f"{model_name.replace('/','_').replace('-','_')}_avrage_pool"

	@torch.no_grad
	def ans(conn,text):
		#inputs={k:torch.IntTensor([v]).to(model.device) for k,v in tokenizer(text).items()}
		#inputs={k:torch.IntTensor([v+[0]*(model.config.max_position_embeddings-len(v))]).to(model.device) for k,v in tokenizer([text,text]).items()}#, return_tensors='pt').items()}
		#inputs={k:[v+[0]*(model.config.max_position_embeddings-len(v))] for k,v in tokenizer(text).items()}#, return_tensors='pt').items()}
		#inputs={k:2*v for k,v in tokenizer(text).items()}
		# print({k:v.shape for k,v in emb.items()})
		#inputs.pop('token_type_ids') 
		assert(type(text)==str)
		inputs=tokenizer([text])
		try:
			emb=run_mean_nomic(inputs['input_ids'],inputs['attention_mask'],model)
			#emb=model(**inputs)
			#emb=model(input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask'])
		except Exception as e:
			print({k:v for k,v in inputs.items()})
			raise e
		#print(type(emb))
		#assert 1==2/4
		emb=emb[0]
		#emb=emb[0].last_hidden_state.mean(1).cpu().tolist()[0]
		#print('ok')
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

def get_pooler_retriver(model_name,embedding_table_name,k=1):
	assert embedding_table_name[-7:]=="_pooler"

	tokenizer=AutoTokenizer.from_pretrained(model_name)
	model=AutoModel.from_pretrained(model_name)
	model.to('cuda')
	#embedding_table_name=f"{model_name.replace('/','_').replace('-','_')}_avrage_pool"

	@torch.no_grad
	def ans(conn,text):
		emb=model(tokenizer.encode(text,return_tensors='pt').to(model.device))
		emb=emb.pooler_output.cpu().tolist()[0]
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

def get_L2_pooler_retriver(model_name,embedding_table_name,k=1):
	assert embedding_table_name[-7:]=="_pooler"

	tokenizer=AutoTokenizer.from_pretrained(model_name)
	model=AutoModel.from_pretrained(model_name)
	model.to('cuda')
	#embedding_table_name=f"{model_name.replace('/','_').replace('-','_')}_avrage_pool"

	@torch.no_grad
	def ans(conn,text):
		emb=model(tokenizer.encode(text,return_tensors='pt').to(model.device))
		emb=emb.pooler_output.cpu().tolist()[0]
		with conn.cursor() as cursor:
			cursor.execute(f"""SELECT snippet_id
								FROM {embedding_table_name}
								WHERE embedding IS NOT NULL
								ORDER BY embedding <-> %s
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
		#strats=["10 wikipedia gpt4"]#["1000 gpt3.5"]
		strats=["hebrew squad (question->context)"]
		#strats=["ensglish squad (question->context)"]
		#strats=["ensglish squad (question->context) v2"]
		#strats=["hebrew squad (question->context) v2"]

		#trans_strats=["basic: facebook/nllb-200-3.3B"]
		strategy_ids=[get_strategy_by_name(conn,s)['strategy_id'] for s in strats]
		#translation_strategy_ids=[get_strategy_by_name(conn,s)['strategy_id'] for s in trans_strats]
		
		translation_strategy_ids=[None]
		data=[get_question_answer_pairs(conn,x1,x2) for x1,x2 in zip(strategy_ids,translation_strategy_ids)]
		data=sum(data,[])
		#print(data)
		#hack=[x[1] for x in data]
		#hack=list(range(100))
		#hack=[data[0][1]]

		#model_name="imvladikon/sentence-transformers-alephbert" #hard total corect: 588 accuracy: 0.07887323943661972

		#model_name="sentence-transformers/all-MiniLM-L6-v2"# total corect: 1297 accuracy: 0.12270577105014191 easy: total corect: 8563 accuracy: 0.8101229895931883

		
		#model_name="llmrails/ember-v1"#total corect: 1600 accuracy: 0.15137180700094607 easy total corect: 8314 accuracy: 0.7865657521286661

		#model_name="thenlper/gte-base" #total corect: 1572 accuracy: 0.14872280037842953 easy: total corect: 8583 accuracy: 0.8120151371807001
		#model_name="BAAI/bge-large-en-v1.5"#total corect: 1608 accuracy: 0.1521286660359508 easy: total corect: 8016 accuracy: 0.7583727530747398

		#model_name="models/bert-base-uncased_L2_v0"
		#model_name="models/bert-base-uncased_v1"
		#model_name="bert-base-uncased"#hard total corect: 681 accuracy: 0.06442762535477767

		#model_name="avichr/heBERT"#hard total corect: 416 accuracy: 0.05580147551978538

		#model_name="avichr/Legal-heBERT"
		#model_name="bert-base-multilingual-cased"

		#model_name="nomic-ai/nomic-embed-text-v1" #total corect: 1624 accuracy: 0.15364238410596026

		#model_name="google/gemma-7b"#hard #total corect: 24 accuracy: 0.002270577105014191 #hebrew_hard total corect: 18accuracy: 0.002414486921529175

		model_name="my_model" #total corect: 3 accuracy: 0.00040241448692152917 #EASY total corect: 15 accuracy: 0.002012072434607646


		model_path="/media/user/8a594cab-20d9-43ef-8d0e-b60b5cf43462/hebrew_search_stuff/results/checkpoint-2040000"
		tokenizer_path="avichr/heBERT"




		table_extra="squad_ContextFromQuestion_v1_hebrew"#"squad_ContextFromQuestion_v2_hebrew"#"squad_ContextFromQuestion_v2_"#"squad_ContextFromQuestion_"#"wiki_"
		embedding_table_name=f"{table_extra}{model_name.replace('/','_').replace('-','_').replace('.','_')}_avrage_pool"
		#embedding_table_name=f"{table_extra}{model_name.replace('/','_').replace('-','_').replace('.','_')}_pooler"

		print(f"evaluating {model_name}")
		print(targets_not_in_embeddings(conn,data,embedding_table_name))
		

		#retrive=get_naive_retriver(model_name,embedding_table_name,1)#100)#1)#327285 #100_000
		#retrive=get_nomic_retriver(model_name,embedding_table_name,1)#100)#1)#327285 #100_000
		#retrive=get_pooler_retriver(model_name,embedding_table_name,1)
		#retrive=get_L2_pooler_retriver(model_name,embedding_table_name,1)
		#retrive=get_random_retriver(model_name,embedding_table_name,3)#327285 #100_000
		#retrive=get_quant_retriver(model_name,embedding_table_name,1)
		retrive=get_naive_retriver_parts(embedding_table_name,tokenizer_path,model_path)
		
		ans=evaluate_retriver(conn,retrive,data,2)
		
		# with ThreadPoolExecutor() as ex:
		# 	ans=sum(ex.map(lambda x:x[1] in retrive(conn,x[0]),data))

		print(f"total corect: {ans}\naccuracy: {ans/len(data)}")

	