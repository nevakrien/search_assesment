from transformers import AutoModelForSeq2SeqLM, NllbTokenizerFast
import torch
import psycopg2

from vars import conn_params,get_strategy_by_name,make_strategy
from documents import create_in_memory_csv
#from concurrent.futures import ProcessPoolExecutor

@torch.no_grad
def translate_text(text,tokenizer,model):
    # Tokenize and translate the text
    encoded_text = tokenizer(text, return_tensors="pt")
    #manual fix to hf bug 
    encoded_text['input_ids'][:,1]=tokenizer.lang_code_to_id[tokenizer.src_lang]

    encoded_text={k:v.to(model.device) for k,v in encoded_text.items()}
    generated_tokens = model.generate(**encoded_text,forced_bos_token_id=tokenizer.lang_code_to_id[tokenizer.tgt_lang]).cpu()

    # Decode and return the translated text
    return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)



def get_questions_by_strategy(conn, strategy_id):
    """
    Fetch questions associated with a specific strategy ID.

    :param conn: psycopg2 connection object to the database
    :param strategy_id: The strategy ID to filter questions
    :return: List of questions
    """
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT question_id,contents FROM questions
            WHERE strategy_id = %s;
        """, (strategy_id,))

        # Fetch column names from the cursor
        column_names = [desc[0] for desc in cursor.description]

        # Fetch rows and convert each to a dictionary
        questions = [dict(zip(column_names, row)) for row in cursor.fetchall()]
        return questions

def get_translated_questions_by_parent_and_own_strategy(conn, original_strategy_id, translated_strategy_id):
    """
    Fetch translated questions where their corresponding original question has a specific strategy ID,
    and they themselves have a specific strategy ID.

    :param conn: psycopg2 connection object to the database
    :param original_strategy_id: The strategy ID of the original questions
    :param translated_strategy_id: The strategy ID of the translated questions
    :return: List of translated questions
    """
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT translated_questions.translated_question_id, translated_questions.contents
            FROM translated_questions
            INNER JOIN questions ON translated_questions.question_id = questions.question_id
            WHERE questions.strategy_id = %s AND translated_questions.strategy_id = %s;
        """, (original_strategy_id, translated_strategy_id))

        return cursor.fetchall()

def bulk_move(csv_buffer,conn):
    with conn.cursor() as cursor:
        cursor.copy_expert("COPY translated_questions (strategy_id,question_id, contents) FROM STDIN WITH CSV", csv_buffer)

def make_translation_entry(tokenizer,model,question):
    try:
        question["translation"]=translate_text(question["contents"],tokenizer,model)
    except Exception as e:
        print(e)
        question["translation"]=None

    return question

def make_translations_for(conn,read_id,write_id,tokenizer,model):
    questions=get_questions_by_strategy(conn,read_id)
    #with ProcessPoolExecutor() as ex:
    questions=map(lambda s: make_translation_entry(tokenizer,model,s),questions)

    questions=[s for s in questions if s['translation']]
    csv_buffer=create_in_memory_csv([(write_id,f["question_id"],f["translation"]) for f in questions])
    bulk_move(csv_buffer,conn)


# Example usage
if __name__ == "__main__":
    model_name="facebook/nllb-200-3.3B"
    # Initialize the tokenizer and model
    tokenizer = NllbTokenizerFast.from_pretrained(model_name,tgt_lang="heb_Hebr",src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to('cuda')

    # Connect to the database
    with psycopg2.connect(**conn_params) as conn:
        read_id=get_strategy_by_name(conn,"1000 gpt3.5")['strategy_id']
        #for realease build use:
        #read_id=get_strategy_by_name(conn,"testing with 3 gpt3.5")['strategy_id']
        
        write_id=get_strategy_by_name(conn,f"basic: {model_name}")
        if(write_id==None):
            write_id=make_strategy(conn,f"basic: {model_name}")
        else:
            write_id=write_id['strategy_id']
        make_translations_for(conn,read_id,write_id,tokenizer,model)
        print(get_translated_questions_by_parent_and_own_strategy(conn,read_id,write_id))

        #raise NotImplemented
