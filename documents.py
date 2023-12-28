import psycopg2
from vars import conn_params,make_strategy
import datasets
from tqdm import tqdm
import re 

import io

def document_iter(text_list):
    bads = (
            re.compile(r'<!--\s*\n\s*/\*\s*Font Definitions\s*\*/'),
            re.compile(r'\uFFFD'),
            re.compile(r'endstream.*?(?=\n|$)'),
            re.compile(r'/Author \(user\)\n|/Creator'),
            re.compile(r'\ue51d'),
            re.compile(r'[\u0080-\u00FF\u0100-\u017F]'), #weird latin dialects like ã
           )
            

    
    for i,text in enumerate(text_list):
        if not text:
            continue
        text=text.strip()
        
        if text in ("File not found",""):
            continue
            
        skip=False
        for b in bads:
            if b.search(text):
                skip=True
                break
        if skip:
            continue

        yield (i,re.sub(r'\n\s*\n\s*\n+', '\n\n', text))
        

def create_in_memory_csv(documents):
    # Create an in-memory bytes buffer
    csv_buffer = io.BytesIO()

    # Write data to the buffer
    for doc in documents:
        snippet = ','.join(['"' + str(field).replace('"', '""') + '"' for field in doc]) + '\n'
        csv_buffer.write(snippet.encode('utf-8'))

    # Reset buffer position to the beginning
    csv_buffer.seek(0)
    return csv_buffer

def bulk_move(csv_buffer,conn):
    with conn.cursor() as cursor:
        cursor.copy_expert("COPY files (strategy_id,file_name, contents) FROM STDIN WITH CSV", csv_buffer)



if __name__ == "__main__":
    print('loading dataset')#{"plain_text": {"description": "Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable. This Hebrew dataset is an automatic translation of the English SQuAD dataset.", "homepage": "https://github.com/TechnionTDK/hebwiki-qa/", "license": "", "features": {"id": {"dtype": "string", "id": null, "_type": "Value"}, "title": {"dtype": "string", "id": null, "_type": "Value"}, "context": {"dtype": "string", "id": null, "_type": "Value"}, "question": {"dtype": "string", "id": null, "_type": "Value"}, "answers": {"feature": {"text": {"dtype": "string", "id": null, "_type": "Value"}, "answer_start": {"dtype": "int32", "id": null, "_type": "Value"}}, "length": -1, "id": null, "_type": "Sequence"}}, "post_processed": null, "supervised_keys": null, "task_templates": [{"task": "question-answering-extractive", "question_column": "question", "context_column": "context", "answers_column": "answers"}], "builder_name": "squad", "config_name": "plain_text", "version": {"version_str": "1.0.0", "description": "", "major": 1, "minor": 0, "patch": 0}, "splits": {"train": {"name": "train", "num_bytes": 62387110, "num_examples": 52405, "dataset_name": "Hebrew_Squad_v1"}, "validation": {"name": "validation", "num_bytes": 9482653, "num_examples": 7455, "dataset_name": "Hebrew_Squad_v1"}}, "download_checksums": {"https://github.com/TechnionTDK/hebwiki-qa/tree/main/dataset_creation/data_files/HUGGING_FACE_FORMAT_TRANSLATED_DIRECTORY/train.json": {"num_bytes": 30288272, "checksum": "3527663986b8295af4f7fcdff1ba1ff3f72d07d61a20f487cb238a6ef92fd955"}, "https://github.com/TechnionTDK/hebwiki-qa/tree/main/dataset_creation/data_files/HUGGING_FACE_FORMAT_TRANSLATED_DIRECTORY/validation.json": {"num_bytes": 4854279, "checksum": "95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9"}}, "download_size": 35142551, "post_processing_size": null, "dataset_size": 89789763, "size_in_bytes": 124932314}}
    data = datasets.load_dataset('LevMuchnik/SupremeCourtOfIsrael')
    texts = data['train']['text']

    
    # Connect to the database
    with psycopg2.connect(**conn_params) as conn:
        print('making strategy')
        strat_id=make_strategy(conn,"deafualt Hebrew_Squad_v1","a cleaned version of the hf database")
        print('making in memory csv')
        csv_buffer = create_in_memory_csv((strat_id,)+x for x in document_iter(tqdm(texts)))
        print('moving from memory to disk this may take a while...')
        bulk_move(csv_buffer,conn)
    print('done')
