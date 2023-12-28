# search_assesment
working on building a working assesment for information retrival using machine generated data
documentation is still lacking

# build 
python 3.10

'''bash
pip3 install -r requirments.txt 
'''

'''bash 
createdb -h localhost -U postgres hebrew_search_reaserch
'''

'''bash 
psql -h localhost -U postgres hebrew_search_reaserch < sql_src/setup.sql 
psql -h localhost -U postgres hebrew_search_reaserch < sql_src/question.sql 
'''

## squad

get data that falls into this schenma {'data':list_of_dicts_with(['id', 'title', 'context', 'question', 'answers'])}
got from https://huggingface.co/datasets/tdklab/Hebrew_Squad_v1/blob/main/validation.json

# deafualt documents
python documents.py 
python snipets.py
