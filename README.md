# search_assesment
working on building a working assesment for information retrival using machine generated data

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

# deafualt documents
python documents.py 
python snipets.py
