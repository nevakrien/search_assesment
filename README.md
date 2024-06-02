# search_assesment

working on building a working assesment for information retrival using machine generated data
documentation is still lacking

the core file of this project is evaluate.py where the results of the expirements are saved

this is mainly my working sketch of things and the usage is generally based in the command line itself. important to note that this project is no longer in active devlopment and we are unlikely to see changes

the statistical anylisis in visual.py should be easy enough to use for most people so the results should be meaningful regardless of your math backround	

# build 
python 3.10

'''bash
pip3 install -r requirments.txt 
'''

'''bash 
pg_ctl start
createdb -h localhost -U postgres hebrew_search_reaserch
'''

'''bash 
psql -h localhost -U postgres hebrew_search_reaserch < sql_src/setup.sql 
psql -h localhost -U postgres hebrew_search_reaserch < sql_src/question.sql 
'''

## squad

get data that falls into this schenma {'data':list_of_dicts_with(['id', 'title', 'context', 'question', 'answers'])}
got from https://huggingface.co/datasets/tdklab/Hebrew_Squad_v1/blob/main/validation.json

# usage guidelines 
in general the main function of every script is where input varibles should be inserted 
this is to be done manualy for now gona work on fixing it in the future. 

every* main function uses a single transction for the entire operation so closing it prematurly would leave the database in a usble state


## strategies
strategies are an identifier used for every data generation action regardless of table. thier main use is in runing specific expirements on a subset of the data based on generation

strategies should be retrived by name and not passed by Id since Ids are not garnteed to be the same by postgres.

manual intervension with the database is alowed and should not cause major errors. 

renaming stratgies is premited and should be used whenever there was an error in the original excution.


except for translation strategy, the entire database operation can be described by a single strategy Id there is no need to track parents down.
(this may change in the future)

## donts

dont alow 2 strategies to share a name

dont add aproximate indcies to an existing table since that would break old expirements silently. 

dont run this in production 
