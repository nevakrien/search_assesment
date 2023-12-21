conn_params = {
    'dbname': 'hebrew_search_reaserch',
    'user': 'postgres',
    'password': 'postgres',
    'host': 'localhost'
}
def make_strategy(conn,source_name,description=None):
    with conn.cursor() as cursor:
        cursor.execute( """
                INSERT INTO Strategies (source_name, description, date_created)
                VALUES (%s, %s, CURRENT_TIMESTAMP) RETURNING strategy_id;
            """,
            [source_name, description])

        return cursor.fetchone()[0]
        
def get_strategy_by_name(conn, strategy_name):
    """
    Fetch a strategy by its name.

    :param conn: psycopg2 connection object to the database
    :param strategy_name: Name of the strategy to fetch
    :return: Dictionary containing strategy data
    :raises: Exception if more than one strategy is found with the same name
    """
    with conn.cursor() as cursor:
        cursor.execute("SELECT * FROM Strategies WHERE source_name = %s;", (strategy_name,))
        rows = cursor.fetchall()
        
        if len(rows) > 1:
            raise Exception("Name clash detected. There are multiple strategies with the name '{}'. Please rename one of these strategies.".format(strategy_name))
        elif len(rows) == 1:
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, rows[0]))
        else:
            return None

if __name__=="__main__":
    import psycopg2
    with psycopg2.connect(**conn_params) as conn:
        print(get_strategy_by_name(conn,"deafualt supreme court israel"))