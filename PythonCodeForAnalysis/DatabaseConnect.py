import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT 

def connect (password, db_name):
    DB_user = "postgres"      
    DB_pass = password
    DB_host = "localhost"
    DB_name = db_name 
    conn = psycopg2.connect(dbname = DB_name, user = DB_user, password = DB_pass, host = DB_host)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    return conn.cursor ()

