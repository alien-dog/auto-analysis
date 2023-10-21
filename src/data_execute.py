from source_import import app
from flask_mysqldb import MySQL

app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = '123456'
app.config['MYSQL_DB'] = 'insurance_clause'
app.config['MYSQL_HOST'] = '127.0.0.1'

mysql = MySQL(app)
class DataExecutor():

    def data_insert(self, data_array):
        for data in data_array:
            data_id = data['data_id']
            data_vector = data['data_embedding']
            doc_id = data['doc_id']
            cur = mysql.connection.cursor()
            cur.execute('''INSERT INTO data_embedding ( doc_id, data_id, data_vector) VALUES (%s, %s, %s)''', (doc_id, data_id, data_vector))
        mysql.connection.commit()

    def data_query(self, doc_id):
        cur = mysql.connection.cursor()
        cur.execute(f"SELECT  * FROM data_embedding where \'doc_id\' = 'doc_id'")
        data = cur.fetchall()
        cur.close()
        return data

    def data_query_origin(self, doc_id):
        cur = mysql.connection.cursor()
        cur.execute(f"SELECT  * FROM article_section where \'doc_id\' = 'doc_id'")
        data = cur.fetchall()
        cur.close()
        return data

    def insert_result(self, data_array):
        for data in data_array:
            data_id = data['data_id']
            similarity = data['similarity']
            doc_id = data['doc_id']
            query_data = data['query_data']
            cur = mysql.connection.cursor()
            cur.execute('''INSERT INTO section_vector_result ( doc_id, data_id, similarity, query_data) VALUES (%s, %s, %s, %s)''', (doc_id, data_id, similarity, query_data))
        mysql.connection.commit()