from flask import request
from data_flow import WorkFlow
from source_import import app

wf = WorkFlow()

@app.route('/getTopk', methods=['GET'])
def getTopk():
    data = request.get_data()
    k = request.args.get('k')
    query = request.args.get('query')
    doc_id = request.args.get('doc_id')
    return wf.getSimilarity(query, doc_id)

@app.route('/trigger_embedding', methods=['GET'])
def trigger_embedding():
    doc_id = request.args.get('doc_id')
    wf.generateEmbeddingData(doc_id)
    return "success"

@app.route('/trigger_embedding_1', methods=['POST'])
def trigger_embedding_1():
    data = request.get_data()
    return "success"

if __name__ == '__main__':
    # run app in debug mode on port 8080
    app.run(debug=True, port=8080)