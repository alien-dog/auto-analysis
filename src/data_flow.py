import torch
import ast
from data_execute import DataExecutor
from data_calculate import Calculate
from torch import nn

class WorkFlow():

    def __init__(self):
        self.de = DataExecutor()
        self.cal = Calculate()

    def generateEmbeddingData(self, doc_id):
        data_embedding = []
        data = self.de.data_query_origin(doc_id)
        for element in data:
            vector = self.cal.embedding_open_resource(element[3])
            data = {
                'data_id': element[0],
                'data_embedding': vector,
                'doc_id': doc_id
            }
            data_embedding.append(data)
        self.de.data_insert(data_embedding)

    def getSimilarity(self, query, doc_id):

        vector = self.cal.embedding_open_resource(query)
        data = self.de.data_query(doc_id)
        array = []
        map_return = {}
        for element in data:
            ret = nn.functional.cosine_similarity(torch.tensor(ast.literal_eval(element[3])), torch.tensor(ast.literal_eval(vector)), dim=0)
            data_element = {
                'data_id': element[2],
                'similarity': ret.item(),
                'doc_id': doc_id,
                'query_data': query
            }
            array.append(data_element)
            map_return[element[2]] = ret.item()
        max_key = max(map_return, key=lambda k: map_return[k])
        print(max_key)
        return array
