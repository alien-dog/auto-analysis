import random

import data_execute as data_executor
import requests
from llama_index.embeddings import HuggingFaceEmbedding

URL = 'http://openapi-cloud.pub.za-tech.net/ai/test'
ACCESS_KEY = 'b3a9ff1e6f0d4680b5ea86c6ee72a0fc'
URI_EMBEDDING = '/azure/ada002/embeddings'

# embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

class Calculate():

    def embedding_GPT(self, data):
        headers = {
            'Access-Key': f'{ACCESS_KEY}'
        }
        input = {
            "input": data
        }
        response = requests.post(url=URL + URI_EMBEDDING, headers=headers, data=input)
        return response.content

    def embedding_open_resource(self, data):
        input = {
            "input": data
        }
        response = requests.post(url="http://49bd-34-126-177-51.ngrok-free.app/get_embedding", json=input)
        return response.text
                # Set up ngrok tunnel
        # return embed_model.get_text_embedding(data)
        # return [0.1]
    # def similarity(self, query):

# call = Calculate()
# dog_embeddings = call.embedding_open_resource("this is dog")
# big_dog_embeddings =  call.embedding_open_resource("this is big dog")
# cat_embeddings =  call.embedding_open_resource("this is cat")
# import torch
# from torch import nn
# import ast
# ret = nn.functional.cosine_similarity(torch.tensor(ast.literal_eval(dog_embeddings)),
#                                       torch.tensor(ast.literal_eval(big_dog_embeddings)), dim=0)
#
# ret_cat = nn.functional.cosine_similarity(torch.tensor(ast.literal_eval(dog_embeddings)),
#                                       torch.tensor(ast.literal_eval(cat_embeddings)), dim=0)
# print(ret)
# print(ret_cat)

