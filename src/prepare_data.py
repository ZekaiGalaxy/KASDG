from sentence_transformers import SentenceTransformer, util
import numpy as np
from tqdm import tqdm
import utils
import params
import os
import torch

# data/
# - dialogue
# - style
# - test
# - context_embeddings.pkl
# style/
# - retrieval.pkl
# - style_retrieval.pkl
# - test_retrieval.pkl
# - style_embeddings.pkl
# - test_embeddings.pkl

model = SentenceTransformer('model/all-MiniLM-L12-v2', device="cuda")

# retrieve 
def retrieve(x,y,embed_x,embed_y):
    topk_idx_lst = []
    num = len(x)//500 + 1
    # 500 as a batch
    for i in range(num):
        xi = embed_x[i*500:(i+1)*500]
        sim_matrix = util.cos_sim(xi, embed_y)
        topk_idx = torch.topk(sim_matrix, 101, dim=1).indices.tolist()
        topk_idx_lst.extend(topk_idx)
    
    return [[y[idx] for idx in topk] for topk in topk_idx_lst]

def preprocess(style):
    def load_dialogue():
        with open("data/dialogue.txt","r") as f:
            data = f.readlines()
            context = [x.strip().split('\t')[1] for x in data]
            response = [x.strip().split('\t')[2] for x in data]
        return context, response

    def load_formal():
        with open("data/formal.txt","r") as f:
            # filter
            formal_corpus = [x.split('\t')[2] for x in f.readlines()]
            formal_corpus = [x.strip() for x in formal_corpus if not (("I am sorry" in x) or ("I don't know" in x))] 
        return formal_corpus
    
    def load_holmes():
        with open("data/holmes.txt","r") as f:
            holmes_corpus = f.readlines()
            holmes_corpus = [x.strip() for x in holmes_corpus]
        return holmes_corpus
    
    def load_arxiv():
        with open("data/arxiv.txt", "r") as f:
            # remove the same
            arxiv_corpus = f.readlines()
            arxiv_corpus = [x.strip() for x in arxiv_corpus]
            arxiv_corpus1 = list(set(arxiv_corpus))
            arxiv_corpus1.sort(key=arxiv_corpus.index)
        return arxiv_corpus1

    def load_formal_test():
        with open("data/formal_test.txt","r") as f:
            test_data = f.readlines()
            test_context = []
            test_response = {}
            for x in test_data:
                _, context, response = x.strip().split('\t')
                if not context in test_response.keys():
                    test_response[context] = [response]
                    test_context.append(context)
                else:
                    test_response[context].append(response)
        return test_context, test_response

    def load_holmes_test():
        with open("data/holmes_test.txt", "r") as f:
            test_data = f.readlines()
            test_context = []
            test_response = {}
            for x in test_data:
                context, response = x.strip().split('\t')[:2]
                if not context in test_response.keys():
                    test_response[context] = [response]
                    test_context.append(context)
                else:
                    test_response[context].append(response)
        return test_context, test_response

    def load_arxiv_test():
        with open("data/arxiv_test.txt", "r") as f:
            test_data = f.readlines()
            test_context = []
            test_response = {}
            for x in test_data:
                context, response = x.strip().split('\t')[:2]
                if not context in test_response.keys():
                    test_response[context] = [response]
                    test_context.append(context)
                else:
                    test_response[context].append(response)
        return test_context, test_response

    dialogue_dataset = load_dialogue()
    if style == "formal":
        style_dataset = load_formal()
        test_dataset = load_formal_test()
    elif style == "holmes":
        style_dataset = load_holmes()
        test_dataset = load_holmes_test()
    elif style == "arxiv":
        style_dataset = load_arxiv()
        test_dataset = load_arxiv_test()
    else:
        print("style error")
        exit()

    style_dataset = style_dataset
    context = dialogue_dataset[0]
    response = dialogue_dataset[1]
    context_test = test_dataset[0]
    label_test = test_dataset[1]

    if not os.path.exists("data/context_embeddings.pkl"):
        context_embeddings = np.array([model.encode(x) for x in tqdm(context)])
        utils.write_list(context_embeddings, "data/context_embeddings.pkl")
    else:
        context_embeddings = utils.read_list("data/context_embeddings.pkl")

    if not os.path.exists("data/response_embeddings.pkl"):
        response_embeddings = np.array([model.encode(x) for x in tqdm(response)])
        utils.write_list(response_embeddings, "data/response_embeddings.pkl")
    else:
        response_embeddings = utils.read_list("data/response_embeddings.pkl")
    
    if not os.path.exists(f"data/{style}/style_embeddings.pkl"):
        style_embeddings = np.array([model.encode(x) for x in tqdm(style_dataset)])
        utils.write_list(style_embeddings, f"data/{style}/style_embeddings.pkl")
    else:
        style_embeddings = utils.read_list(f"data/{style}/style_embeddings.pkl")
    
    if not os.path.exists(f"data/{style}/test_embeddings.pkl"):
        test_embeddings = np.array([model.encode(x) for x in tqdm(context_test)])
        utils.write_list(test_embeddings, f"data/{style}/test_embeddings.pkl")
    else:
        test_embeddings = utils.read_list(f"data/{style}/test_embeddings.pkl")

    # retrieval s0
    context_retrieved = retrieve(context, style_dataset, context_embeddings, style_embeddings)
    response_retrieved = retrieve(response, style_dataset, response_embeddings, style_embeddings)
    retieval = [[context[idx], response[idx], context_retrieved[idx][:100], response_retrieved[idx][:100]] for idx in range(len(context))]
    utils.write_list(retieval, f"data/{style}/retrieval.pkl")
            
    # retrieval s1 (for the training phase after back translation)
    style_retrieved = retrieve(style_dataset, style_dataset, style_embeddings, style_embeddings)
    style_retrieval = [[style_dataset[idx], style_retrieved[idx][1:]] for idx in range(len(style_dataset))]
    utils.write_list(style_retrieval, f"data/{style}/style_retrieval.pkl")

    # test
    test_retrieved = retrieve(context_test, style_dataset, test_embeddings, style_embeddings)
    test_retrieval = [[context_test[idx], label_test[context_test[idx]], test_retrieved[idx][:100]] for idx in range(len(context_test))]
    utils.write_list(test_retrieval, f"data/{style}/test_retrieval.pkl")

if __name__ == '__main__':
    args = params.parse_args()
    if args.preprocess:
        preprocess(args.dataset)
