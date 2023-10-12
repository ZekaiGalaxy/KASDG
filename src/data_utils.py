from . import params
from . import utils
import nltk
from nltk.corpus import stopwords
import torch
from transformers import BartTokenizer
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, RandomSampler

args = params.parse_args()
style_token_num = args.style_token_num
compression_method = args.compression_method

# tokenizer
tokenizer = BartTokenizer.from_pretrained(args.p2r_model_path)
special_tokens = ['<s1>', '<s0>']+['_eqn_', '_cite_','_ix_','i . e .','_url_']
tokenizer.add_special_tokens({'additional_special_tokens':special_tokens})

# remove stopwords
my_stp = ["I'm","i'm","It'll","it'll","you'll","we'll","she'll","he'll","they'll","us","_URL_","I'd","he'd","we'd","she'd","me","I've","it","u","ur","we're","him","her"]
stp = list(set(stopwords.words('english')+my_stp))

def compress(s, mode):
    if mode == 'none':
        return s
    if mode == 'stp':
        # remove stop words
        _s = s.split()
        ss = [x for x in _s if not x.lower() in stp]
        return ' '.join(ss)
    return None
    
def style_tokenize_batch(style_sents, num, mode='none'):
    bos_token, pad_token, eos_token = 0, 1, 2
    input_ids, attention_mask = [], []
    for i in range(len(style_sents)):
        # compression
        s_i = style_sents[i]
        s_i = [compress(s, mode) for s in s_i]
        s_i = [tokenizer(s)['input_ids'][1:] for s in s_i]

        tokens = [bos_token]
        for k in range(len(s_i)):
            # retrieve sentences until larger than token num
            if len(tokens) + len(s_i[k]) <= num and (i < len(s_i)-1):
                tokens.extend(s_i[k])
            else:
                tokens.extend([pad_token]*(num-len(tokens)))
                input_ids.append(tokens)
                attention_mask.append([1]*len(tokens)+[0]*(num-len(tokens)))
                break
                
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)

    return input_ids, attention_mask

def tokenize_batch(data):
    tokenized = tokenizer(data,padding=True,truncation=True,max_length=30)
    input_ids = torch.tensor(tokenized['input_ids'])
    attention_mask = torch.tensor(tokenized['attention_mask'])
    return input_ids, attention_mask

def collate_fn_train(batch):
    context, response, style_sents = zip(*batch)
    context, response, style_sents = list(context),list(response),list(style_sents)
    context_ids, context_mask = tokenize_batch(context)
    response_ids, response_mask = tokenize_batch(response)
    style_ids, style_mask = style_tokenize_batch(style_sents, style_token_num, compression_method)
    return [context_ids,context_mask,response_ids,response_mask,style_ids,style_mask]

def collate_fn_style(batch):
    style_ref, style_sents = zip(*batch)
    style_ref, style_sents = list(style_ref), list(style_sents)
    style_ids, style_mask = tokenize_batch(style_ref)
    paired_style_ids, paired_style_mask = style_tokenize_batch(style_sents, style_token_num, compression_method)
    return [style_ids,style_mask,paired_style_ids,paired_style_mask]

def collate_fn_test(batch):
    context, refs, style_sents = zip(*batch)
    context, refs, style_sents = list(context), list(refs), list(style_sents)
    context_ids, context_mask = tokenize_batch(context)
    style_ids, style_mask = style_tokenize_batch(style_sents, style_token_num, compression_method)
    return [context_ids,context_mask,style_ids,style_mask]

def load_s0_train(dataset, retrieve_method='r'):
    # dialogue
    f = open(args.data_path + '/dialogue.txt', 'r', encoding='utf8')
    dialogue_dataset = [i.strip().split('\t') for i in f.readlines()]
    dialogue_dataset = [[x[1],x[2]] for x in dialogue_dataset]

    # context & response guided retrieval
    retrieval_save_path = args.data_path + f'/{dataset}/retrieval.pkl'
    ret_data = utils.read_list(retrieval_save_path)
    c_ret = [x[2] for x in ret_data]
    r_ret = [x[3] for x in ret_data]
    retrieval = r_ret if retrieve_method == 'r' else c_ret

    # dataset
    train_dataset = []
    assert len(dialogue_dataset) == len(retrieval)
    for i in tqdm(range(len(dialogue_dataset))):
        train_dataset.append(['<s0> '+dialogue_dataset[i][0], dialogue_dataset[i][1]] + [retrieval[i]])

    return train_dataset

def load_style_train(dataset):
    style_save_path = args.data_path + f'/{dataset}/style_retrieval.pkl'
    style_data = utils.read_list(style_save_path)
    style_dataset = []
    for i in range(len(style_data)):
        style_dataset.append([style_data[i][0],style_data[i][1][1:]]) # 1: remove the style sentence itself
    return style_dataset

def load_s1_test(dataset):
    test_save_path = args.data_path + f'/{dataset}/test_retrieval.pkl'
    test_data = utils.read_list(test_save_path)
    test_dataset = []
    for i in tqdm(range(len(test_data))):
        if dataset != 'informal':
            test_dataset.append(['<s1> '+test_data[i][0], test_data[i][1], test_data[i][2]]) # context, [refs], [style sents] 
        else:
            test_dataset.append(['<s0> '+test_data[i][0], test_data[i][1], test_data[i][2]])
    return test_dataset

def prepare_data(args):
    s0_train_data = load_s0_train(args.dataset, args.retrieve_method)
    style_data = load_style_train(args.dataset)
    valid_data = load_s1_test(args.dataset)

    train_s0_dataset = utils.DatasetWrapper(s0_train_data)
    style_dataset = utils.DatasetWrapper(style_data)
    valid_dataset = utils.DatasetWrapper(valid_data)

    train_s0_loader = DataLoader(train_s0_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_train)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_test)
    style_sampler = RandomSampler(style_dataset, replacement=True, num_samples=len(train_s0_dataset))
    style_loader = DataLoader(style_dataset, batch_size=args.batch_size, sampler=style_sampler, collate_fn=collate_fn_style)

    return train_s0_loader, valid_loader, style_loader