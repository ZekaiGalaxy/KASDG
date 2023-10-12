from tqdm import tqdm
from . import params
from transformers import BartTokenizer
import torch
from . import data_utils
from . import utils

# tokenizer
args = params.parse_args()
tokenizer = BartTokenizer.from_pretrained(args.p2r_model_path)
special_tokens = ['<s1>', '<s0>']+['_eqn_', '_cite_','_ix_','i . e .','_url_']
tokenizer.add_special_tokens({'additional_special_tokens':special_tokens})

# predict 
@torch.no_grad()
def predict(model, valid_loader, args):
    model.eval()
    device = model.device
    generated=[]
    for idx,data in enumerate(tqdm(valid_loader)):
        input_ids, attention_mask, style_ids, style_mask = [x.to(device) for x in data]
        output = model.generate(
                input_ids=(input_ids, style_ids), max_length=40, use_cache=True, 
                attention_mask=(attention_mask, style_mask), num_beams=4, 
                do_sample=False, early_stopping=True
        )    
        # decode
        tgt_ids = output.to('cpu')
        tgt_ids = tgt_ids.tolist()
        generated += tokenizer.batch_decode(tgt_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return generated

# back translation, style sentences -> pseudo context
@torch.no_grad()
def back_translation(r2p_model, args, response_ids, response_mask, retrieved_ids, retrieved_mask):
    r2p_model.eval()
    device = r2p_model.device

    def process(x):
        x = x.replace('<s0>','<s1>')
        return x if '<s1>' in x else '<s1> ' + x

    output = r2p_model.generate(
        input_ids=response_ids, max_length=50, use_cache=True,
        attention_mask=response_mask, num_beams=4,
        do_sample=False, early_stopping=True,
    )
    tgt_ids = output.to('cpu')
    tgt_ids = tgt_ids.tolist()
    r2p_model.train()

    response = tokenizer.batch_decode(response_ids.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
    generated = tokenizer.batch_decode(tgt_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    # filter out repeat
    remain_ids = []
    for i in range(len(generated)):
        if not utils.filterout(generated[i], response[i], args.filter_repeat):
            remain_ids.append(i)
    
    generated = [process(x) for x in generated]
    context_ids, context_mask = data_utils.tokenize_batch(generated)
    context_ids, context_mask = context_ids.to(device), context_mask.to(device)
    try:
        context_ids, context_mask, response_ids, response_mask, retrieved_ids, retrieved_mask = context_ids[remain_ids], context_mask[remain_ids], response_ids[remain_ids], response_mask[remain_ids], retrieved_ids[remain_ids], retrieved_mask[remain_ids]
    except:
        pass

    return  context_ids, context_mask, response_ids, response_mask, retrieved_ids, retrieved_mask

def p2r_train(p2r_model, data):
    context_ids, context_mask, response_ids, response_mask, style_ids, style_mask = data[:6]
    response_ids[response_mask==0] = -100 # needed because CE no ignore pad in BartForConditionalGeneration
    inputs={
        "input_ids": (context_ids, style_ids),
        "attention_mask": (context_mask, style_mask),
        "labels": response_ids,
        "return_dict": True,
    }
    loss = p2r_model(**inputs).loss
    return loss

def r2p_train(r2p_model, data):
    context_ids, context_mask, response_ids, response_mask = data[:4]
    context_ids[context_mask==0] = -100 # needed because CE no ignore pad in BartForConditionalGeneration
    inputs={
        "input_ids": response_ids,
        "attention_mask": response_mask,
        "labels": context_ids,
        "return_dict": True,
    }
    loss = r2p_model(**inputs).loss
    return loss