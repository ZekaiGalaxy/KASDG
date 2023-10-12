from src import train_utils
from src import data_utils
from src import params
from src.modeling import BartForConditionalGeneration as KASDG
from transformers import BartTokenizer, BartConfig, BartForConditionalGeneration
import torch

# args
args = params.parse_args()
load_path = args.load_path + f'/step_{args.load_step}.pt'
pred_path = args.load_path + f'/step_{args.load_step}.txt'
device = f'cuda:{args.gpu}'

# model
tokenizer = BartTokenizer.from_pretrained(args.p2r_model_path)
special_tokens = ['<s1>', '<s0>']+['_eqn_', '_cite_','_ix_','i . e .','_url_']
tokenizer.add_special_tokens({'additional_special_tokens':special_tokens})
config = BartConfig.from_pretrained(args.p2r_model_path)
p2r_model = KASDG(config, args)
p2r_model.resize_token_embeddings(len(tokenizer))
p2r_model.set_args(args)
p2r_model.load_state_dict(torch.load(load_path, map_location=device)['model_state_dict'])
p2r_model.to(device)

# data
train_s0_loader, valid_loader, style_loader = data_utils.prepare_data(args)
predict = train_utils.predict(p2r_model, valid_loader, args)

# save predict
with open(pred_path, 'w') as f:
    for line in predict:
        f.write(line+'\n')