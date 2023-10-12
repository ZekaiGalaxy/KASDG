from src import params
from src import data_utils
from src import train_utils
from src import utils
import torch
from transformers import set_seed, get_linear_schedule_with_warmup
from transformers import BartTokenizer, BartConfig, BartForConditionalGeneration
from src.modeling import BartForConditionalGeneration as KASDG
from tqdm import tqdm
import copy

# args
args = params.parse_args()
args.save_path = 'checkpoint/'
set_seed(args.seed)
model_name = params.get_exp_name(args)
device = f'cuda:{args.gpu}'

# tokenizer
tokenizer = BartTokenizer.from_pretrained(args.p2r_model_path)
special_tokens = ['<s1>', '<s0>']+['_eqn_', '_cite_','_ix_','i . e .','_url_']
tokenizer.add_special_tokens({'additional_special_tokens':special_tokens})
config = BartConfig.from_pretrained(args.p2r_model_path)
args.len_tokenizer = len(tokenizer)

# forward model(p2r): post to response
p2r_model = KASDG.from_pretrained(args.p2r_model_path, config=config, args=args)
p2r_model.model.init_style_encoder_weights()
p2r_model.resize_token_embeddings(len(tokenizer))
p2r_model.set_args(args)

# backward model(r2p): response to post
r2p_model = BartForConditionalGeneration.from_pretrained(args.r2p_model_path)
r2p_model.resize_token_embeddings(len(tokenizer))

args.p_s0 = 1.0 if args.retrieve_method == 'r' else 0.0

# data
train_s0_loader, valid_loader, style_loader = data_utils.prepare_data(args)

# optimizer & scheduler
t_total = len(train_s0_loader) * args.n_epoch
p2r_optimizer = torch.optim.AdamW(p2r_model.parameters(), lr=args.p2r_lr, eps=args.adam_epsilon)
p2r_scheduler = get_linear_schedule_with_warmup(p2r_optimizer, num_warmup_steps=0, num_training_steps=t_total)
r2p_optimizer = torch.optim.AdamW(r2p_model.parameters(), lr=args.r2p_lr, eps=args.adam_epsilon)
r2p_scheduler = get_linear_schedule_with_warmup(r2p_optimizer, num_warmup_steps=0, num_training_steps=t_total)

total_step = 0
# train
def train_step(epoch):
    global total_step, train_s0_loader, style_loader, valid_loader, valid_references
    pbar = tqdm(range(len(train_s0_loader)))

    p2r_model.train()
    r2p_model.train()

    for idx, (s0_btach, style_bacth) in enumerate(zip(train_s0_loader, style_loader)):
        s0_batch = [s.to(device) for s in s0_btach]
        style_batch = [s.to(device) for s in style_bacth]
        bt_batch = copy.deepcopy(s0_batch)

        p2r_optimizer.zero_grad()
        r2p_optimizer.zero_grad()

        # forward
        forward_loss = train_utils.p2r_train(p2r_model, s0_batch)
        s0_loss = forward_loss['total_loss']

        # backward
        r2p_loss = train_utils.r2p_train(r2p_model, bt_batch)

        # back translation
        if total_step >= args.bt_freeze_step:
            pseudo_batch = train_utils.back_translation(r2p_model, args, *style_batch)
            backward_loss = train_utils.p2r_train(p2r_model, pseudo_batch)
            s1_loss = backward_loss['total_loss']

            bt_weight = min(1,(total_step - args.bt_freeze_step) / (args.bt_warmup_step - args.bt_freeze_step)) * args.lambda_bt
            loss = s0_loss + bt_weight * s1_loss + args.lambda_r2p * r2p_loss
        else:
            loss = s0_loss + args.lambda_r2p * r2p_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(p2r_model.parameters(), args.clip)
        torch.nn.utils.clip_grad_norm_(r2p_model.parameters(), args.clip)
        p2r_optimizer.step()
        p2r_scheduler.step()
        r2p_optimizer.step()
        r2p_scheduler.step()

        if total_step % args.save_every_step == 0:
            train_utils.predict(p2r_model, valid_loader, args)
            utils.save_model(p2r_model, args.save_path+'/'+model_name, total_step)
        
        total_step+=1
        pbar.update(1)  
        pbar.set_description(f'Epoch {epoch} | Step {total_step} | loss {loss.item():.4f} | s0_loss {s0_loss.item():.4f} | r2p_loss {r2p_loss.item():.4f}')

def train():
    r2p_model.to(device)
    p2r_model.to(device)
    for epoch in range(args.n_epoch):
        train_step(epoch)

train()