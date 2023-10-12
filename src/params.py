import argparse

def parse_args():
    parser=argparse.ArgumentParser()

    # path
    parser.add_argument('--data_path',default='data',type=str)
    parser.add_argument('--p2r_model_path',default='model/bart_dialogue',type=str)
    parser.add_argument('--r2p_model_path',default='model/bart',type=str)
    parser.add_argument('--load_path',default='checkpoint',type=str)
    parser.add_argument('--load_step',default=0,type=int)
    parser.add_argument('--save_path',default='checkpoint',type=str)

    # train
    parser.add_argument('--gpu',default=0,type=int)
    parser.add_argument('--batch_size',default=4,type=int)
    parser.add_argument('--n_epoch',default=5,type=int)
    parser.add_argument('--p2r_lr',default=5e-5,type=float)
    parser.add_argument('--r2p_lr',default=5e-5,type=float)
    parser.add_argument('--adam_epsilon',default=1e-8,type=float)
    parser.add_argument('--clip',default=1.0,type=float)
    parser.add_argument('--save_every_step',default=20000,type=int)
    parser.add_argument('--seed',default=42,type=int)

    # data
    parser.add_argument('--dataset',default='formal',type=str,
        choices=['formal','informal','holmes','arxiv'])
    parser.add_argument('--retrieve_method',default='r',type=str,
        choices=['c','r'])
    parser.add_argument('--preprocess',default=0,type=int)
    
    # retrieval
    parser.add_argument('--style_token_num',default=400,type=int)
    parser.add_argument('--compression_method',default='stp',type=str,
        choices=['stp','none'])

    # contrastive
    parser.add_argument('--temperature',default=0.1,type=float)

    # SRKL
    parser.add_argument('--use_KL',default=0,type=int)

    # back translation
    parser.add_argument('--filter_repeat',default=0.3,type=float)
    parser.add_argument('--bt_weight',default=1.0,type=float)
    parser.add_argument('--bt_freeze_step',default=0,type=int)
    parser.add_argument('--bt_warmup_step',default=10000,type=int)

    # lambdas
    parser.add_argument('--lambda_contrastive',default=1.0,type=float)
    parser.add_argument('--lambda_dir',default=1.0,type=float)
    parser.add_argument('--lambda_resp',default=1.0,type=float)
    parser.add_argument('--lambda_bt',default=1.0,type=float)
    parser.add_argument('--lambda_r2p',default=1.0,type=float)

    args=parser.parse_args()

    return args

def get_exp_name(args):
    model_name = f'{args.seed}'
    model_name += f'_({args.dataset}_{args.retrieve_method})'
    model_name += f'_(sel_{args.style_token_num}_{args.compression_method})'

    if args.use_KL:
        model_name += '_(KL)'
    else:
        model_name += f'_(SRKL_{args.lambda_resp}_{args.lambda_dir})'
    
    model_name += f'_(lr_{args.p2r_lr}_{args.r2p_lr})'
    model_name += f'_(cl_{args.temperature}_{args.lambda_contrastive})'
    model_name += f'_(bt_{args.filter_repeat}_{args.bt_weight}_{args.bt_freeze_step}_{args.bt_warmup_step})'

    return model_name
