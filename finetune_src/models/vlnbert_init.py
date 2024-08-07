import mindspore

def get_tokenizer(args):
    from mindformers import BertTokenizer
    if args.dataset == 'rxr' or args.tokenizer == 'xlm':
        cfg_name = 'xlm-roberta-base'
    else:
        cfg_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(cfg_name)
    return tokenizer

def get_vlnbert_models(args, config=None):
    
    from models.vilmodel_cmt import NavCMT

    model_class = NavCMT

    model_name_or_path = args.bert_ckpt_file
    new_ckpt_weights = {}
    if model_name_or_path is not None:
        ckpt_weights = mindspore.load_checkpoint(model_name_or_path)
        for k, v in ckpt_weights.items():
            if k.startswith('module'):
                new_ckpt_weights[k[7:]] = v
            else:
                # add next_action in weights
                if k.startswith('next_action'):
                    k = 'bert.' + k
                new_ckpt_weights[k] = v
    
    if args.dataset == 'rxr' or args.tokenizer == 'xlm':
        cfg_name = 'xlm-roberta-base'
    else:
        cfg_name = 'bert_base_uncased'

    from mindnlp.transformers.models.bert import BertConfig
    vis_config_1 = BertConfig()

    from mindnlp.transformers.configuration_utils import PretrainedConfig
    vis_config = PretrainedConfig.from_pretrained(pretrained_model_name_or_path=cfg_name)
    pretrained_config = vis_config_1.to_dict()
    for key, value in pretrained_config.items():
        vis_config.__setattr__(key,value)

    if args.dataset == 'rxr' or args.tokenizer == 'xlm':
        vis_config.type_vocab_size = 2
    
    vis_config.max_action_steps = 100
    vis_config.image_feat_size = args.image_feat_size
    vis_config.angle_feat_size = args.angle_feat_size
    vis_config.num_l_layers = args.num_l_layers
    vis_config.num_r_layers = args.num_r_layers
    vis_config.num_h_layers = args.num_h_layers
    vis_config.num_x_layers = args.num_x_layers
    vis_config.hist_enc_pano = args.hist_enc_pano
    vis_config.num_h_pano_layers = args.hist_pano_num_layers

    vis_config.fix_lang_embedding = args.fix_lang_embedding
    vis_config.fix_hist_embedding = args.fix_hist_embedding
    vis_config.fix_obs_embedding = args.fix_obs_embedding

    vis_config.update_lang_bert = not args.fix_lang_embedding
    vis_config.output_attentions = True
    vis_config.pred_head_dropout_prob = 0.1

    vis_config.no_lang_ca = args.no_lang_ca
    vis_config.act_pred_token = args.act_pred_token
    vis_config.max_action_steps = 50 

    vis_config.args= args

    visual_model = model_class.from_pretrained(
        pretrained_model_name_or_path=None,
        config=vis_config,
        state_dict=new_ckpt_weights)
        
    return visual_model
