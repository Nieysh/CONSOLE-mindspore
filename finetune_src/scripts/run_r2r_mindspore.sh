ob_type=pano
feedback=sample

features=vitbase_r2rfte2e
ft_dim=768

ngpus=1
seed=0


flag="--root_dir ../datasets

      --dataset r2r

      --vlnbert ${vlnbert}
      --ob_type ${ob_type}
      
      --world_size ${ngpus}
      --seed ${seed}
      
      --num_l_layers 9
      --num_x_layers 4
      
      --hist_enc_pano
      --hist_pano_num_layers 2

      --fix_lang_embedding
      --fix_hist_embedding

      --features ${features}
      --feedback ${feedback}

      --max_action_len 15
      --max_instr_len 60

      --image_feat_size ${ft_dim}
      --angle_feat_size 4

      --lr 1e-6
      --iters 300000
      --log_every 200
      --batch_size 4
      --optim adamW

      --ml_weight 0.2      

      --feat_dropout 0.4
      --dropout 0.5"


# inference
python main.py --output_dir ../datasets/R2R/trained_models/vitbase-finetune/CONSOLE_test \
              --test --submit --CONSOLE_test --dataset r2r \
              --bert_ckpt_file "Input the location of bert pretrained ckpt file here" \
              --resume_file "Input the released best val_unseen ckpt file here"


