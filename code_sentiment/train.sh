CUDA_VISIBLE_DEVICES=0
  python \
  train.py \
     --model="sentiment-predict" \
     --if_train=True \
     --if_load_from_checkpoint=False \
     --checkpoint_name="None"  \
     --model_file="model_params.model" \
     --seed=666 \
     --num_epoch=30000000 \
     --pretrained_bert_name='bert-base-uncased'\
     --max_seq_len=80\
     --batch_size=100\
     --dropout=0.1\
     --bert_dim=768\
     --polarities_dim=2\
     --lr=1e-4\
     --max_grad_norm=1.0 \
     --num_total_steps=1000 \
     --num_warmup_steps=100



