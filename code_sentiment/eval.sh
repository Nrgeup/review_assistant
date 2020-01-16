CUDA_VISIBLE_DEVICES=0
  python \
  eval.py \
     --model="sentiment-predict" \
     --checkpoint_name="1578973534"  \
     --model_file="model_params.model" \
     --pretrained_bert_name='bert-base-uncased'\
     --bert_dim=768\
     --polarities_dim=2




