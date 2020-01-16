CUDA_VISIBLE_DEVICES=0
  python \
  eval.py \
     --model="useful-predict" \
     --checkpoint_name="1579143972"  \
     --model_file="model_params.model" \
     --pretrained_bert_name='bert-base-uncased'\
     --bert_dim=768\
     --polarities_dim=2




