CUDA_VISIBLE_DEVICES=0
  python \
  eval.py \
     --model="sentiment-summary" \
     --checkpoint_name="1579061867"  \
     --model_file="model_params.model" \
     --pretrained_bert_name='gpt2'



