export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

interval=1
time_unit=week

height=10
width=10
num_nodes=$((height * width))

alpha=0
python -u run.py \
  --is_training 1 \
  --model_id syn2d_${time_unit}_24_4 \
  --model $model_name \
  --data synthetic_2d \
  --alpha $alpha \
  --features M \
  --freq w \
  --seq_len 24 \
  --pred_len 4 \
  --label_len 12 \
  --e_layers 4 \
  --enc_in $num_nodes \
  --dec_in $num_nodes \
  --c_out $num_nodes \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --train_epochs 30 \
  --batch_size 16 \
  --learning_rate 0.000001 \
  --lradj type3 \
  --patience 10 \
  --itr 1

alpha=0.25
python -u run.py \
  --is_training 1 \
  --model_id syn2d_${time_unit}_24_4 \
  --model $model_name \
  --data synthetic_2d \
  --alpha $alpha \
  --features M \
  --freq w \
  --seq_len 24 \
  --pred_len 4 \
  --label_len 12 \
  --e_layers 4 \
  --enc_in $num_nodes \
  --dec_in $num_nodes \
  --c_out $num_nodes \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --train_epochs 30 \
  --batch_size 16 \
  --learning_rate 0.000001 \
  --lradj type3 \
  --patience 10 \
  --itr 1

alpha=0.5
python -u run.py \
  --is_training 1 \
  --model_id syn2d_${time_unit}_24_4 \
  --model $model_name \
  --data synthetic_2d \
  --alpha $alpha \
  --features M \
  --freq w \
  --seq_len 24 \
  --pred_len 4 \
  --label_len 12 \
  --e_layers 4 \
  --enc_in $num_nodes \
  --dec_in $num_nodes \
  --c_out $num_nodes \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --train_epochs 30 \
  --batch_size 16 \
  --learning_rate 0.000001 \
  --lradj type3 \
  --patience 10 \
  --itr 1

alpha=0.75
python -u run.py \
  --is_training 1 \
  --model_id syn2d_${time_unit}_24_4 \
  --model $model_name \
  --data synthetic_2d \
  --alpha $alpha \
  --features M \
  --freq w \
  --seq_len 24 \
  --pred_len 4 \
  --label_len 12 \
  --e_layers 4 \
  --enc_in $num_nodes \
  --dec_in $num_nodes \
  --c_out $num_nodes \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --train_epochs 30 \
  --batch_size 16 \
  --learning_rate 0.000001 \
  --lradj type3 \
  --patience 10 \
  --itr 1

alpha=1
python -u run.py \
  --is_training 1 \
  --model_id syn2d_${time_unit}_24_4 \
  --model $model_name \
  --data synthetic_2d \
  --alpha $alpha \
  --features M \
  --freq w \
  --seq_len 24 \
  --pred_len 4 \
  --label_len 12 \
  --e_layers 4 \
  --enc_in $num_nodes \
  --dec_in $num_nodes \
  --c_out $num_nodes \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --train_epochs 30 \
  --batch_size 16 \
  --learning_rate 0.000001 \
  --lradj type3 \
  --patience 10 \
  --itr 1
