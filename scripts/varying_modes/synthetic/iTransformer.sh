export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

interval=1
time_unit=week

mode=temporal
python -u run.py \
  --is_training 1 \
  --model_id syn_t_${time_unit}_24_4 \
  --model $model_name \
  --data synthetic \
  --mode $mode \
  --features M \
  --freq w \
  --seq_len 24 \
  --pred_len 4 \
  --label_len 12 \
  --e_layers 4 \
  --enc_in 77 \
  --dec_in 77 \
  --c_out 77 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --train_epochs 30 \
  --batch_size 16 \
  --learning_rate 0.000001 \
  --lradj type3 \
  --patience 10 \
  --itr 1

mode="spatial"
python -u run.py \
  --is_training 1 \
  --model_id syn_s_${time_unit}_24_4 \
  --model $model_name \
  --data synthetic \
  --mode $mode \
  --features M \
  --freq w \
  --seq_len 24 \
  --pred_len 4 \
  --label_len 12 \
  --e_layers 4 \
  --enc_in 77 \
  --dec_in 77 \
  --c_out 77 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --train_epochs 30 \
  --batch_size 16 \
  --learning_rate 0.000001 \
  --lradj type3 \
  --patience 10 \
  --itr 1

mode="spatio-temporal"
python -u run.py \
  --is_training 1 \
  --model_id syn_st_${time_unit}_24_4 \
  --model $model_name \
  --data synthetic \
  --mode $mode \
  --features M \
  --freq w \
  --seq_len 24 \
  --pred_len 4 \
  --label_len 12 \
  --e_layers 4 \
  --enc_in 77 \
  --dec_in 77 \
  --c_out 77 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --train_epochs 30 \
  --batch_size 16 \
  --learning_rate 0.000001 \
  --lradj type3 \
  --patience 10 \
  --itr 1

mode="none"
python -u run.py \
  --is_training 1 \
  --model_id syn_n_${time_unit}_24_4 \
  --model $model_name \
  --data synthetic \
  --mode $mode \
  --features M \
  --freq w \
  --seq_len 24 \
  --pred_len 4 \
  --label_len 12 \
  --e_layers 4 \
  --enc_in 77 \
  --dec_in 77 \
  --c_out 77 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --train_epochs 30 \
  --batch_size 16 \
  --learning_rate 0.000001 \
  --lradj type3 \
  --patience 10 \
  --itr 1


