export CUDA_VISIBLE_DEVICES=0

model_name=Informer_M

interval=1
time_unit=week

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/crime/ \
  --data_path "crime_${interval}_${time_unit}_pivot.csv" \
  --model_id crime_${time_unit}_2_4 \
  --model $model_name \
  --data crime \
  --features M \
  --freq w \
  --seq_len 2 \
  --pred_len 4 \
  --label_len 1 \
  --e_layers 4 \
  --enc_in 77 \
  --dec_in 77 \
  --c_out 77 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --train_epochs 20 \
  --batch_size 16 \
  --learning_rate 0.000001 \
  --lradj type3 \
  --patience 10 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/crime/ \
  --data_path "crime_${interval}_${time_unit}_pivot.csv" \
  --model_id crime_${time_unit}_4_4 \
  --model $model_name \
  --data crime \
  --features M \
  --freq w \
  --seq_len 4 \
  --pred_len 4 \
  --label_len 2 \
  --e_layers 4 \
  --enc_in 77 \
  --dec_in 77 \
  --c_out 77 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --train_epochs 20 \
  --batch_size 16 \
  --learning_rate 0.000001 \
  --lradj type3 \
  --patience 10 \
  --itr 1 \

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/crime/ \
  --data_path "crime_${interval}_${time_unit}_pivot.csv" \
  --model_id crime_${time_unit}_8_4 \
  --model $model_name \
  --data crime \
  --features M \
  --freq w \
  --seq_len 8 \
  --pred_len 4 \
  --label_len 4 \
  --e_layers 4 \
  --enc_in 77 \
  --dec_in 77 \
  --c_out 77 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --train_epochs 20 \
  --batch_size 16 \
  --learning_rate 0.000001 \
  --lradj type3 \
  --patience 10 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/crime/ \
  --data_path "crime_${interval}_${time_unit}_pivot.csv" \
  --model_id crime_${time_unit}_24_4 \
  --model $model_name \
  --data crime \
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
  --train_epochs 20 \
  --batch_size 16 \
  --learning_rate 0.000001 \
  --lradj type3 \
  --patience 10 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/crime/ \
  --data_path "crime_${interval}_${time_unit}_pivot.csv" \
  --model_id crime_${time_unit}_52_4 \
  --model $model_name \
  --data crime \
  --features M \
  --freq w \
  --seq_len 52 \
  --pred_len 4 \
  --label_len 26 \
  --e_layers 4 \
  --enc_in 77 \
  --dec_in 77 \
  --c_out 77 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --train_epochs 20 \
  --batch_size 16 \
  --learning_rate 0.000001 \
  --lradj type3 \
  --patience 10 \
  --itr 1
