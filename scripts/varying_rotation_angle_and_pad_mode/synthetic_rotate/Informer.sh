export CUDA_VISIBLE_DEVICES=0

model_name=Informer

interval=1
time_unit=week

theta_deg=0
python -u run.py \
  --is_training 1 \
  --model_id r_syn_${time_unit}_24_4 \
  --model $model_name \
  --data synthetic_rotate \
  --pad_mode manual \
  --theta_deg $theta_deg \
  --noise_strength 0.1 \
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

theta_deg=30
python -u run.py \
  --is_training 1 \
  --model_id r_syn_${time_unit}_24_4 \
  --model $model_name \
  --data synthetic_rotate \
  --pad_mode manual \
  --theta_deg $theta_deg \
  --noise_strength 0.1 \
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

theta_deg=45
python -u run.py \
  --is_training 1 \
  --model_id r_syn_${time_unit}_24_4 \
  --model $model_name \
  --data synthetic_rotate \
  --pad_mode manual \
  --theta_deg $theta_deg \
  --noise_strength 0.1 \
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

theta_deg=60
python -u run.py \
  --is_training 1 \
  --model_id r_syn_${time_unit}_24_4 \
  --model $model_name \
  --data synthetic_rotate \
  --pad_mode manual \
  --theta_deg $theta_deg \
  --noise_strength 0.1 \
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

theta_deg=90
python -u run.py \
  --is_training 1 \
  --model_id r_syn_${time_unit}_24_4 \
  --model $model_name \
  --data synthetic_rotate \
  --pad_mode manual \
  --theta_deg $theta_deg \
  --noise_strength 0.1 \
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


theta_deg=30
python -u run.py \
  --is_training 1 \
  --model_id r_syn_${time_unit}_24_4 \
  --model $model_name \
  --data synthetic_rotate \
  --pad_mode wrap \
  --theta_deg $theta_deg \
  --noise_strength 0.1 \
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

theta_deg=45
python -u run.py \
  --is_training 1 \
  --model_id r_syn_${time_unit}_24_4 \
  --model $model_name \
  --data synthetic_rotate \
  --pad_mode wrap \
  --theta_deg $theta_deg \
  --noise_strength 0.1 \
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

theta_deg=60
python -u run.py \
  --is_training 1 \
  --model_id r_syn_${time_unit}_24_4 \
  --model $model_name \
  --data synthetic_rotate \
  --pad_mode wrap \
  --theta_deg $theta_deg \
  --noise_strength 0.1 \
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

theta_deg=90
python -u run.py \
  --is_training 1 \
  --model_id r_syn_${time_unit}_24_4 \
  --model $model_name \
  --data synthetic_rotate \
  --pad_mode wrap \
  --theta_deg $theta_deg \
  --noise_strength 0.1 \
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
