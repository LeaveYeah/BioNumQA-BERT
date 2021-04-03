#!/bin/bash
BASEDIR="."
DATA_DIR="$BASEDIR/data/bnqa_fold"
OUTPUT_DIR="$BASEDIR/out/bnqa_fold"

MODEL_DIR="$BASEDIR/pretrained_weights/"

for i in {0..9}
do
    python run_bionumqa.py --model_name_or_path $MODEL_DIR --do_train --do_eval --max_seq_length=512 --data_dir "$DATA_DIR$i" --train_file train.json --predict_file dev.json  --per_gpu_eval_batch_size=8 --per_gpu_train_batch_size=8 --learning_rate=5e-5 --num_train_epochs=4.0 --output_dir=$OUTPUT_DIR$i --overwrite_output_dir --save_steps 10000 --max_answer_length 512 --gradient_accumulation_steps 4
    
done