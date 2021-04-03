# BioNumQA-BERT: Answer Biomedical Questions by Numerical Facts Using Deep Language Representation Model


Contact: Ye Wu	
Email: ywu@cs.hku.hk  

## Introduction
Biomedical question answering (QA) plays an increasingly significant role in medical knowledge translation. However, current biomedical QA datasets and methods have limited capacity, as they neglected the important role of numerical facts in biomedical QA. In this paper, we introduced BioNumQA, a novel biomedical QA dataset that answers research questions by the relevant numerical facts. To solve BioNumQA, we explored a novel input representation approach to improve upon the popular biomedical language model BioBERT.  Our method, BioNumQA-BERT, introduced numerical encodings to represent the numerical values in the input text. Our experiments show that BioNumQA-BERT significantly outperformed other state-of-art models, including DrQA, BERT and BioBERT (39.0% vs 33.2% in strict accuracy). To improve the generalization ability of BioNumQA-BERT, we further pretrained it on a large biomedical text corpus and achieved an additional 2.5% improvement in strict accuracy.

## Download

We provide the pretrained BioNumQA-BERT model implemented with multi-dimension numerical encoding [here](https://drive.google.com/file/d/13pZECqmak75wpXeyPX-k1BqPJ_bPLyQ2/view?usp=sharing).
The BioNumQA dataset is already in `./data/.` To split the dataset for cross validation,
```shell
python split_data.py
```

## Installation

```shell
git clone https://github.com/LeaveYeah/BioNumQA-BERT.git
cd BioNumQA-BERT
```
## Prerequisition
BioNumQA-BERT requires Python ≥ 3.7. 
Make sure you have PyTorch ≥ 1.5.1 installed, please refer [here](https://pytorch.org/) for installing a suitable PyTorch version.
For other prerequisites, run the following command,
```python
pip install -r requirements.txt
```




## Fine-tuning BioNumQA-BERT

### For Question Answering on BioNumQA
After downloading the pre-trained weights, unpack it to any directory you want, and we will denote this as $MODEL_DIR. Let $DATA_DIR be the dataset directory. Also set $OUT_DIR as a directory for QA outputs. For example, 
```bash
export MODEL_DIR=./pretrained_weights/BioNumQA-BERT
export DATA_DIR=./data/bnqa_fold0
export OUT_DIR=./out/bnqa_fold0
```
Following command runs fine-tuning code on BioNumQA with default arguments.
```bash
python run_bionumqa.py --model_name_or_path $MODEL_DIR --do_train --do_eval --max_seq_length=512 --data_dir "$DATA_DIR$i" --train_file train.json --predict_file dev.json  --per_gpu_train_batch_size=8 --learning_rate=5e-5 --num_train_epochs=4.0 --output_dir=$OUT_DIR --max_answer_length 512 --gradient_accumulation_steps 4
```

### Validate Test Results
Set  $MODEL_DIR as the directory of downloaded pretrained BioNumBERT model. The following command will run the cross validation with default arguments.
```bash
cd BioNumQA-BERT
export $MODEL_DIR = #your downloaded pretrained BioNumBERT model
sh test.sh
```

