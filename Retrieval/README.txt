# Install using
pip install colbert-ai

# Or follow the Official Instruction of Colbert Repo (Link Below)

# Colbert Pre-trained checkpoint is from the Official Repo.
./colbert/colbertv2.0/pytorch_model.bin

# Download from ColBERT official Repo
https://github.com/stanford-futuredata/ColBERT

# For Troubleshooting CUDA issues set the below env variables for the current shell session or update in your bash.rc file.
export PATH=/home/user/miniconda3/envs/colbert/bin:$PATH
export CUDA_HOME=/home/user/miniconda3/envs/colbert
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRAATH


# Run using below command.
python ColBERT-Inference.py --subset 100 --checkpoint ./colbert/colbertv2.0 --dataset_path ../cric/test_v1_questions.json --output sample-colbert-output-100.txt


# Finetune ColBERT using the below command.

python ColBERT-Finetune.py \
  --queries Contrastive-Training-Data/testData160000Final/queries_train_FAISS_1pos_1neg.tsv \
  --collection Contrastive-Training-Data/testData160000Final/collection_train_FAISS_1pos_1neg.tsv \
  --mapping Contrastive-Training-Data/testData160000Final/mapping_train_FAISS_1pos_1neg.jsonl \
  --checkpoint colbert-ir/colbertv1.9 \
  --device 0 \
  --bsize 32 --lr 1e-4 --warmup 20000 --doc-maxlen 180 --dim 128 \
  --nway 2 --accumsteps 1 --similarity cosine --use-ib-negatives \
  --inspect

# Finetuned checkpoint can be found on the below HF link:
https://huggingface.co/dutta18/Colbert-Finetuned

# Or, in the below link:
https://drive.google.com/file/d/1ShIA4GtiWQIr_O0DYo5k1bra-B2LA9mj/view?usp=sharing