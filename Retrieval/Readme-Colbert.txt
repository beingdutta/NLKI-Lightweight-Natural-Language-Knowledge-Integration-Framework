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