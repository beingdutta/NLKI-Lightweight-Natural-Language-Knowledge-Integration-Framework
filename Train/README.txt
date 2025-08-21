# Run FLAVA Training using GCE + CE Loss on CRIC on Type-5 Explanations using the below sample 
# command. Modify the path accordingly.

python Train-FLAVA-CRIC-GCE-CE-Loss-Type-5.py \
--train_file /home/user/cric/train_questions.json \
--val_file /home/user/cric/val_questions.json \
--test_file /home/user/cric/test_v1_questions.json \
--image_dir /home/user/cric/images/img \
--error_files /home/user/
--train_expl /home/user/main/LLAMA-Generated-Explanations/cric/llama3_explanations_without_labels_florence_dense_captions_cric/llama3_explanation_with_florence_dense_captions_train_set_cric.txt \
--val_expl /home/user/main/LLAMA-Generated-Explanations/cric/llama3_explanations_without_labels_florence_dense_captions_cric/llama3_explanation_with_florence_dense_captions_val_set_cric.txt \
--test_expl /home/user/main/LLAMA-Generated-Explanations/cric/llama3_explanations_without_labels_florence_dense_captions_cric/llama3_explanation_with_florence_dense_captions_test_set_cric.txt \
--batch_size 32 \
--epochs 6 \
--device cuda:0 \
--checkpoint ./flava-chkpt-cric-gce-ce.pth


# Run FLAVA Training on CRIC using the normal CE loss on Type-5 Explanations using the below command.

python Train-FLAVA-CRIC-CE-Loss-Type-5.py \
--train_file /home/user/cric/train_questions.json \
--val_file /home/user/cric/val_questions.json \
--test_file /home/user/cric/test_v1_questions.json \
--image_dir /home/user/cric/images/img \
--error_files /home/user/text_files/error1.txt /home/user/text_files/error2.txt /home/user/text_files/error3.txt \
--train_expl /home/user/main/LLAMA-Generated-Explanations/cric/llama3_explanations_without_labels_florence_dense_captions_cric/llama3_explanation_with_florence_dense_captions_train_set_cric.txt \
--val_expl /home/user/main/LLAMA-Generated-Explanations/cric/llama3_explanations_without_labels_florence_dense_captions_cric/llama3_explanation_with_florence_dense_captions_val_set_cric.txt \
--test_expl /home/user/main/LLAMA-Generated-Explanations/cric/llama3_explanations_without_labels_florence_dense_captions_cric/llama3_explanation_with_florence_dense_captions_test_set_cric.txt \
--batch_size 32 \
--epochs 6 \
--device cuda:0 \
--checkpoint ./flava-chkpt-cric-ce.pth


# Run ViLT Training on CRIC using the CE+SCE Loss combination Type-5 
# Explanations using the below command. Modify the paths accordingly.

python Train-ViLT-CRIC-SCE-CE-Loss-Type-5.py \
--train_file /home/aritra/cric/train_questions.json \
--val_file /home/aritra/cric/val_questions.json \
--test_file /home/aritra/cric/test_v1_questions.json \
--image_dir /home/aritra/cric/images/img \
--error_files /home/aritra/text_files/error1.txt /home/aritra/text_files/error2.txt /home/aritra/text_files/error3.txt \
--train_expl /home/aritra/main/LLAMA-Generated-Explanations/cric/llama3_explanations_without_labels_florence_dense_captions_cric/llama3_explanation_with_florence_dense_captions_train_set_cric.txt \
--val_expl /home/aritra/main/LLAMA-Generated-Explanations/cric/llama3_explanations_without_labels_florence_dense_captions_cric/llama3_explanation_with_florence_dense_captions_val_set_cric.txt \
--test_expl /home/aritra/main/LLAMA-Generated-Explanations/cric/llama3_explanations_without_labels_florence_dense_captions_cric/llama3_explanation_with_florence_dense_captions_test_set_cric.txt \
--batch_size 32 \
--epochs 3 \
--device cuda:0 \
--checkpoint ./vilt-chkpt-cric-sce-ce


