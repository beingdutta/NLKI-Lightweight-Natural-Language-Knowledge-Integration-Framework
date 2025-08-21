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


# Run ViLT Training on CRIC using the CE+SCE Loss combination on Type-5 
# Explanations using the below command. Modify the paths accordingly.

python Train-ViLT-CRIC-SCE-CE-Loss-Type-5.py \
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
--checkpoint ./vilt-chkpt-cric-sce-ce


# Run ViLT Training on CRIC using the normal CE Loss combination on Type-5 
# Explanations using the below command. Modify the paths accordingly.

python Train-ViLT-CRIC-CE-Loss-Type-5.py \
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
--device 0 \
--save_dir ./vilt-chkpt-cric-ce

# Run ViLT Training on CRIC using the normal CE Loss combination on Type-5 
# Explanations using the below command. Modify the paths accordingly.

python Train-ViLT-eSNLIVE-GCE-CE-Loss.py \
--train_csv /home/user/esnli/csv/esnlive_train.csv \
--val_csv /home/user/esnli/csv/esnlive_dev.csv \
--test_csv /home/user/esnli/csv/esnlive_test.csv \
--train_expl /home/user/main/LLAMA-Generated-Explanations/esnli/llama3_explanations_without_labels_florence_dense_captions_esnli/llama3_explanation_with_florence_dense_captions_train_set_esnli.txt \
--val_expl /home/user/main/LLAMA-Generated-Explanations/esnli/llama3_explanations_without_labels_florence_dense_captions_esnli/llama3_explanation_with_florence_dense_captions_test_set_esnli.txt \
--test_expl /home/user/main/LLAMA-Generated-Explanations/esnli/llama3_explanations_without_labels_florence_dense_captions_esnli/llama3_explanation_with_florence_dense_captions_test_set_esnli.txt \
--image_dir /home/user/esnli/flickr30k_images/flickr30k_images \
--batch_size 32 \
--epochs 6 \
--device 0 \
--lr 5e-5 \
--q 0.8 \
--lam 0.6 \
--step_size 3000 \
--output_dir ./vilt-chkpt-cric-gce-ce
