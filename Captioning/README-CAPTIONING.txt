# Run Dense captioning using below command.

python Generate-Dense-Caption.py \
--dataset_path ../../cric/train_questions.json \
--image_dir ../../cric/images/img \
--output sample-100-dense-captions.txt \
--batch_size 8 \
--device 0 \
--max_samples 100


# Run Region captioning using below command.

python Generate-Region-Captions.py \
--dataset_path ../../cric/train_questions.json \
--image_dir ../../cric/images/img \
--output sample-100-region-captions.txt \
--batch_size 8 \
--device 0 \
--max_samples 100