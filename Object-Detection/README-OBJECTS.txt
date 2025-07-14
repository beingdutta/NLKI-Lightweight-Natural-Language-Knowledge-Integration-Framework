# Run object-detection using below command.

python Object-Detection.py \
--dataset_path ../../cric/train_questions.json \
--image_dir ../../cric/images/img \
--output sample-100-detected-object.txt \
--batch_size 12 \
--device 0 \
--max_samples 100