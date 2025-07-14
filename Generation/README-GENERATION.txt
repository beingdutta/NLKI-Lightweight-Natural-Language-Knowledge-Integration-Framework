# Run explanation generation using below command.

python generate-type5-explanations.py \
--test_json ../../cric/test_v1_questions.json \
--dense_caption ../../main/FlorenceCaptionsAllDatasets/cric/test/denseCaptions_cric_test.txt \
--region_caption ../../main/FlorenceCaptionsAllDatasets/cric/test/denseRegionCaptions_florence_cric_test.txt \
--object_caption ../../main/FlorenceCaptionsAllDatasets/cric/test/objects_florence_cric_test.txt \
--traditional_caption ../../cricImageCaptions/blip_image_captions_full_test_set.txt \
--retrieved_facts ../../cricStoredFacts/cricTestStoredFacts.txt \
--output ./sample-100-type-5-explanations.txt \
--device cuda:0 \
--max_samples 100