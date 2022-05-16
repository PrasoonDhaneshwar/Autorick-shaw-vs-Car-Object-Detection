# Auto-rickshaw-vs-Car-Object-Detection
# Instructions
Make sure you have tensorflow-gpu installed.
Follow these installation steps for dependencies and TensorFlow installation.
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md

refer
darknet_to_PascalVOC.py

Create pascal records
https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_pascal_tf_record.py

Run the following command from tensorflow/models/research/ directory.

1. For training, run the following command from tensorflow/models/research/ directory.

	Link to train.py
	https://github.com/tensorflow/models/blob/master/research/object_detection/train.py

		python object_detection/train.py \
    		--logtostderr \
    		--pipeline_config_path=${PATH_TO_YOUR_PIPELINE_CONFIG} \
    		--train_dir=${PATH_TO_TRAIN_DIR}

		Example:

	 	python object_detection/training_twoclass/train.py --logtostderr --train_dir=object_detection/training_twoclass/models/train/ --pipeline_config_path=object_detection/training_twoclass/ssd_mobilenet_v1_coco.config

 2. Export the graph and save the checkpoint to .pb file:
 		python export_inference_graph.py --input_type image_tensor --pipeline_config_path ./rfcn_resnet101_coco.config --trained_checkpoint_prefix ./models/train/model.ckpt-5000 --output_directory ./fine_tuned_model

 		Example:
 		python object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path object_detection/training_twoclass/ssd_mobilenet_v1_coco.config --trained_checkpoint_prefix object_detection/training_twoclass/models/train/model.ckpt-400000 --output_directory object_detection/training_twoclass/fine_tuned_model/

 3. Run the model:
 Make sure you in OpenCV environment
 Run the following command from /tensorflow/models/research/object_detection/training_twoclass directory
 	python testing.py

 Above script takes 'input.mp4' as a video stream and writes into 'output.mp4'
 
Link to output video:
https://drive.google.com/file/d/15bbi2fUAbqyIVqFj1n7lWnB1o6_YX0NO/view?usp=sharing
