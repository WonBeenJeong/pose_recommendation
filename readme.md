
## preparing data ##

####	1. Download raw images
* Edit `crawl_google/config.json` to specify the key words, outputDir and etc
	
	`
	$ Python crawl_google.py
	`
#### 2. Copy the downloaded images to the face detection folder to filter out those images containing zero or multiple humans
	
* the original face detection [repo and set up](https://github.com/tensorflow/models/tree/master/research/object_detection)

	`
	$ scp output_dir  face_detection/raw_data/
	`
	
* (currently, the `raw_data` folder already contains multiple scenes with multiple raw images)
	
	```	
	$ cd  face_detection/models/research  
	```

	```   
	$ export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim  
	```

* Change the paths in line 76-82 and line 134 of file `object_detection/detector.py`, then run:
	`
	$ python object_detection/detector.py file_folder_in_raw_data
	`
## pose feature extraction and pose clustering ##

####	3. Pose estimation: input an image containing a single person, output the coordinates of the key points in that person

- the original [pose estimation repo](https://github.com/eldar/pose-tensorflow)
	
	`
	$ cd pose_estimation/pose-tensorflow
	`
	

*	Change the input, output path in Line 25,28, 40, 42 of file `demo/singleperson.py`, then run:

	`
	$ TF_CUDNN_USE_AUTOTUNE=0 python3 demo/singleperson.py whether_debug
	`	
*	If you would like to see the heat map results, open the debug flag: 

	`	
	TF_CUDNN_USE_AUTOTUNE=0 python3 demo/singleperson.py 1
	`

	Otherwise:

	`
		TF_CUDNN_USE_AUTOTUNE=0 python3 demo/singleperson.py 0
	`
	, it will output a key point file (each line an image) in the current directory.

Run pose clustering based on the pose features:
	
* Change the output path in Line 119 of file `spec_cluster.py`, then run:
	
	`	
	$ python spec_cluster.py
	`
## scene feature: scene parsing ##
(need matlab)

#### 4. PSPNET: model the scene => scene parsing, for each input image, output a 151-D vector, indicating whether the image contains the i-th class of object
	
* the original [PSPNET repo](https://github.com/hszhao/PSPNet)
	
Prepare a list of input image paths, one line each and specify this file in `PSPNet/evaluation/eval_all.m`, Line 17; 
the output path is specified in Line 18


	$ cd PSPNet/evaluation    
	$ export LD_PRELOAD=$LD_PRELOAD:/usr/lib/x86_64-linux-gnu/libstdc++.so.6:/usr/lib/x86_64-linux-gnu/libprotobuf.so.9  
	$ ./run.sh  

## Composition feature ##
* the original [image composition repo](https://github.com/posgraph/coupe.composition-score-calculator)	

	`
	$ cd coupe.composition-score-calculator/Composition Score Calculator
	`

Change the input image path in Line 2 of file `getCompScore_demo.m`

	$ matlab -nodisplay -r "getCompScore_demo" 2>&1   # it will output 4 values for each picture

## transfer learning of pose aesthetic values ##
Use AlexNet as a feature extractor, manually label some data indicating the pose aesthetic values, then train a Lasso.

	$ cd Naïve_transfer_learning

Prepare input label file:
Each line: `image_path \n label_value`

Extract features once:
	
`
$ python main.py train_img_dir test_img_dir
`

Do transfer learning and tune hyper-parameters multiple times:

`	
	$ python read2.py  train_img_path test_img_path
`

	
	
## use explicit pose features, scene features, composition features to do regression on pose aesthetics ##
`
	$ cd naive_regression_explicit_feature
`

Put the feature file under feature/, and put the data (as well as label) file under data/

`	
	$ python train.py
`

## simply do transfer learning on binary classification of human-background interaction detection ##

`
	$ cd interaction_simple_binary_classification
`

The protocols for my labeling  whether there is such an interaction in the images is recorded in `notes`

`
	$ Python naïve_transfer.py 
`
