# Traffic Sign Classification in Tensorflow

This project trains and test a very simple classification neural network. The neural network has a LeNet architecture with dropout added in the fully connected layers to improve accuracy. The model architecture is trained on traffic signs for recognition and later tested on new images. Prior steps such as detection and alignment are not part of this repo. This implies ONE object per image as oppose to Detection which implies MULTIPLE objects per image. This repo uses [Udacity's CarND-Traffic-Sign-Classifier-Project repo](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project) as a base template and guide. 

[//]: # (List of Images used in this README.md)
[image1]: ./README_images/visualization.gif "Visualization"
[image2]: ./README_images/traffic_sign_catalog.png "Catalog"
[image3]: ./README_images/train_set_dist.png "Training Set Distribution"
[image4]: ./README_images/architecture.png "Model Architecture"
[image5]: ./README_images/NNparam.png "Model Parameters"
[image6]: ./README_images/traffic_signs.png "Traffic Signs"
[image7]: ./README_images/stopinspect.png "Stop Sign Inspect"

![alt text][image1]


## Directory Structure
```
.Traffic-Sign-Classifier
├── demo.ipynb                   # Main file
├── .gitignore                   # git file to prevent unnecessary files from being uploaded
├── README_images                # Images used by README.md
│   └── ...
├── README.md
├── net_weights                  # the model's kernel weights are stored here
│   └── ...
└── dataset
    ├── German_Traffic_Sign_Dataset
    │   ├── signnames.csv        # name and ID of each traffic sign in the dataset
    │   ├── my_test_images       # images found by me on the web for testing (ADD YOUR OWN HERE)
    │   │   └── ...
    │   ├── test.p               # second collection of test images for testing and comes
    │   ├── train.p              #in same format as training and validation, i.e, resized
    │   └── valid.p              #images stored as pickle files 
    └── US_Traffic_Sign_Dataset
```

## Demo File
#### Overview
The demo file has the following sections:

- Load the data set (see below for links to the project data set)
- Explore and visualize the data set
- Model architecture Design
- Train and Validate Model
- Test Model
- Inpection & Analysis of Model

#### Dataset
The demo file makes use of the German traffic sign dataset to show results. However, once you have run and understood the `demo.ipynb`, feel free to try your own dataset by changing the input directories from 'Load The Data' section in the demo file. The dataset provided for training and validation are pickle files containing RESIZED VERSIONS (32 by 32) of the [original dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). There exist two test sets for evaluation, one as a pickle file with the same instructions as validation and training set, and the other is `my_test_images` with images I collected from the web. The following is a catalog of the signs in the dataset as labeled in `signnames.csv`:
![alt text][image2]

The dataset contains 43 classes labeled from [0-42]. The image shape is (32, 32, 3) for all pickle files while `my_test_images` has images of random sizes which are later preprocessed to be resized to (32, 32, 3). <b>ADD in `my_test_images` YOUR OWN GERMAN TRAFFIC SIGNS FOR TESTING.</b>

Training Set   : 34799 samples
Validation Set :  4410 samples
Test Set       : 12630 samples
My Test Set    :     6 samples

The distribution of the training set is:
![alt text][image3]


#### Model Architecture
The model of the neural network has a LeNet architecture with dropout added in the fully connected layers to improve accuracy. The LeNet architecture requires the input image to be 32x32 so every image goes through preprocessing for rescaling and normalization. Normalization is done by substracting from every image the mean and dividing it by the standard deviation. Both mean and std are from the training set. The following image shows the architecture and its specifications.

![alt text][image4]

My final model consisted of the following layers:

###### Note: 
![alt text][image5]

- Normalization helps to find faster better weights during training 
- The dropout added to the LeNet architecture increased accuracy by 4%
- The optimizer used was Adam


## Analysis of Model on Test Images

Here are five random German traffic signs that I found on the web:

![alt text][image6]

The model's top 5 softmax predictions on each of these new traffic signs were:

For <b>Stop</b> The top 5 softmax results:
|Prob:| Prediction|
|:--:|:--:|
| 100.0%   | Priority road |
| 0.0%   | End of no passing by vehicles over 3.5 metric tons |
| 0.0%   | No passing for vehicles over 3.5 metric tons |
| 0.0%    |End of no passing |
| 0.0%   |Right-of-way at the next intersection |

For "Children crossing" The top 5 softmax results:
	 99.64%   : Children crossing 
	 0.3%   : Pedestrians 
	 0.03%   : Right-of-way at the next intersection 
	 0.02%   : Road narrows on the right 
	 0.01%   : Dangerous curve to the right 

For "Go straight or right" The top 5 softmax results:
	 100.0%   : Go straight or right 
	 0.0%   : General caution 
	 0.0%   : Keep right 
	 0.0%   : Roundabout mandatory 
	 0.0%   : Turn left ahead 

For "Speed limit (30km/h)" The top 5 softmax results:
	 98.78%   : Speed limit (30km/h) 
	 0.86%   : Speed limit (20km/h) 
	 0.36%   : Speed limit (70km/h) 
	 0.0%   : Speed limit (50km/h) 
	 0.0%   : Speed limit (80km/h) 

For "Roundabout mandatory" The top 5 softmax results:
	 99.96%   : Roundabout mandatory 
	 0.04%   : Turn right ahead 
	 0.0%   : Ahead only 
	 0.0%   : Go straight or left 
	 0.0%   : Turn left ahead 


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the original test set of 100.0 %

The `STOP` traffic sign failed predicting it as `Priority Road` so let's inspect the feature maps of LeNet's first convolution, first max pool, and second convolution.
![alt text][image7]


## Drawbacks and improvements
With a very basic architecture we were able to reach 95% accuracy. Possible improvements would be to make the architecture deeper to reach higher accuracy, evaluate different traffic sign dataset, and add detection to classify multiple traffic signs per image.   


