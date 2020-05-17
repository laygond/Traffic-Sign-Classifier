# Traffic Sign Classification in Tensorflow V1

This project trains and test a LeNet classification neural network using Tensorflow V1. As regularizations to the LeNet architecture, dropout has been added in the fully connected layers to improve accuracy. The model architecture is trained on traffic signs for recognition and later tested on new images. Prior steps such as detection and alignment are not part of this repo. This implies ONE object per image as oppose to Detection which implies MULTIPLE objects per image. This repo uses [Udacity's CarND-Traffic-Sign-Classifier-Project repo](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project) as a base template and guide. 

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
    │   ├── test.p               # second collection of test images for testing and it comes
    │   ├── train.p              # \\in the same format as training and validation, i.e, resized
    │   └── valid.p              # \\images stored as pickle files 
    └── US_Traffic_Sign_Dataset
```

## Demo File
The `demo.ipynb` file is the main file of this project and has the following sections:

- Load the data set (links to the project data set are included)
- Explore and visualize the data set
- Pre-process the data set
- Model architecture Design
- Train and Validate Model
- Test Model
- Inpection & Analysis of Model

## Data set
The demo file makes use of the German traffic sign dataset to show results. However, once you have run and understood the `demo.ipynb`, feel free to try your own dataset by changing the input directories from 'Load The Data' section in the demo file.

The dataset provided for training and validation are pickle files containing RESIZED VERSIONS (32 by 32) of the [original dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). There exist two test sets for evaluation, one as a pickle file with the same instructions as validation and training set, and the other is `my_test_images` with images I collected from the web.

The following is a catalog of the signs in the dataset as labeled in `signnames.csv`:

![alt text][image2]

The dataset contains 43 classes labeled from [0-42]. The image shape is (32, 32, 3) for all pickle files while `my_test_images` has images of random sizes which are later preprocessed to be resized to (32, 32, 3). <b>ADD YOUR OWN GERMAN TRAFFIC SIGNS FOR TESTING IN  `my_test_images`.</b>

| Set | Samples |
| :---  | :--------: |
| Training Set   | 34799  |
| Validation Set |  4410  |
| Test Set       | 12630  |
| My Test Set    |     6  |

The distribution of the training set is:
![alt text][image3]

#### Pre-processing of data set
Besides resizing all images to (32,32) we have included three types of normalization preprocessing. Normalization is done by substracting from every image the mean and dividing it by the standard deviation.
- The first type takes an abrupt approach by just substracting 128 and divinding by 128.
- The second type uses the mean and std from the training set. 
- The third type uses the mean and std from the Imagenet set which is a collection of images from all over the world. 
```
def preprocess(X, using_type=3):
    """ 
    Preprocess assumes X is of shape [numOfImages, height, width, channels]. 
    Set type number to specify the type of normalization you desire:
        Type 1: Quick way to approximately normalize the data (pixel - 128)/ 128.
        Type 2: Classic normalization based on Training set.
        Type 3: ImageNet normalization
    """
    if (using_type == 1):
        return (X - 128)/ 128
    elif (using_type == 2):
        return (X - np.mean(X_train))/ np.std(X_train)
    elif (using_type == 3):
        RGB_mean = [0.485, 0.456, 0.406]
        RGB_std = [0.229, 0.224, 0.225]
        X = X/255.0
        X[:, 0, :, :] -= RGB_mean[0]
        X[:, 1, :, :] -= RGB_mean[1]
        X[:, 2, :, :] -= RGB_mean[2]
        X[:, 0, :, :] /= RGB_std[0]
        X[:, 1, :, :] /= RGB_std[1]
        X[:, 2, :, :] /= RGB_std[2]
        return X
        
```
A discussion of their performance is examined in the "Analysis" section further below.

## Model Architecture
The model of the neural network has a LeNet architecture with dropout added in the fully connected layers to improve accuracy. The LeNet architecture requires the input image to be 32x32 so every image goes through preprocessing for rescaling and normalization. The following image shows the architecture and its specifications.

![alt text][image4]

#### Note: 

- `d = 075` in this case means you keep 75% of the fully connected layer
- Normalization helps to find faster better weights during training 
- The dropout added to the LeNet architecture increased the training accuracy by 4% than without.
- The optimizer used was Adam
- The Networks parameters were  the following:

![alt text][image5]


## Analysis of Model on Test Images

Here are five random German traffic signs that I found on the web:

![alt text][image6]

The model's top 5 softmax predictions on each of these new traffic signs were:

<b>Stop</b> 

| Prob: | Top 5 Predictions |
| ---:  | :-------- |
| 100.0% | Priority road |
| 0.0%   | End of no passing by vehicles over 3.5 metric tons |
| 0.0%   | No passing for vehicles over 3.5 metric tons |
| 0.0%   | End of no passing |
| 0.0%   | Right-of-way at the next intersection |


<b>Children crossing</b>

| Prob: | Top 5 Predictions |
| ---:  | :-------- |
| 99.64%  | Children crossing |
| 0.3%    | Pedestrians |
| 0.03%   | Right-of-way at the next intersection |
| 0.02%   | Road narrows on the right |
| 0.01%   | Dangerous curve to the right |


<b>Go straight or right</b>

| Prob: | Top 5 Predictions |
| ---:  | :-------- |
| 100.0% | Go straight or right |
| 0.0%   | General caution |
| 0.0%   | Keep right |
| 0.0%   | Roundabout mandatory |
| 0.0%   | Turn left ahead |


<b>Speed limit (30km/h)</b>

| Prob: | Top 5 Predictions |
| ---:  | :-------- |
| 98.78% | Speed limit (30km/h) |
| 0.86%  | Speed limit (20km/h) |
| 0.36%  | Speed limit (70km/h) |
| 0.0%   | Speed limit (50km/h) |
| 0.0%   | Speed limit (80km/h) |


<b>Roundabout mandatory</b>

| Prob: | Top 5 Predictions |
| ---:  | :-------- |
| 99.96% | Roundabout mandatory | 
| 0.04%  | Turn right ahead |
| 0.0%   | Ahead only |
| 0.0%   | Go straight or left |
| 0.0%   | Turn left ahead |


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. There are several reasons why the accuracy is not high. The model was trained on images that were taken from a single camera device. Therefore all these training images resemble specific single camera parameters while the images in `my_test_images`, which were taken from the web, resemble multiple camera parameters. This introduces object distortion which our model is not robust against. Also the preprocessing for `my_test_images` should be specially adjusted to consider aspect ratio. The original resolution of these images before they are resized to (32x32) vary and therefore a great deal of distortion is introduced.This compares favorably to the accuracy on the original test set of 100.0 %

The `STOP` traffic sign failed in the prediction with `Priority Road` so let's inspect the its feature maps in for LeNet's first convolutional, first max pool, and second convolutional layers.

![alt text][image7]


## Drawbacks and improvements
The architecture is very basic, although we were able to reach:

| Set   | Accuracy |
| :---  | --------: |
| Validation Set |   95.4% |
| Test Set       | 100.0%  |
| My Test Set    |     80%  |

Possible improvements would be to make the architecture deeper to reach higher accuracy, incorporate more regularization techniques such as data augmentation. Also evaluate how well the model performs if trained on other countries\` traffic signs as well as include detection to classify multiple traffic signs per image.   


