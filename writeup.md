# **Traffic Sign Recognition**

## Writeup

[//]: #	"Image References"
[image1]: ./data_hist/train_hist.png	"Training examples histogram"
[image2]: ./data_hist/valid_hist.png	"Validation examples histogram"
[image3]: ./data_hist/test_hist.png	"Test examples histogram"
[image4]: ./dl_img/thirty_resized.png	"Traffic Sign 1"
[image5]: ./dl_img/sixty_resized.png	"Traffic Sign 2"
[image6]: ./dl_img/bumpyroad_resized.png	"Traffic Sign 3"
[image7]: ./dl_img/yield_resized.png	"Traffic Sign 4"
[image8]: ./dl_img/stop_resized.png	"Traffic Sign 5"
[image9]: ./preprocess_img/original_img.png	"Original Image"
[image10]: ./preprocess_img/normalized_img.png	"Normalized Image"



You're reading it! and here is a link to my [project code](https://github.com/aamir207/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### Training set statistics

I used the pandas library to calculate summary statistics of the traffic
signs data set:

- The size of training set is **34799**

- ```
  Number of training examples = 34799
  ```

- The size of the validation set is  4410

- ```
  Number of validation examples= 4410
  ```

- The size of test set is **12630**

- ```
  Number of testing examples = 12630
  ```

- The shape of a traffic sign image is **32X32X3**

- ```
  Image data shape = (32, 32, 3)
  ```

- The number of unique classes/labels in the data set is **43**

- ```
  Number of classes = 43
  ```

#### 2. Exploratory visualization

Here is an exploratory visualization of the data set. It is a histogram of the training, validation and test examples. This data give helps get a idea of the number of training examples in each class. Although I did not implement data augmentation, it's clear that certain classes have very few examples and would benefit greatlly from augmentation. The large imbalance in the number of training examples between the classed is likely to cause the model to favor the better represented classes.

![alt text][image1]



![alt text][image2]



![alt text][image3]

### Model Design and Architecture

#### 1. Preprocessing

The only pre-processing step I implemented was an approximate normalization to ensure that the input features had the same distribution (zero mean and equal variance). Below is an example image and the normalized version.

![alt_text][image9]

![alt_text][image10]

I did attempt to convert the to a different color space, specifically to grayscale and HSV, however I found that this led to a degradation in performance, which is why I decided against such conversion



The difference between the original data set and the augmented data set is the following ... 

#### 2. Model Architecture

My final model architecture is basically the LeNet architecture with the addition of dropout and consisted of the following layers:

|      Layer      | Description                              |
| :-------------: | :--------------------------------------- |
|      Input      | 32x32x3 RGB image                        |
| Convolution 5x5 | 1x1 stride, valid padding, outputs 28x28x6 |
|      RELU       |                                          |
|   Max pooling   | 2x2 stride,  outputs 14x14x6             |
| Convolution 5x5 | 1x1 stride, valid padding, outputs 10x10x16 |
|      RELU       |                                          |
|   Max pooling   | 2x2 stride,  outputs 5x5x6               |
|     Flatten     | output 400                               |
| Fully Connected | Input 400 output 120                     |
|      RELU       |                                          |
|     dropout     | keep_prob 0.6                            |
| Fully Connected | Input 120 Output 84                      |
|      RELU       |                                          |
|     dropout     | keep_prob 0.6                            |



#### 3. Training

| Parameter     | Value                 |
| ------------- | --------------------- |
| Batch size    | 128                   |
| Epochs        | 15                    |
| Optimizer     | Adam                  |
| Learning rate | 0.001                 |
| Cost          | softmax cross entropy |



#### 4. Results

My final model results were:

- training set accuracy of ?
- validation set accuracy of ? 
- test set accuracy of ?

If a well known architecture was chosen:

- I decided to use the LeNet architecture for this classification problem. This architecture used 2 convolutional layers and 2 fully connected layers. I used RELU non-linearity and dropout regularization. 
- The reason I chose the LeNet architecture was because the model size seemed sufficient to capture the important details in the input images. It also includes 2 convolutional layers which are ideal for image classification tasks as they preserve the spatial information in the image
- I experimented with sigmoid non-linearity's but found that the performance was inferior to RELU's. I also iterated on different keep_probs to prevent over fitting.
- Initially I found that my architecture suffered from over fitting resulting in good results on the training set, but poor results on the validation and test sets
- In order to reduce over fitting I added dropout to the outputs of the fully connected layers
- The model has a validation accuracy of 95% and a test accuracy of 93%.

### Test a Model on New Images

#### 1. German traffic signs found on web with discussions of qualities

Here are five German traffic signs that I found on the web:

Before I discuss the performance expectations on individual images I believe the noise introduced by resizing interpolation might hinder classification performance. Additionally background details like trees in the 30km/h image might be another factor that affects performance.

The first 2 images are speed limit signs. I expect my model to do a pretty good job of classifying these as the LeNet architecture does a good job of  classifying character based on it's performance on the MNIST data.

![alt text][image4] ![alt text][image5] 

I expect the bumpy road image to be more difficult for the model to classify as it contains elements that overlap with other road signs. Additionally, looking at the histogram for the training set the bumpy road class is under-represented in the dataset and needs to be augmented. The dataset contains a much larger number of examples for the road work sign, which might explain why the road work sign is assigned the the highest softmax probability for this input image. 

![alt text][image6] 

I expect the model to do pretty well with these input images, because they are well represented in the training dataset. The inverted triangle in the of the yield sign is a unique feature that should allow the network to classify it with high confidence. The same applies to the stop sign image



![alt text][image7] ![alt text][image8]

#### 2. Discussion of prediction results of web images

Here are the results of the prediction:

|   Image    | Prediction |
| :--------: | :--------: |
|  30 km/h   |  30 km/h   |
|  60 km/h   |  60 km/h   |
| Bumpy road | Road work  |
|   Yield    |   Yield    |
|    Stop    |    Stop    |

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. The performance of the model is in  line with expectations based on the dataset distribution. The input images which are better represented in the dataset perform significantly better than the under represented class ( bumpy road). This illustrated the importance of data augmentation for deep learning, a technique that I intend to experiment with. In order to improve performance I plan on analyzing the validation/ test images that were incorrectly classified to get a better understanding of the how additional pre-processing, model architecture changes and hyper-parameter tuning might help improve prediction performance.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. 

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability |    Prediction     |
| :---------: | :---------------: |
|     .87     |      30 km/h      |
|     .12     |      80 km/h      |
|    .005     | Bicycles crossing |
|    .001     |       Yield       |
|    .001     |       Stop        |

The second image was also classified correctly as 60 km/h. The model predicted with high confidence (98%) 

| Probability | Prediction |
| :---------: | :--------: |
|    0.98     | Road work  |
|    0.013    |  80 km/h   |
|   5.5e-05   |  120 km/h  |
|   2.64-06   |   20 km    |
|  4.99e-07   |    Stop    |

The third image was classified incorrectly. The sign is bumpy road but the model predicted road with 70% probablity. The second highest softmax probability was in fact bumpy road (14%)

| Probability |        Prediction        |
| :---------: | :----------------------: |
|    0.70     |        Road Work         |
|    0.14     |        Bumpy Road        |
|    0.06     |         No entry         |
|    0.04     |    Bicycles crossing     |
|    0.02     | Dangeroud curve to right |

The fourth image is a yield sign which was also predicted correctly with high level of confidence (100%)

| Probability |                Prediction                |
| :---------: | :--------------------------------------: |
|    1.00     |                  Yield                   |
|  5.119e-18  | No passing for vehicles over 3.5 metric tons |
|  3.514e-20  |               No vehicles                |
|  1.27e-20   |              Priority Road               |
|  2.18e-22   |                Ahead only                |

The fourth image is a stop which was also predited correctlyl with a high confidence of 99%

| Probability |  Prediction   |
| :---------: | :-----------: |
|    0.99     |     Stop      |
|  1.38e-04   |   No Entry    |
|   9.6e-06   | Priority Road |
|  4.07e-06   |     Yield     |
|  1.33e-06   |   Road Work   |



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)

#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

