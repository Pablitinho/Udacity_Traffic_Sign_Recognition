# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/writeup/sample_1.png "Sample 1"
[image2]: ./images/writeup/sample_2.png "Sample 2"
[image3]: ./images/writeup/sample_3.png "Sample 3"
[image4]: ./images/writeup/sample_4.png "Sample 4"
[image5]: ./images/writeup/sample_5.png "Sample 5"

[image6]: ./images/writeup/original_color.png "Original Color"
[image7]: ./images/writeup/original_gray.png "Original gray"
[image8]: ./images/writeup/original_gray_Equalized.png "Equalized"

[image9]: ./images/writeup/before_augmentation.png "Before Augmentation"
[image10]: ./images/writeup/after_augmentation.png "After Augmentation"

[image11]: ./images/writeup/120_small.bmp "Sample 1"
[image12]: ./images/writeup/general_caution_small.bmp "Sample 2"
[image13]: ./images/writeup/priority_road_small.bmp "Sample 3"
[image14]: ./images/writeup/stop_signal_small.bmp "Sample 4"
[image15]: ./images/writeup/Turn_right_ahead_small.bmp "Sample 5"

[image16]: ./images/writeup/120_result.png "Result 1"
[image17]: ./images/writeup/caution_result.png "Result 2"
[image18]: ./images/writeup/priority_result.png "Result 3"
[image19]: ./images/writeup/stop_result.png "Result 4"
[image20]: ./images/writeup/right_result.png "Result 5"

[image21]: ./images/writeup/caution_layer_3_output.png "Result 5"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32,32,3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because is more fast to compute the training and also the classification. The shape is the most important in the image, the color information is not needed for this kind of problem.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image6]
![alt text][image7]

As a last step, I normalized the image data because the dataset has not a homogeneous histogram and most of the image are quite dark or with a lot of brightness. Here there is an example after apply the equalization normalization in the grayscale image.

![alt text][image8]

I decided to generate additional data because after train the neuronal network I was not able to get more than 91% of accuracy in the validation dataset. 

To add more data to the the data set, I used the augmentation technique because make similarity to some images that are included in the validation dataset that is not included in the training data set. In the augmentation it was used translation, rotation, brightness and affine warping.

Here is an example of an original image and an augmented image:

![alt text][image9]
![alt text][image10]

The difference between the original data set and the augmented data set is the following ... 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer           | Description                                |
|-----------------|--------------------------------------------|
| Input           | 32x32x1 Gray image                         |
| Convolution 5x5 | 1x1 stride, same padding, outputs 32x32x16 |
| Relu            | Activation Operation                       |
| Max pooling     | 2x2 stride,outputs 16x16x16                |
| Convolution 5x5 | 1x1 stride, same padding, outputs 16x16x32 |
| Relu            | Activation Operation                       |
| Max pooling     | 2x2 stride,outputs 8x8x16                  |
| Convolution 3x3 | 1x1 stride, same padding, outputs 8x8x64   |
| Relu            | Activation Operation                       |
| Full connected            | Full connected Neuronal Network |
| SoftMax  | Softmax operation |


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used Adam optimizer because in my experiment I realize that is was the best option. After some experiment with batch size of 128 the system converge much faster. For testing I used up to 40 epochs but I have seen that after 20 epochs the system converge with a training of 99.8% of accuracy and the validation with around 97.7%. For learning rate I used the value 0.001 because it was converging fast with a good accuracy. The dropout was set to 0.5 and the L2 regularization value was 0.000001.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were (In notebook):
* training set accuracy of 97.8%
* validation set accuracy of 96.6% 
* test set accuracy of 94.0%

My final model results were (Personal experiments):
* training set accuracy of 99.8%
* validation set accuracy of 97.7% 
* test set accuracy of ??%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen? 
* I used at first with one convolution layer and after I included more layers
* 
* What were some problems with the initial architecture?
* The accuracy was not very good and the convergence was quite slow.
  
* How was the architecture adjusted and why was it adjusted? 
 
* Mainly playing with the L2 value, the number of filters in the CNN and the learning rate.


* Which parameters were tuned? How were they adjusted and why?


* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Lenet5

* Why did you believe it would be relevant to the traffic sign application?
* Because it giving good results.
 
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
  
  Using other images out of the training,validation and test, i.e., images from internet and observe the prediction.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image11]
![alt text][image12]
![alt text][image13]
![alt text][image14]
![alt text][image15]

The first image might be difficult to classify because there are more speed signals in the dataset, the second one because there are more signals with triangles, the third one and forth one were randomly selected and the last one because can be confused with the left signal.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed Limit 120      		| Speed Limit 120   									| 
| General Caution    		| General Caution									|
| Priority					| Priority										|
| Stop	      		              | Stop				 				|
| Turn Right			       | Turn Right      							|

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

![alt text][image16]
![alt text][image17]
![alt text][image18]
![alt text][image19]
![alt text][image20]

In the 5 traffic signs chosen I got almost 1.0 per each prediction. Check at the end of the notebook for more detail and more examples.



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Here we have the output of the layer 3 in the caution traffic sign:

![alt text][image12]
![alt text][image21]

We can see that the output from the layer are mainly features in the image like corners and orientations ( a corner can be a specific orientation, it is just one way to do the interpretation)
