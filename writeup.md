# **Traffic Sign Recognition** 



## **Project Goals: **

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report



## Dataset Exploration

#### 1. Dataset Summary

Numpy and python functions have been used in order to extract the following numbers from the given data set: 

- Number of training examples: **34799**
- Number of validation examples: **4410**
- Number of testing examples: **12630**
- Image data shape: **(32, 32, 32)** 
- Number of classes: **43**



#### 2. Exploratory Visualization

A (random) sample from the training set is visualized below. Note that the quality of the image is not brilliant, due to the 32x32 pixel resolution. So, even for a human, classifying this sample correctly as "Speed Limit 70 km/h" is not a trivial. 

[]: https://github.com/dschmoeller/03TrafficSignClassifierCNN/blob/master/writeupImages/TestSample.png



Analysing the respective number of class labels within the training set can help to acquire some additional knowledge about the training set and the (Machine Learning) problem in general. The corresponding plot is visualized below and shows the class label distribution of the trainig set. In theory one would want to have an uniform distribution, which means that every single class has an identical number of samples in the trainig set. Wan we can learn from the plot below is for instance is that we could expect lower prediction accuracy for classes where we only have roughly 200 samples available for the training. This hasn´t necessarily be the case though, because the respective class can have very well distinguishable features so even a low number of test images can lead to a great prediction performance. Having that said, the point here is rather to understand i.e. to learn that for some classes it might be benefitical to add additional samples. 

[]: https://github.com/dschmoeller/03TrafficSignClassifierCNN/blob/master/writeupImages/classDistributionTrainingSet.png



## Design and Test a Model Architecture

#### 1. Preprocessing

Preprocessing is a vital step to make the input data (images in this case) comparable. This is crucial for the backpropagation step during training, where the gradients are calculated. There migth be training examples with a rather high resolution and high value variation. Those training samples would be more sensitive to the gradient descent than other training samples with lower resolution. That´s why a good and necessary practice is to standardize the data, so they have **zero mean** and **unit variance**. In this project, the **scale( )** functionality from the **preprocessing** module of **scikit-learn** was used to achieve this.  Due to the fact that the input image comprises three channels (RGB), standardization was applied to each single channel individually. The three channels have been stacked again afterwards. The corresponding code snippet is shown below. Note that preprocessing is not only be applied in the training step but also for later prediction. 

```python
from sklearn import preprocessing

def normalizeData(inputFeatureSet):
    X_norm = np.zeros_like(inputFeatureSet, np.float32)
    for i, RGB in enumerate(inputFeatureSet): 
        R_scaled = preprocessing.scale(RGB[:,:,0])
        G_scaled = preprocessing.scale(RGB[:,:,1])
        B_scaled = preprocessing.scale(RGB[:,:,2])
        X_norm[i] = np.stack((R_scaled, G_scaled, B_scaled), axis=2) 
    return X_norm

# Normalize the data
X_train_norm = normalizeData(X_train)
X_valid_norm = normalizeData(X_valid)
X_test_norm = normalizeData(X_test)
```

The standardized sample from above looks like this:



#### 2. Model Architecture

kdljflkdsjfldsf



#### 3. Model Training

lkfldfjdlsf



#### 2. Solution Approach

kdljflkdsjfldsf



## Test a Model on New Images

#### 1. Acquiring New Images

lkfldfjdlsf



#### 2. Performance on New Images

kdljflkdsjfldsf



#### 3. Model Certainty - Softmax Proabilites

lkfldfjdlsf

### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ?
* The size of the validation set is ?
* The size of test set is ?
* The shape of a traffic sign image is ?
* The number of unique classes/labels in the data set is ?

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because ...

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because ...

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Stop Sign      		| Stop sign   									|
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .60         			| Stop sign   									|
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


