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



Analyzing the respective number of class labels within the training set can help to acquire some additional knowledge about the training set and the (Machine Learning) problem in general. The corresponding plot is visualized below and shows the class label distribution of the training set. In theory one would want to have a uniform distribution, which means that every single class has an identical number of samples in the training set. Wan we can learn from the plot below is for instance is that we could expect lower prediction accuracy for classes where we only have roughly 200 samples available for the training. This hasn´t necessarily be the case though, because the respective class can have very well distinguishable features so even a low number of test images can lead to a great prediction performance. Having that said, the point here is rather to understand i.e. to learn that for some classes it might be beneficial to add additional samples. 

[]: https://github.com/dschmoeller/03TrafficSignClassifierCNN/blob/master/writeupImages/classDistributionTrainingSet.png



## Design and Test a Model Architecture

#### 1. Preprocessing

Preprocessing is a vital step to make the input data (images in this case) comparable. This is crucial for the backpropagation step during training, where the gradients are calculated. There might be training examples with a rather high resolution and high value variation. Those training samples would be more sensitive to the gradient descent than other training samples with lower resolution. That´s why a good and necessary practice is to standardize the data, so they have **zero mean** and **unit variance**. In this project, the **scale( )** functionality from the **preprocessing** module of **scikit-learn** was used to achieve this.  Due to the fact that the input image comprises three channels (RGB), standardization was applied to each single channel individually. The three channels have been stacked again afterwards. The corresponding code snippet is shown below. Note that preprocessing is not only be applied in the training step but also for later prediction. 

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

[]: https://github.com/dschmoeller/03TrafficSignClassifierCNN/blob/master/writeupImages/TestSampleStandardized.png



#### 2. Model Architecture

Convolutional layers have to be proven to show good results for image classification problems. The reason for that is the fact that convolutions are like filters which can identify certain features in images, for instance lanes, edges and even shapes or objects in higher level filters. Also, convolutions are cheaper than just applying brute force fully connected layers. The reason for this is that the Neural Network rather learns the filter parameters of predefined filter structures rather than additionally building this structure using only fully connected layers. That´s the reason why the model architecture of this project comprises three convolutional layers. In order to decrease the number of parameters, Max Pooling is applied.  The outcome of the three concatenated convolutional layers is flattened and feed into three fully connected layers. Relu is consistently used through the entire architecture as activation function. An overview of the final model architecture is shown below. 

[]: https://github.com/dschmoeller/03TrafficSignClassifierCNN/blob/master/writeupImages/Architecture.png



#### 3. Model Training

The above described model has been trained using the following hyperparameters: 

- Epochs: **200**
- Batch size: **64**
- Optimizer: **Adam** (Stochastic gradient descent)
- Learning rate: **0.001**



#### 4. Solution Approach

The LeNet architecture was used as a starting point but could not achieve the desired 93% accuracy on the validation set. That´s why the strategy was to increase the model complexity. This has been done by increasing both the number of parameters and depth within the model architecture. After increasing the model complexity, it was desired to overfit the model and apply regularization using dropout layers afterwards. However, even with 200 epochs of training, there was no overfitting visible in the validation accuracy numbers (i.e. the validation accuracy converges but doesn´t a point where the values start decreasing). The batch size was used to be rather small in order to prevent overshooting the cost minimum but rather slowly converge towards it. Since applying dropout layers didn´t show promising results, the final model doesn´t comprise regularization. The validation accuracy which has been achieved with this model is **95.4 %**. Steps to improve this number could be to further increase the model complexity and add additional training data (e.g. applying data augmentation techniques).     





## Test a Model on New Images

#### 1. Acquiring New Images

There have been five images of German traffic signs found on the web. They are stored here:

[]: https://github.com/dschmoeller/03TrafficSignClassifierCNN/tree/master/trafficSignPictures

Pictures 1, 3, 4 and 5 should be rather easy to classify as there are almost no disruptive factors in them. However, picture 2 might be hard to classify, because this type of speed limit 30 sign is rather special. Also the lighting conditions are relatively bad for this one. The images have been resized using the **cv2.resize( )** function. The format was given as **.png** which comes with four channels. That´s why only the three (RGB) channels have been extracted.     



#### 2. Performance on New Images

It turns out that the trained model does a good job on classifying pictures 1, 3, 4 and four. As expected, picture 2 is indeed being misclassified. This gives a prediction performance of **80 %** for these five images. As a comparison, the accuracy on the test set was identified as **93,8 %**. Note that picture 2 was chosen to be hard to classify on purpose. So, choosing the right and easy to classify input images would (potentially) lead to an accuracy of **100 %**. That´s why comparing both numbers doesn´t make too much sense. However, using challenging pictures helps a lot to identify gaps. For instance, this particular finding shows that the training data might not comprise enough data to classify those special type of sign (30 km/h sign).     



#### 3. Model Certainty - Softmax Proabilites

The top 5 softmax probabilites for the above five pictures shows that the model is relatively certain about each prediction: 

- Speed Limit 20: **[1.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00]**
- Speed Limit 30: **[9.9950969e-01 3.9310352e-04 6.0671326e-05 3.6433790e-05 7.7851709e-08]**
- Speed Limit 50: **[1.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00]**
- Priority: **[1.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00]**
- Stop: **[1.0000000e+00 1.6811786e-26 1.8262277e-31 2.2284581e-33 2.9428920e-35]**

#### 

One can see that the prediction probability is about **100 %** for all five images.  The corresponding predicted labels for each image are shown below: 

| Input Image                | Prediction     | Correctly classified? |
| -------------------------- | -------------- | --------------------- |
| Picture 1 (Speed Limit 20) | Speed Limit 20 | Yes                   |
| Picture 2 (Speed Limit 30) | Stop           | No                    |
| Picture 3 (Speed Limit 50) | Speed Limit 20 | Yes                   |
| Picture 4 (Priority)       | Priority       | Yes                   |
| Picture 5 (Stop)           | Stop           | Yes                   |



It´s interesting to note that even in the one misclassification case, the prediction probability is very high. This means that the Neural Network is very certain that picture 2 is actually a stop sign. My guess is, that the model identifies and highly incorporates the (iron) frame around the actual traffic sign. 


