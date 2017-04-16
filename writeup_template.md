# **Traffic Sign Recognition** 

---

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: report_pictures/data_visualization.JPG "Visualization"
[image4]: report_pictures/1.jpg "Traffic Sign 1"
[image5]: report_pictures/2.jpg "Traffic Sign 2"
[image6]: report_pictures/3.jpg "Traffic Sign 3"
[image7]: report_pictures/4.jpg "Traffic Sign 4"
[image8]: report_pictures/5.jpg "Traffic Sign 5"
[image9]: report_pictures/softmax.JPG "Softmax"

---
### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the IN[8] (second) code cell of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic
signs data set:

Number of training examples = 34799
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the IN[9] (third) code cell of the IPython notebook. 

Here is an exploratory visualization of the data set. It is a bar chart showing how the data looks like. I've also shown an example image of each class in the ipython notebook.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the IN[11] (fifth) code cell of the IPython notebook.

As a first step, I decided to equalize the images to get a better contrast. Then I used imageDataGenerator from the keras framework to rotate, shift, shear and zoom the images to augment the data. The augmented data was too much for my system to handle and hence I used the joblib package in python to save intermediate results.

#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the IN[11] (fifth) code cell of the IPython notebook.  the same preprocessing steps to it to avoid any confusion.

I used imageDataGenerator from the keras framework to rotate, shift, shear and zoom the images to augment the training data. The augmented data was too much for my system to handle and hence I used the joblib package in python to save intermediate results. My final training set had 215000 number of images. My validation set and test set have not been augmented and contain the same number of images as were in the pickle file.

#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the IN[2] cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6	|
| Max pooling	with dropout |2x2 stride,  outputs 14x14x6			|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 |
| Max pooling	with dropout | 2x2 stride,  outputs 5x5x16 |
| Fully connected	| outputs 120 |
| RELU	with dropout |												|
| Fully connected	| outputs 84 |
| RELU	with dropout |												|
| Fully connected	| outputs 43 |
| Softmax |   |

#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the IN[3] cell of the ipython notebook. The model has been trained with adam optimizer and the batch size was set to 128 since I was using a CPU to train it. I ran it for 12 epochs. I chose the values of sigma and mu to be 0.1 and 0 respectively as I choose previously for the LeNet architecture.

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the IN[3] and IN[4] cell of the Ipython notebook.

My final model results were:
* validation set accuracy of 0.946
* test set accuracy of 0.951

If an iterative approach was chosen:
### * What was the first architecture that was tried and why was it chosen?
I choose the LeNet architecture because it the basis for the paper on traffic sign identification by Yann Lecun
### * What were some problems with the initial architecture?
The initial architecture wasn't working very well with the RGB images, I also tried variations in different colorspaces but the accuracy on the validation set was just half a percent short of 93%. So I discarded it and added a few layers of my own.
### * How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
As discussed earlier, I found the the original architecture wasn't giving the results as per the expectations. So I tried and fiddled around with different layers. I also did a lot of reading and found out that dropouts have been encorporated into neural networks to increase their robustness. So I encorporated it into every layer of LeNet. Moreover I also learned that pooling adds to the robustness, so I added maxpool layers them as well. The complete architecture is given in a tabular form in answer 3.
### * Which parameters were tuned? How were they adjusted and why?
I tried and experimented with dropout values and the batch size, since I didn't have a GPU, it was hard to experiment with more parameters due to time contraints. I finally settled with no dropouts i.e. keep_prob = 1.0, which essentially means that the LeNet architecture now just includes the max pooling layers.
### * What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
Since LeNet was already used earlier for traffic sign classifications I tried to build on it using standard pooling and dropout techniques. Pooling and dropouts add to the robustness of the neural network . It doesn't rely much on a particular feature from the data and tries to learn all possible traits of the data to classify it.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The second image is difficult to classify because its not on the dataset. I just wanted to experiment with the network and hence I chose these.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| No stopping     			| Roundabout Mandatory 										|
| Yield					| Yield											|
| No entry	      		| No entry					 				|
| Speed limit (60km/hr)			| Speed limit (60km/hr)    							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 95.1. The network couldn't classify the No stopping sign since it was not on the dataset but that was to be expected. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the last cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of above 90%), and the image does contain a stop sign. The results are quite similar for all the images except the No stopping sign. The no stopping sign as we can see below has a split between the labels 40 and 38 with approximate probabilites 0.9 and 0.1 respectively. This confusion in the no stopping sign is to be expected since it has predominant blue and red colors with the blue far more than the red. So the network classified it as keep right (38) and roundabout mandatory (40) which are both blue. However, the roundabout mandatory has more blue in a concentric circular shape so it has more probability than the keep right.

![alt text][image9]
