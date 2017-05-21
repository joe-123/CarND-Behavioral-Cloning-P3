#**Behavioral Cloning** 

[//]: # (Image References)

[image1]: ./examples/placeholder_small.png "Model Visualization"
[image2]: ./middle.jpg "Center driving"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./normal.jpg "Normal Image"
[image7]: ./flipped.jpg "Flipped Image"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and depths between 6 and 14. In the first two convolutional layers I use  a stride of 2x2 is used. The third convolutional layer does use 1x1 strides. (model.py lines 18-24) To reduce overfitting, all 3 conv. layers are followed by dropouts (0.5). The convolutional layers are followed by 3 dense layers.

The model uses ELU activations to introduce nonlinearity (code line 20). To reduce the amount of data the images are cropped in the first layer of the model. In the second layer the data is normalized using a Keras lambda layer. (code line 18). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).
The batch size was set to 512. This value was figured out by experimenting.
The number of total epochs was set to __. The model was trained in an iterative manner (10-15epochs a time) to be able to test driving behaviour in between (lines x and y).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I mainly used center lane driving. Data for recovering from the left and right sides of the road was not recorded. For this, I just used the left and right cameras. For sections where the model had trouble to stay on the road. I recorded some extra drives. These also just focus on normal driving.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Documentation

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to build a model that is as simple as possible. This would result in very fast training and few problems with overfitting.

My first step was to use a convolution neural network model similar to the LeNet model from project 2. I thought this model might be appropriate because it is already quite powerfull. The problem the self driving model has to solve did not seem too complex to me. It basically has to find the boarders of the road and then needs to match their curvature and position to steering angles.
During testing I found that a model that is a bit more powerfull helps anyway. Also, since I made good experiances with dropout instead of max pooling, I replaced the max pooling layers with dropouts.

I think, the validation loss alone is not directly assosiated to driving behaviour. I found models with very low loss driving really bad. However the validation loss helps observing the training progress. For one model the behaviour mostly became better with lower validation loss. Also the loss gives a hint if further training could improve the model. I used a split of 0.2 and mean squarred error (like recommended).

To combat the overfitting, I used dropout layers and tried to keep the model small.

In general I did a lot of testing and increased the size of my model from time to time until I found one that worked well. Basically I added a third conv. layer and increased the number of filters for the conv. layers. As mentioned above, I trained the model in an intarative manner to be able to do tests in between.

There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I recorded data for these spots and added it to the training data. Also I checked if training more epochs improved the driving.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of 3 convolutional layers with elu activation and dropout followed by a flatten layer and 3 dense layers.
The following table showes the exact setup:

| Layer					| Description								| 
|:------:|:------:| 
| Cropping				| Input: 160x320x3 , Output: 80x320x3			| 
| Lambda					| Normalizing Image to +-0.5 and zero mean	|
| Convolution 5x5 , 6 Filters	| 1x1 stride, valid padding			 		|
| RELU		        		| Activation								|
| Dropout	        			| keep_prob = 0.5							|
| Convolution 5x5 , 10 Filters	| 1x1 stride, valid padding			 		|
| RELU		        		| Activation								|
| Dropout	        			| keep_prob = 0.5							|
| Convolution 5x5 , 14 Filters	| 1x1 stride, valid padding					|
| RELU					|	Activation							|
| Dropout	        			| keep_prob = 0.5							|
| Flatten					| Output = 							|
| Fully connected			| Output = 200						|
| RELU					| Activation							|
| Fully connected			| Output = 50							|
| RELU					| Activation								|
| Fully connected			| Output = 1								|
| Linear					| Activation								|


####3. Creation of the Training Set & Training Process

The first idea for capturing data was to procude a very small but informative/diverse data set. This would speed up training drastically. However, I realized that it's hard to train bigger models on such small data sets. Even if the data is collected carefully. Also I found that the success in training a good driving model very much depens on the composition of the training data.
Therefore I recorded a lot of specialized data sets (e.g. driving the bridge, driving straight, etc.), so that i was able to play around with different combinations of these sets.
As a preprocessing step I filter the data with regards to the frequency the steering angles occur in the data. I devide the angles in intervals and limit the number of samples for these intervals. 

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:
![alt text][image7]
![alt text][image6]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
