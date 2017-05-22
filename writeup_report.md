#**Behavioral Cloning** 

[//]: # (Image References)

[image1]: ./pictures/middle.jpg "Center driving"
[image2]: ./pictures/color.png "Channel grayed out"
[image3]: ./pictures/gray.png "Left side grayed out"
[image4]: ./pictures/bright.png "Increased brightness"
[image5]: ./pictures/dark.png "Decreased brightness"
[image6]: ./pictures/normal.jpg "Normal"
[image7]: ./pictures/flipped.jpg "Flipped"
[image8]: ./pictures/before.png "Before filtering"
[image9]: ./pictures/after.png "After filtering"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed
model definition: lines 127-149

My model consists of a convolutional neural network with 5x5 filter sizes and depths between 6 and 24. To reduce overfitting, the 3 convolutional layers are followed by MaxPooling or dropout layers. The convolutional part is followed by 3 dense layers. The model uses ELU activations to introduce nonlinearity.
To reduce the amount of data that needs to be passed through the network, the images are cropped in the first layer of the model. In the second layer the data is normalized using a Keras lambda layer.

####2. Attempts to reduce overfitting in the model

The model contains MaxPooling and dropout layers in order to reduce overfitting.  A GaussianNoise layer was added to further improve generalization of the model (line 132).

The model was trained and validated on different data sets to ensure that the model was not overfitting (second cell, third last line). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (line 148).
The batch size was set to 32 since I had out of memory errors with lager batches (line 153).
The number of total epochs was set to 3 (line 173). Further training was not necessary since the data set is quite large.

####4. Appropriate training data

I had a massiv bug in my code which I didn't find for a long time. Therefore I recorded a lot of data. I recorded 8 rounds of normal driving in different resolutions, directions etc. Also I recorded extra data for the tougher spots like the bridge and the curves with the dirty sides. Data for recovering from the side of the track was recorded to.
In total I recorded nearly 64.000 samples (including left and right camera) in different sets so I can combine them to optimize the training result.

###Model Architecture and Training Documentation

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to build a model that is as simple as possible. This would result in very fast training and few problems with overfitting.

My first step was to use a convolution neural network model similar to the LeNet model from project 2. I thought this model might be appropriate because it is already quite powerfull and delivered some good results in first tests.
The problem the self driving model has to solve did not seem too complex to me. It basically has to find the boarders of the road and then needs to match their curvature and position to steering angles.

I think, the validation loss alone is not directly assosiated to driving behaviour. I found models with very low loss driving really bad. However the validation loss helps observing the training progress. Also the loss gives a hint if further training could improve the model. I used a split of 0.1 and mean squarred error (like recommended).

To combat the overfitting, I used MaxPooling, dropout and Gaussian noise layers and tried to keep the model small. Also I used a lot of data with additional augmentation for training. The generator for loading and augmenting the data can be found in lines 60-125.

In general I did a lot of testing and increased the size of my model from time to time until I found one that worked well. Basically I added a third convolutional layer and increased the number of filters for these layers.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (lines 127-149) consisted of 3 convolutional layers with elu activation and dropout followed by a flatten layer and 3 dense layers.
The following table showes the exact setup:

| Layer					| Description								| 
|:------:|:------:| 
| Cropping				| Input: 160x320x3 , Output: 80x320x3			| 
| Lambda					| Normalizing Image to +-0.5					|
| Convolution 5x5 , 8 Filters	| strides=(1, 1), padding='valid'			 	|
| ELU		        		| Activation								|
| MaxPooling        			| pool_size=(2, 2), strides=None, padding='valid'	|
| Convolution 5x5 , 16 Filters	| strides=(1, 1), padding='valid'			 	|
| ELU		        		| Activation								|
| MaxPooling        			| pool_size=(2, 2), strides=None, padding='valid'	|
| Convolution 5x5 , 24 Filters	| strides=(1, 1), padding='valid'				|
| ELU					| Activation								|
| Dropout	        			| keep_prob = 0.5							|
| Flatten					| 										|
| Fully connected			| Output = 400							|
| ELU					| Activation								|
| Fully connected			| Output = 100							|
| ELU					| Activation								|
| Fully connected			| Output = 1								|



####3. Creation of the Training Set & Training Process

The first idea for capturing data was to procude a very small but informative/diverse data set. This would speed up training drastically. However a bug in my code prevented the model from learning. Since I didn't see the bug I tried to solve the problem by recording more data (which didn't help of cause).

I did a lot of center driving. 8 rounds on the test track (changing direction, resolution, quality) and 2 rounds on the challange track. Additionally I recorded seperate data for different spots on the track (bride, curves etc.) and also data for recovering from the side of the road. This should help the model learn to deal with the tough spots on the track.
All the data was recorded into different folders so I can combine them the way I want to. This way I could experiment a lot with using different data for training. (lines 21-25 and 157)

Here is an example image of center lane driving:

![alt text][image1]

After the collection process, I had 64000 data points (incl. left and right camera). Splitting in training and validation set is done right at the beginning when the drive logs are being read. I put 10% of the data into a validation set (line56). Since the recorded data has a strong bias towards driving straight I filter the data / logs befor I load the images. Basically I devide the steering angles of 0.0-1.0 into 100 invervals and limit the number of samples for each interval. See lines 38-47.

Example histograms are shown below (asymmetric because it's before flipping!):
![alt text][image8]
![alt text][image9]

The images from the left and right camera are used to train the model recovering from sides of the road. The steering angle is adjusted by 0.25.

For training I then augmented the images in a generator in different ways:

* flipping
* graying out the left or right side of the image or
* graying out single color chanells completely
* changing brightness

Augment was done to help the model generalize. Example images for the augmentation steps can be seen belo.

Flipping:
![alt text][image6]
![alt text][image7]

Graying out side of the image:
![alt text][image3]

Graying out color channel:
![alt text][image2]

Changing brightness:
![alt text][image4]
![alt text][image5]

Further preprocessing was done directly in the neural net (lines 130-132):

* Cropping the top and bottom of the image (seeing the ski is not relevant for the model)
* normalizing the images to +- 0.5 to make improve training
* adding gaussian noise to the images, which is also augmentation

Shuffeling is done several times after reading the drive logs (line35) and in the generator (line 64).

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. Training for 3 epoches was more than enogh to make the model pass the track. I used an adam optimizer so that manually training the learning rate wasn't necessary.