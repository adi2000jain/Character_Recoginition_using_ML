# Character_Recoginition_using_ML
Character Recognition Using Machine Learning



Abstract:
In the modern world, data extraction from printed or written text from a scanned document or image file and then converting the text into a machine-readable form to be used for data processing like editing or searching has become essential. Advancements in such extraction is overwhelming. This work is an attempt for a way to detect the English letters using machine learning. Datasets contains (A-Z) handwritten English letter images in size 2828 pixels, each alphabet in the image is centre fitted to 2020 pixel box. Each image is stored as Gray-level. It contains 3,72,450 data entries.
Kaggle link-https://www.kaggle.com/sachinpatel21/az-handwritten-alphabets-in-csv-format

Keywords: 
Convolution Neural Network image classifications takes an input image, processes it, and categorizes it into several groups. An input image is seen by computers as an array of pixels, with the number of pixels varying depending on the image resolution.[4]
Activation Functions in a neural network defines how the weighted sum of the input is transformed into an output from a node or nodes in a layer of the network.[5]

Introduction: 
Over the past few years, there’s been an expansion in new research using neural networks to identify written text from a scanned document or image file. These models, mainly developed at tech giants like Google, can generate increasingly accurate results. CNN is one of the most common types of neural networks used to recognize and classify pictures. CNNs are commonly utilized in domains such as object detection, face recognition, and so on. Figure 1 precisely show the working of kernel in a Convolutional Layer.[17] CNN image classifications takes an input image, processes it, and categorizes it into several groups. An input image is seen by computers as an array of pixels, with the number of pixels varying depending on the image resolution.[4]
<img width="159" alt="image" src="https://user-images.githubusercontent.com/43815354/137901840-b52d0dc9-4f4c-4007-babe-2ba3c808209a.png">

Convolution Neural Network image classifications takes an input image, processes it, and categorizes it into several groups. An input image is seen by computers as an array of pixels, with the number of pixels varying depending on the image resolution.[4]
Figure 1 Working of Kernel in a Convolution Layer


Methodology: 
The main ambition of the project is to segregate and identify various English letters using techniques of machine learning in python. First, We proceeded with collection and preprocessing of a dataset that has some data that can be used for both visual and theoretical presentation.Thus, data set is preprocessed using various filters like Gaussian blur, Grayscale conversion etc .The dataset is then divided into categories mainly training (80%) and testing Dataset(20-30%). As the name suggests, the former dataset is used to train the model while the latter is used for model testing.The images are thresholded to keep the image smooth without any sort of hazy gray colors in the image that could lead to wrong predictions. The process is repeated until the end of the file and in case of presence of a shorter frame, same paddings are added to it. 
Maximum pooling, or max pooling, is a pooling operation that calculates the maximum, or largest, value in each patch of each feature map. This helps in down sampling of images and retaining the most important information.[6][7] 
<img width="277" alt="image" src="https://user-images.githubusercontent.com/43815354/137901895-78cc4199-19ed-4532-9e75-da4cbb283e00.png">

Figure 3 Working of Max Pooling
CNN is one of the most common types of neural networks used to recognize and classify pictures. CNNs are commonly utilized in domains such as object detection, face recognition, and so on. CNN image classifications take an input image, processes it, and categorizes it into several groups. An input image is seen by computers as an array of pixels, with the number of pixels varying depending on the image resolution. The code for the Convolution neural network is written in Python [9] using Keras [10] Package. As we add more convolutional layers, the accuracy tends to saturate and then degrades quickly. Features learned from one of the previous layers are stacked with a new layer. So as the layers increase, almost certainly new features can be learned due to residual features which are extending from layers before.
Dataset:

<img width="369" alt="image" src="https://user-images.githubusercontent.com/43815354/137901908-083da758-a388-400c-b6e8-d2f605eaf120.png">

Architecture of CNN: 
As observed after every few stacked layers input of the primary layer gets added to the last layer. This enables features to be learned efficiently by deeper layers. A variety of operations can be performed on the two layers like addition, average, concatenation etc.[11] 
Properties of the Convolutional Layer: 
Additional details of the convolution layers are given in Table 1. 

Table 1. Properties of each convolution layer in the proposed architecture 

Layer Number	Properties Kernel Size	Filter	Activation
Conv_2D_1	(3,3)	32	Relu
Max Pool	Pool size = (2, 2)	-	-
Conv_2D_2	(3,3)	64	Relu
Max Pool	Pool size = (2, 2)	-	-
Conv_2D_3	(3,3)	128	Relu
Max Pool	Pool size = (2, 2)	-	-

The layers used in the proposed architecture are explained below.
Conv2D: This creates a convolutional layer. The first parameter is the filter count, and the second one is the filter size. For example in the first convolution layer we create 16 filters of size 3x3. Relu has been used as the activation function.
MaxPooling2D: Creates a max pooling layer, the only argument is the window size. A 2x5 window has been used. [12]

To standardize our results, we used the following architecture (Table 2) to use each of the models.

Table 2. Complete Architecture of Network Used

Layer Number	Name	Specifications
1	Input Layer	Resize according to requirements of architecture
2	Existing Convolutional Layers (from Table 1)	-
3	Flatten	-
4	Dense	Activation = Relu Shape = 64
5	Dense	Activation = Relu Shape = 128
6	Dense	Activation = Softmax = 26

Flatten: After the convolution and pooling layers we flatten their output to feed into the fully connected layers.[12]
Dense: It is basically used for changing the dimensions of the vector. It receives input from all neurons of its previous layer and performs a matrix-vector multiplication. The values used in the matrix are actually parameters that can be trained and updated with the help of backpropagation.

output = activation(dot(input, kernel) + bias)				(1)

In the equation 1, activation is used for performing element-wise activation and the kernel is the weights matrix created by the layer, and bias is a bias vector created by the layer.[13]

Implementation and Results: 
Using the proposed complete Architecture as given in Table 1 and Table 2, the results obtained are shown in table 3.

Table 3. Results Obtained

Parameter	% obtained/ score achieved
Validation Accuracy	98.31%
Training Accuracy	98.26%
Validation Loss	6.93%
Training Loss	6.62%
<img width="108" alt="image" src="https://user-images.githubusercontent.com/43815354/137901943-b05ea1b2-7e55-4711-89c3-b40c99c7778c.png">

The proposed architecture has also been tested on External Images uploaded by the user.

Conclusion:
The proposed architecture works efficiently on the applied dataset. Using this architecture, the model learns quickly, achieving comparable accuracy on the datasets. The model achieves 98.31 % accuracy which shows the potential of the architecture to perform better with further works. 

Future Work:
Additionally more datasets of other languages shall be tested to generalise the architecture. This also open the future capability of the network to be used for tasks like data extraction from printed or written text from a scanned document or image file. Furthermore with the help of GANs, the deep CNN achieves better performance in terms of classification accuracy compared with that of the traditional CNN. In addition, the overfitting problem raised by CNNs
is mitigated. GANs are unsupervised deep learning techniques. It's usually done using two neural networks: the Generator and the Discriminator. In a gaming context, these two models compete with each other. Real data and data created by the generator would be used to train the GAN model. The discriminator's task is to tell the difference between false and true data.[18] In more detail, two frameworks are designed: the first one, called the 1D-GAN, is based on spectral vectors and the second one, called the 3D-GAN, combines the spectral and spatial features. These two architectures demonstrated excellent abilities in feature extraction and image classification compared with other state-of-the-art methods. In the proposed GANs, PCA is used to reduce the high dimensionality of inputs, which is really important to stabilize the training procedure. [19]However, testing with more data is required. A diversified testing data shall be created in future work to check it with real case scenarios and more noise.



 


REFERENCES


1.	Angel Das (2020),“Convolution Neural Network for Image Processing — Using Keras ”, https://towardsdatascience.com/convolution-neural-network-for-image-processing-using-keras-dc3429056306
2.	Jason B.(2021), “How to Choose an Activation Function for Deep Learning ” , https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/
3.	Computer Science Wiki,(2018),“Max-pooling/Pooling”, https://computersciencewiki.org/index.php/Max-pooling_/_Pooling, (Mar. 1 2021).
4.	Mahajan P.(2020),“Max Pooling”, https://poojamahajan5131.medium.com/max-pooling-210fc94c4f11, (Mar. 1 2021).
5.	Chollet, F., &others.(2015). Keras. GitHub. Retrieved from https://github.com/fchollet/keras
6.	"G. van Rossum, Python tutorial, Technical Report CS-R9526, Centrumvoor Wiskundeen Informatica (CWI), Amsterdam, May 1995."
7.	Arden Dertat (2017), “Convolutional Neural Networks”, https://medium.com/@ardendertat
8.	Arden Dertat (2017), “Applied Deep Learning - Part 4: Convolutional Neural Networks”, https://towardsdatascience.com/applied-deep-learning-part-4-convolutional-neural-networks-584bc134c1e2
9.	Palash Sharma (2020), “Keras Dense Layer Explained for Beginners”, https://machinelearningknowledge.ai/ keras-dense-layer-explained-for-beginners/
10.	“ How to determine the quality and correctness of classification models? Part 2 – Quantitative quality indicators”, https://algolytics.com/how-to-determine-the-quality-and-correctness-of-classification-models-part-2-quantitative-quality-indicators/
11.	Christopher T. (2019), “An introduction to Convolutional Neural Networks”, https://towardsdatascience.com/an-introduction-to-convolutional-neural-networks-eb0b60b58fd7
12.	Mohammed A. (2020), “ Generative Adversarial Networks GANs: A Beginner’s Guide”, https://towardsdatascience.com/generative-adversarial-networks-gans-a-beginners-guide-f37c9f3b7817
13.	Lin Zhu, Yushi Chen, Pedram Ghamis, and Jón Atli (2018), “ Generative Adversarial Networks for Hyperspectral Image Classification”, IEEE Transactions on Geoscience and Remote Sensing. 56, No. 9  


