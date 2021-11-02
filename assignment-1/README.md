Build 2 layer network for Handwritten Digit Classification using pure python/numpy/PyTorch (without the layers from PyTorch)
Input data 
x
:
20
×
784
 (
20
 is the batch size, 
784
=
28
×
28
))
Hidden layer of 64 neurons
label 
y
:
20
×
10
Weight Variable 
W
 with random initialization (e.g., Uniform[-0.5,0.5])
Bias variable 
b
 with zeros or no bias
Sigmoid nonlinearity in the hidden layer
Use softmax activation => 
s
o
f
t
m
a
x
(
W
⋅
x
+
b
)
Use OneHot encoding and MSE loss! Not the best way for classification but just for practice
OneHot encoding: 5 --> [0 0 0 0 0 1 0 0 0 0]
Train the model on MNIST through gradient descent

