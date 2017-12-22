# TVM Benchmark
A benchmark set for [TVM](https://github.com/dmlc/tvm). We mainly target the applications that are **NOT** used for deep learning.

## Case List
### Clustering
#### K-Means
A method of vector quantization which is popular for cluster analysis in data mining.

### Classification
#### Logistic Regression
The implementation of multinomial logistic regression using logistic loss gradient decent.

#### Linear Support Vector Machine
The implementation of binary classification using hinge loss gradient decent.

(TODO: Add regularizer. Now we simply use zero regularizer.)

### Regression
#### Linear Least Square
The implementation of linear regression using squared loss gradient decent.

### Image Processing
#### Image Blurring
Simple image blurring process that generates a 224x244 gray image and blurs it using a 3x3 window.

(TODO: Implement reading images from files.)
