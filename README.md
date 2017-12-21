# TVM Benchmark
A benchmark set for [TVM](https://github.com/dmlc/tvm). We mainly target the applications that are **NOT** used for deep learning.


## Case List
### K-Means Clustering
A method of vector quantization which is popular for cluster analysis in data mining.

### Logistic Regression
The implementation of multinomial logistic regression using logistic loss gradient decent.

#### TODO:
1. Add one more dimension for weight to model the bias.
2. Add regularizer (now we simply use zero regularizer).
3. Improve the data generation (the current one is too uniform to show the proper trend).

### Polynomial Regression
The implementation of polynomial regression using squared loss gradient decent.

