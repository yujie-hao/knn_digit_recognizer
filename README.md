# KNN algorithm

## Pros and Cons

### Pros
* Simple algorithm, easy to be implemented
* Doesn't need to train the data in advance
* When the data set is small, the accuracy is as good as complex/advanced algorithm

### Cons
* Lazy algorithm
* Memory-based learning, requires large space to store the data
* Time-complexity is high: need to compare the test data with all known data (training data). As the training data set grows, the time complexity grows.

## Improvement
* Use [Kd tree] to reduce time complexity --> O(log(n))