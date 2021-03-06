
# Table of Contents

1.  [Machine Learning](#org28f55a0)
2.  [Statistics & Probability Theory](#org8e7dc0a)
3.  [Linear Models](#orgb19b2d1)
4.  [Frequent Interview Questions](#org3c87a81)
5.  [Hadoop & Spark](#org59ed870)
6.  [A/B Testing & Hypothesis Testing](#orgda98359)



<a id="org28f55a0"></a>

# Machine Learning


## Bayes rule

-   Assign x to the class with the largest posterior class probability given x.
-   Bayes rule minimizes the risk function.


## kNN


### Pros:


### Cons:


## Decision Tree


### Pros:

-   inexpensive to construct
-   interpretability


### Cons:

-   greedy, not global optimization
-   instability: an error at the top will affect everything below (ensemble can solve this)


## Random Forests


### Random?

-   Bagging: create B bookstrap samples from the data and fit B maximum decision trees.
-   At each split, find the best split from a subset of predictors (p/3). This controls overfitting.


### Why a subset of variables?

-   de-correlates the trees, the smaller m, the smaller the correlation


### Pros:

-   number of variables used in each split can control overfitting
-   stable results


## Boosting


### Pros:

-   works well with trees


### Cons:

-   can overfit, need to limit number of iterations and each model's complexity


## Clustering


### Basics

-   Cluster homogeneity: Within sum of square error (WSSE, sum of square of the distance to each point's centroid) is used as performance measure. The smaller the better. Usually decreases when k increase.
-   Cluster separation: Between sum of square error (BSSE, weighted sum of square of the distance between cluster centroids to total mean). The bigger the better. Usually increases when k increases.
-   Silhouette coefficient:


### Possible dissimilarity measures


### Hierarchical clustering

1.  Setup

    1.  An agglomerative (bottom-up) approach.
    2.  Start with each point as its own cluster
    3.  Merge the closest pair of clusters at each step until only one cluster left.
    4.  Can be visualized by a dendrogram.

2.  Pros

    -   A series of possible solutions
    -   Computationally fast
    -   Meaningful hierarchy, usually
    -   Does not require raw data, only distance matrix

3.  Cons

    -   No (explicit) optimization criterion, greedy, No objective way to choose the final solution (where to stop).
    -   Different ways of measuring distance between clusters can give rise to very different solutions


### K-means

1.  Setup

    1.  Initialize starting centroids arbitrarily
    2.  assign cluster to each point based on the closest centroid
    3.  recompute centroids based on new assignment
    4.  repeat 2) and 3) until convergence

2.  Operational meaning

    -   K-means obtains a (locally) optimal solution to the optimization problem of minimizing WSSE, given a fixed number of clusters.


### K-means++

1.  Algorithm

    A way to initialize centroids. Reduce chances of bad clustering due to bad initial centroids.
    
    1.  Choose the first centroid randomly from data points
    2.  For each point, compute its distance \(D(x)\) to its nearest existing centroid
    3.  Choose a new centroid from the data points. The probability of being chosen is proportional to its \(D(x)^2\)
    4.  Repeat 2) and 3) until all centroids are initialized

2.  Advantages over K-means

    -   The initializtion takes longer, but the clustering tends to converge faster. So in all it is usually faster than K-means
    -   Avoids getting poor clustering with regard to the objective function (min WSSE). A good example is on [wiki page of K-means++](https://en.wikipedia.org/wiki/K-means%252B%252B).


### EM algorithm

1.  Setup

    -   An example of model based clustering
    -   Assuming data are independent samples from a mixture model
    -   This algorithm is a "soft version" that generalizes the K-means algorithm
    -   this algorithm ultimately obtains a local optimum of the likelihood function

2.  Algorithm

    1.  initialize cluster probabilities and distribution parameters arbitrarily
    2.  compute posterior probabilities of latent variable \(P(Z_k|X_n)\) (cluster assignments) for each point
    3.  recompute distribution parameters based on probabilities computed in last step
    4.  repeat 2) and 3) until convergence


<a id="org8e7dc0a"></a>

# Statistics & Probability Theory


## Uncorrelated but not independent

-   example: \(X \sim U(-1,1),\ Y = X^2\), or \(Y = |X|\)


## Convergence

-   WLLN: \(X_1, ... X_n iid\), then the sample mean converges in probability to theoretical mean \(\mu\)


## Bootstrap

-   treate available data as the actual population and sample with replacement from it
-   another interpretation is to sample based on the empirical CDF
-   generate B bootstrap samples and compute the statistics, then can use these statistics to get CI of an estimator
-   bootstrap sometimes will fail when the sample population is heavy tailed and don't follow CLT


## Basic statistics

-   skewness: \(E(X - \mu)^3 / \sigma^3\)
    -   left skewed (heavier left tail) if negative (mean < median < mode)
    -   right skewed (heavier right tail) if positive (mode < median < mean)
-   kurtosis: \(E(X - \mu)^4 / \sigma^4 - 3\)
    -   indicates heavier(positive) or lighter(negative) tails compared to a normal distribution with the same mean and variance


## Hypothesis testing


### Power Analysis

1.  Statistical Power: The probability of correctly rejects the null hypothesis when H1 is true, Pr(reject H0 | H1).

2.  Power function: The probability of rejecting H0 given the parameter, as a function of the parameter.


<a id="orgb19b2d1"></a>

# Linear Models


## Linear regression


### Heteroscedasticity

-   OLS gives equal weights to observations when minimizing RSS. The cases with larger disturbances have more "pull" than other observations.
-   A more serious problem is that the standard error estimate will be biased. This will render hypothesis testing, coefficient significance, and CI less meaningful.
-   Either transform the data or use weighted least squares.


### Collinearity

-   If a predictor is nearly the linear combination of other predictors, \(X^TX\) is close to singular.
-   Recall that \(COV(\hat{\beta}) = \sigma^2(X^TX)^{-1}\). Collinearity will cause this to be large.
-   Variance Inflation Factors: \(1 / (1 - R_j^2)\), \(R_j^2\) is the R square of regress \(X_j\) on other predictors. \(Var(\hat \beta^j) = VIF * Var(\hat \beta^j(0)\), where \(\beta_j(0)\) is the estimate of the one predictor model


## Regularization

[Link](https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-ridge-lasso-regression-python/)


### Usefulness

-   Can create parsimonious models in the presence of a large number of features. Too many features might cause overfitting and/or computational issue.
-   Size of coefficients increase exponentially with increase in model complexity (# of features). If this happens, model is very sensitive to small difference in features and causes overfitting.


### Ridge regression

-   L2 regularization. Adds a penalty term proportional to the sum of square of coefficients.
-   An appropriate


### Lasso regression

-   L1 regularization. Adds a penalty proportional to the sum of absolute values of coefficients.


## Fixed effects vs. Random effects


### Fixed effects


<a id="org3c87a81"></a>

# Frequent Interview Questions

[Link](https://www.springboard.com/blog/machine-learning-interview-questions/)


## Q1: Trade-off between bias and variance

-   Bias is caused due to erroneous or overly simplistic assumptions in the model. Will cause underfitting, i.e. can't accurately capture the data's pattern
-   Variance is error due to too much model complexity. Will cause overfitting, i.e., fitting the noise rather than data, won't generalize well in test set
-   strike a balance between these two by controlling the model complexity


## Q2: Difference between supervised and unsupervised learning


## Q3: Difference between kNN and k-means

-   kNN is used in supervised learning. k-means is unsupervised.
-   "k" has different meanings. In kNN, it stands for number of neighbors. In k-means, it's number of clusters.


## Q4: How ROC(receiver operating characteristics) curve works

-   Plot true positive rate (sensitivity) against false positive rate (fall-out)
-   Represents the trade-off between TPR and FPR. Higher TPR usually means it tends to make FP occurs more easily.


## Q5: Define precision and recall

-   precision (TPR): percentage of TP out of # of positive cases in data, TP / P
-   recall: percentage of TP out of # of predicted positive cases
-   false positive rate (FPR): FP / N
-   true negative rate (TNR): TN / N
-   false negative rate (FNR): FN / P


## Q6: Bayes theorem

-   Bayes’ Theorem is the basis behind a branch of machine learning that most notably includes the Naive Bayes classifier.


## Q7: Why is "Naive" Bayes naive?

-   Naive Bayes has a strong assumption: the conditional(on class label) distributions of features are independent to each other, i.e. the conditional probability is the product of individual probabilities
-   Has far fewer parameters than LDA and QDA. Works well even when p is very large (better than LDA)


## Q8: The difference between L1 and L2 regularization


## Q9: Favorite algorithm and reason


## Q10: Type I and Type II errors

-   Type I: False positive. Means claiming something has happened when it hasn’t.
-   Type II: False negative. Claiming nothing happened when it has.


## Q11: What is Fourier Transform?


## Q12: Difference between probability and likelihood

-   Different view point. Likelihood treats parameters as variables and data points as constants.


## Q13: Deep Learning


## Q14: Difference between generative and discriminative model

-   Generative models try to model the full joint distribution of inputs and outputs. Estimate conditional distribution of inputs conditional on outputs, then plug into Bayes rule to get conditional probability of output based on input.
-   Discriminative models model only the conditional probability of outputs based on inputs. Generally out-performs Generative models in classification tasks.


## Q15: Cross-validation techniques for time series


## Q16: How is decision tree pruned

-   Reduced error pruning: Starting from the leaves, replace each node with majority vote, if the prediction accuracy is not affected then change is kept.
-   Cost complexity pruning: Generate a series of trees by replacing node into a single leaf, each time selects the subtree that results in the smallest (increase in error rate / decrease in # of leaves). The select the tree in this series with the best train/CV accuracy.


## Q17: Deferent model performance statistics


## Q18: F1 score and how to use it

-   F1 score is the harmonic mean of precision and recall. It is valuable when true negative rate is not important.


## Q19: How to handle imbalanced datasets?

-   collect more data
-   use different performance metric: precision, recall, confusion matrix
-   resampling: SMOTE, upsampling, downsampling.
-   use ROC to determine if the model is good.
-   tweak threshold
-   try tree-based models
-   others: [link](https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/)


## Q20: When should you use classification over regression?


## Q21: Name an example where ensemble techniques might be useful.


## Q22: How do you ensure you’re not overfitting with a model?


## Q23: What evaluation approaches would you work to gauge the effectiveness of a machine learning model?


## Q24: How would you evaluate a logistic regression model?


## Q25: What’s the "kernel trick" and how is it useful?


<a id="org59ed870"></a>

# Hadoop & Spark


<a id="orgda98359"></a>

# A/B Testing & Hypothesis Testing

[Reference article](https://conversionxl.com/blog/ab-testing-statistics/)


## What is conversion rate?

-   For a website, conversion rate is the percentage of visitors that do a thing that converts to profit. Example: amazon.com, purchase rate.
-   Exit rate: percentage of visitors leave the site.
-   Drop-off rate: percentage of visitors stay but don't convert.


## What are A/A testing and A/B testing?

-   **A/A testing**: compare the conversion rates of two identical websites. If significant difference is observed, need to investigate cause.
-   **A/B testing**: compared the control version and a modified version of a website. See if there is significant increase in conversion rate.


## How to interpret p-values

-   P-value does not tell us the probability that B is better than A.
-   Similarly, it doesn’t tell us the probability that we will make a mistake in selective B over A.
-   P-value is the probability of seeing a result or more extreme given that the null hypothesis is true. Or, "How surprising is that result when null is true?"


## Significance and Power


### Statistical Significance

-   It is the probability of seeing an effect when none exists, aka. falsely rejecting null, aka. false positive rate, aka. Type I error.
-   To say a result has statistical significance is to say it's very unlikely to occur given the null hypothesis.
-   Significance level (alpha) is a defined level of how big the Type I error can be acceptable.
-   P-value is the probability of obtaining a result at least as extreme given null is true. If it's smaller than alpha, then we have statistical significance.
-   5% is the standard significance level. Can be lower depending on the subject.


### Statistical Power

-   It is the probability of seeing an effect when there is actually an effect, aka. true positive rate, aka. 1 - Type II error.
-   **Type II error**: Claiming there's no effect when there is. So a test with high power has low Type II error.
-   80% is the standard level for statistical power.


### Four levels to be set in A/B testing

-   **Effect size**: Percentage increase in conversion rate.
-   **Sample size (N)**
-   **Significance level (alpha)**
-   **Statistical power**


## Confidence Interval

-   The range corresponds to the margin of error you are willing to accept.


## Multiple Comparisons Problem

This refers to the problem of using the same significance level for testing multiple hypotheses at once as testing a single hypothesis. See [this wiki-page](https://en.wikipedia.org/wiki/Multiple_comparisons_problem).


### Key Concepts

-   **Family-wise error rate(FWER)**: The probability of having at least one false positive out of all tests. This is smaller than the sum of all FPRs(significance levels).


### Countermeasure

-   **Bonferroni Correction**: Say we are testing \(m\) hypotheses. Set the significance level to \(\alpha/m\) instead of \(\alpha\). This is very conservative in that it makes sure the FWER is smaller than the upper bound (sum of \(m\) levels). So it's hard for a test to be significant using this method.
-   **Holm-Bonferroni Method**: Sort the p-values by ascending order. Find the first \(p_{(k)}\) such that \(p_{(k)} > \alpha / (m - k + 1)\). Then reject only the first \(k - 1\) tests. This is less conservative than Bonferroni correction and still contains the FWER under significance level.

