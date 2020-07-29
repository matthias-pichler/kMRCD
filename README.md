# kMRCD

Minimal working example of the kernel Minimum Regularized Covariance Determinant estimator.
Iwein Vranckx and Joachim Schreurs.

## Abstract

The minimum regularized covariance determinant (MRCD) is a robust estimator for multivariate location and scatter, which detects outliers by fitting a robust covariance matrix to the data. The MRCD assumes that the observations are elliptically distributed. However, this property does not always apply to modern datasets. Together with the time criticality of industrial processing, small $n$, large $p$ problems pose a challenging problem for any data analytics procedure. 
Both shortcomings are solved with the proposed kernel Minimum Regularized Covariance Determinant estimator, where we exploit the kernel trick to speed-up computations. More specifically, the MRCD location and scatter matrix estimate are computed in a kernel induced feature space, where regularization ensures that the covariance matrix is well-conditioned, invertible and defined for any dataset dimension. Simulations show the computational reduction and correct outlier detection capability of the proposed method, whereas  experiments on real-life data illustrate its applicability on industrial applications. 


