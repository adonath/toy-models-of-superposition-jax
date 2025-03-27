# Notes

## Questions / Ideas

- The matrix $W^TW$ is positive semi-definite and can be interpreted as a covariance matrix. It thus represents scaling and rotations in n-dimensional spaces.  `x' = ReLU(W^TW x + b)`
- The study only considered sparsity, but what if the input data forms a lower dimensional (sub) manifold? Repeat experiments with a real 2d structure embedded in a 5d space.


### The Gaussian Case
- Repeat the same experiments using an nd-Gaussian distribution
- Hypothesis: feature sparsity and importance are the same concept then.
- Compute variance vs sparseness. Sparsness forces dimensions to zero, which reduces their variance along the given axis. Make a smooth version, by transitioning the variance to zero
- How does the Gaussian relate to MSE...?
- The autoencoder (AE) "wants" to learn the identity, because, well, that's what AE are supposed to do.
- Now W^TW represent rotations in nd. What are the low rank approximations to the identity?
- SVD of the identity...
- Which role does the ReLU play in the Gaussian scenario? Does it cut off the small eigen-values?
-
