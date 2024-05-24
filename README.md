<script type="text/javascript"
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>

# Trajectory Volatility for Out-of-Distribution Detection in Mathematical Reasoning

The overview and official implementation of **TV score** used in **OOD Detection in Mathematical Reasoning**. 

Details are shown in [our paper](https://arxiv.org/abs/2405.14039).



## Overview

### Why Trajectory as the measure?

#### a. Disadvantages of input/output embedding space:

* *Input Space*: Low distinction between different domains

* *Output Space*: compressed high-density search space -> pattern collapse

<!-- ![Input/Output Embedding Comparison](Assets/io_embedding.png){:height="50%" width="50%"} -->

<div align=center>
<img src="ASSETS/io_embedding.png" width="80%" height="80%" />
</div>



#### b. Advantages of input->output embedding shift trajectory:

* Constraints on trajectory endpoints in mathematical reasoning allow for a greater likelihood of variation in trajectory volatility under different samples. 

<!-- ![Trajectory Comparison](Assets/trajectory.png) -->
<div align=center>
<img src="ASSETS/trajectory.png" width="80%" height="80%" />
</div>

### What is TV score?

A trajectory-based algorithm to detect OOD samples in mathematical reasoning scenarios.

**Algorithm Pipeline:**

We denote $\boldsymbol{y_l}$ as the embedding of $l$-th layer, $\mathcal{G}_l = \mathcal{N}(\boldsymbol{\mu}_l, \boldsymbol{\Sigma}_l)$ as ID Gaussian distribution of $l$-th layer


* *Step 1*: Mahalanobis Distance Mapping 

$$
\mathcal{N}(\boldsymbol{\mu}_l^{(k)}, \boldsymbol{\Sigma}_l^{(k)}) = \mathcal{N}(\sum_{i=0}^k (-1)^{k+i} \mathrm{C}_k^i \boldsymbol{\mu}_{l+k}, \sum_{i=0}^k \mathrm{C}_k^i \boldsymbol{\Sigma}_{l+k}),
\\
\boldsymbol{y_l^{(k)}} = \sum_{i=0}^k (-1)^{k+i} \mathrm{C}_k^i \boldsymbol{y}_{l+k}
\\
f^{(k)}(\boldsymbol{y}_l) = \left[\boldsymbol{y_l^{(k)}} - \boldsymbol{\mu_l^{(k)}}\right]^{\top} \left[ \Sigma_l^{(k)} \right]^{-1} \left[\boldsymbol{y_l^{(k)}} - {\boldsymbol \mu_l^{(k)}}\right]
$$


* *Step 2*:  Average of Absolute Value Difference

$$
\text{TV score} := \frac{1}{L-k-1} \sum_{l=1}^{L-k-1}  \left| f^{(k)}(\boldsymbol{y}_l) - f^{(k)}(\boldsymbol{y}_{l-1}) \right|
$$


TV score w/o Differential Smoothing when $k = 0$, w/ Differential Smoothing when $k>0$.





