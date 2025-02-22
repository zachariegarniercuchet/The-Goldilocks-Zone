# Neural Network Initialization: Exploring the Goldilocks Zone

This tutorial notebook explores and reproduces key findings from recent research on neural network initialization, specifically focusing on the concept of the "Goldilocks zone"—an optimal region for network initialization that promotes better training dynamics and performance.

## Overview
After examining geometric considerations in modern initialization schemes ([Glorot et al., 2010](https://www.semanticscholar.org/paper/Understanding-the-difficulty-of-training-deep-Glorot-Bengio/ea9d2a2b4ce11aaf85136840c65f3bc9c03ab649), [He et al., 2015](https://arxiv.org/abs/1502.01852)), this notebook implements three main experiments based on the seminal work by [Fort and Scherlis (2018)](https://arxiv.org/abs/1807.02581), with additional insights from [Vysogorets et al., 2024](https://arxiv.org/abs/2402.03579). Using a fully connected neural network trained on MNIST, we investigate:

1. **Curvature Statistics**: Analyzing how the fraction of positive curvature evolves with initialization radius, demonstrating the existence of an optimal zone where curvature properties are maximized.

2. **Loss-Curvature Correlation**: Examining the relationship between initialization loss and positive curvature directions, confirming strong correlations that support theoretical predictions.

3. **Validation Accuracy Analysis**: Studying how training on different hyperplanes affects model performance, particularly focusing on the impact of initialization radius.

## Implementation Details
The experiments utilize a fully connected neural network with an architecture of `[28×28, 200, 200, 10]`, using ReLU activations and Xavier initialization with controllable scaling.

Key techniques include:
- **Dimensionality reduction** inspired by [Li et al., 2018](https://arxiv.org/abs/1712.09913).
- **Hessian approximation** methods following [Yao et al., 2018](https://proceedings.neurips.cc/paper_files/paper/2018/file/102f0bb6efb3a6128a3c750dd16729be-Paper.pdf) and [Yao et al., 2021](https://ojs.aaai.org/index.php/AAAI/article/view/17275).
- **Exploration of low-dimensional subspaces**, leveraging randomly generated orthogonal bases to efficiently study curvature properties and training dynamics.

## Key Findings
Our results largely align with those reported by [Fort and Scherlis (2018)](https://arxiv.org/abs/1807.02581), confirming the existence of the Goldilocks zone, where initialization leads to optimal curvature properties. We observe:
- **A clear correlation** between initialization loss and positive curvature.
- **A strong dependency** between validation accuracy and the initialization weight norm.
- **Consistent trends** in curvature statistics that match theoretical predictions, supporting the hypothesis that proper initialization fosters better learning dynamics.

These insights contribute to a deeper understanding of neural network initialization and its impact on training efficiency and generalization.

---
## References

[1] Stanislav Fort and Adam Scherlis. The goldilocks zone: Towards better understanding of neural network loss landscapes. *AAAI Conference on Artificial Intelligence*, 2018. doi: 10.1609/aaai.v33i01.33013574.

[2] Xavier Glorot and Yoshua Bengio. Understanding the difficulty of training deep feedforward neural networks, 2010. URL: [https://www.semanticscholar.org/paper/Understanding-the-difficulty-of-training-deep-Glorot-Bengio/ea9d2a2b4ce11aaf85136840c65f3bc9c03ab649](https://www.semanticscholar.org/paper/Understanding-the-difficulty-of-training-deep-Glorot-Bengio/ea9d2a2b4ce11aaf85136840c65f3bc9c03ab649).

[3] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification, 2 2015. URL: [https://arxiv.org/abs/1502.01852](https://arxiv.org/abs/1502.01852).

[4] Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer, and Tom Goldstein. Visualizing the loss landscape of neural nets. *Advances in Neural Information Processing Systems*, 31, 2018.

[5] Artem Vysogorets, Anna Dawid, and Julia Kempe. Deconstructing the goldilocks zone of neural network initialization. *International Conference on Machine Learning*, 2024. doi: 10.48550/arXiv.2402.03579.

[6] Zhewei Yao, Amir Gholami, Qi Lei, Kurt Keutzer, and Michael W Mahoney. Hessian-based analysis of large batch training and robustness to adversaries. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, editors, *Advances in Neural Information Processing Systems*, volume 31. Curran Associates, Inc., 2018. URL: [https://proceedings.neurips.cc/paper_files/paper/2018/file/102f0bb6efb3a6128a3c750dd16729be-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2018/file/102f0bb6efb3a6128a3c750dd16729be-Paper.pdf).

[7] Zhewei Yao, Amir Gholami, Sheng Shen, Mustafa Mustafa, Kurt Keutzer, and Michael Mahoney. Adahessian: An adaptive second order optimizer for machine learning. *Proceedings of the AAAI Conference on Artificial Intelligence*, 35(12):10665–10673, May 2021. doi: 10.1609/aaai.v35i12.17275. URL: [https://ojs.aaai.org/index.php/AAAI/article/view/17275](https://ojs.aaai.org/index.php/AAAI/article/view/17275).



