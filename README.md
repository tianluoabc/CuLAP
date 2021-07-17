<img src="./static/logo2.png" alt="drawing" width="200"/>

## CUDA Hungarian Single Device Solver

Implementation of **O(n^3) Alternating Tree Variant** of Hungarian Algorithm on NVIDIA CUDA-enabled GPU.

This implementation solves a batch of **k** **Linear Assignment Problems (LAP)**, each with *nxn* matrix of single floating point cost values. At optimality, the algorithm produces an assignment with **minimum** cost. Further, The API calls can be used to query optimal primal and dual costs, optimal assignment vector, and optimal row/column dual vectors for each subproblem in the batch.

- [Documentation](In-Development)
- [Setup and Installation](void)

**This library is licensed under Apache License 2.0. Please cite our paper, if this library helps you in your research.**

- Harvard citation style

  Date, K. and Nagi, R., 2016. GPU-accelerated Hungarian algorithms for the Linear Assignment Problem. Parallel Computing, 57, pp.52-72.

- BibTeX Citation block to be used in LaTeX bibliography file:

```
@article{date2016gpu,
  title={GPU-accelerated Hungarian algorithms for the Linear Assignment Problem},
  author={Date, Ketan and Nagi, Rakesh},
  journal={Parallel Computing},
  volume={57},
  pages={52--72},
  year={2016},
  publisher={Elsevier}
}
```

- The paper is available online on [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S016781911630045X).
