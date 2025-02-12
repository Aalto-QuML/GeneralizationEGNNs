## Generalization of Equivariant Graph Neural Networks 

This is the official repo for the paper [On the Generalization of Equivariant Graph Neural Networks](https://openreview.net/pdf?id=Yqj3DzIC79) (ICML 2024).

This repo consists of a minor modification of the repo [EGNN](https://github.com/vgsatorras/egnn). We consider the full EGNN model (i.e., including features/coordinates updates) on QM9 datasets. We also adopt the margin loss function (for testing) and reduced the number of training samples (we use 2K). 

To run the experiments, we choose ```--property``` (QM9 dataset), ```--lr``` (learning rate), ```--epochs```, ```--regularizer``` (none, spectral, or ours), and ```--lambda_reg``` (regularization factor), e.g.:
```
python main_qm9.py --property homo --epochs 2000 --lr 1e-3 --regularizer ours --lambda_reg 1e-7   
```

### Citation
```
@inproceedings{Karczewski2024,
  title={On the generalization of equivariant graph neural networks},
  author={Rafal Karczewski and Amauri H. Souza and Vikas Garg},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2024}
}
```



