# EY Data Challenge 2022 Level 2 : Local Frog Discovery Tool

## Notebook

1. **training_notebook.ipynb** is the jupyter notebook that contains the source code for training a Species Distribution Model (SDM).
2. **prediction_notebook.ipynb** contains the source code for test prediction

## To pull this repo & install the dependencies Using Conda

```
git clone https://github.com/chong915/2022DSC.git
```
```
conda env create -n <env_name> -f 2022DSC/environment.yaml
```
```
conda activate <env_name>
```

## To execute `train.py` from CLI
train.py is the source code for training the SDM model. **MLflow** is also used for managing the trained models.

```
python DSC2022/train.py <n_iter> <cv>
```

1. **n_iter (Default - 10)** : Number of parameters settings that are sample in randomized search cross validation
2. **cv (Default - 5)** : Determines how many folds of cross validation

## To view MLFlow UI
```
mlflow ui
```

