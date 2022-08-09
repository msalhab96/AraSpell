# AraSpell (Arabic Spelling Correction with Transformer) 
This work introduces AraSpell (Arabic Spell Correction with transformer) trained on more than 6.5 Million sentences crawled from Wikipedia

# Model 
We implemented the transformer model as described [here](https://arxiv.org/abs/1706.03762?context=cs)

# Datasets
The data set used is Wikipedia dataset and can be found [here](https://www.kaggle.com/datasets/z3rocool/arabic-wikipedia-dump-2021) on Kaggle, and the table below are the training and testing sets we have used after performing the proper transformations.

| Distortion/Noise Ratio      | Training | Testing |
| ----------- | ----------- | ----------- |
| 0.05      | [train_05.csv]()       | [test_05.csv]() |
| 0.1   | [train_1.csv]()        | [test_1.csv]() |
| 0.15   | [train_15.csv]()        | [test_15.csv]() |

# Setup
### Setup development environment
* create enviroment 
```bash
python -m venv env
```
* activate the enviroment
```bash
source env/bin/activate
```
* install pytorch

pick the version with the proper cuda version for your hardware from [here](https://pytorch.org/) (optional)
```bash
pip install torch --extra-index-url https://download.pytorch.org/whl/cu116
```
* install the required dependencies
```bash
pip install -r requirements.txt
```
### Setting up docker image
```bash
docker build . 
```

# Try it out
To be added

# Results
To be added

# Training on Other Languages
To be added
