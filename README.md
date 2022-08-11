# AraSpell (Arabic Spelling Correction with Transformer) 
This work introduces AraSpell (Arabic Spell Correction with transformer) trained on more than 6.5 Million sentences crawled from Wikipedia

# Model 
We implemented the transformer model as described [here](https://arxiv.org/abs/1706.03762?context=cs), the image below shows the model architecture.

![5625](https://user-images.githubusercontent.com/61272193/183622776-894b3701-6ab3-4749-80c3-013638fb69ac.jpg)

# Datasets
The data set used is Wikipedia dataset and can be found [here](https://www.kaggle.com/datasets/z3rocool/arabic-wikipedia-dump-2021) on Kaggle, and the table below are the training and testing sets we have used after performing the proper transformations.

| Distortion/Noise Ratio      | Training | Testing |
| ----------- | ----------- | ----------- |
| 0.05      | [train_05.csv](https://drive.google.com/file/d/1-3msyooepqJOFbsaEkK0r7wyquxfJvfP/view?usp=sharing)       | [test_05.csv](https://drive.google.com/file/d/1rATupv9LL6dSkdXwJd_-JbMLonMPLL5p/view?usp=sharing) |
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

There are 2 different ways you can train on your data:
1. Your data is ready: 
 * Your data on a form of 3 CSV files train, test, and dev and each CSV file looks like the following
```
clean, distorted
clean line, distorted line 
clean line, distorted line
```
* Change the valid characters in the constants.py file according to your language
* Start the training using the following command after changing the value to fit your needs
```bash
python train.py --epochs number_of_epochs --n_gpus max_sent_length --train_path path/to/train.csv \
      --test_path path/to/train.csv --max_len max_sent_length --distortion_ratio distortion_ratio
```
2. Process the data using our processing pipeline
* format the data to be in the following structuer
```
data
│   file1.txt
│   file2.txt
│   file3.txt
```
* Change the valid characters in the constants.py file according to your language
* run the below command to process the data
```bash
python process_data.py
```
* Split your data into train, test and dev CSV files
* Start the training using the following command after changing the value to fit your needs
```bash
python train.py --epochs number_of_epochs --n_gpus max_sent_length --train_path path/to/train.csv \
      --test_path path/to/train.csv --max_len max_sent_length --distortion_ratio distortion_ratio
```
