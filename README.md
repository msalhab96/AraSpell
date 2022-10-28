# AraSpell (Arabic Spelling Correction) 
This work introduces AraSpell (Arabic Spelling Correction) trained on more than 6.9 Million sentences, the offical implemntation of [AraSpell Paper]().

**The paper would be available [here]()**

# Model 
We implemented 3 models arcthictures as described in the paper they are:
- Attentional vanilla Seq2Seq using RNN
- Attentional Seq2Seq with stacked RNN blocks
- Transformer

The image below shows all the model architectures proposed and implemented

![Capture](https://user-images.githubusercontent.com/61272193/193535699-fa42dceb-c651-4c8f-a0a8-54745af25feb.PNG)



# Datasets
The data set used is Wikipedia dataset and can be found [here](https://www.kaggle.com/datasets/z3rocool/arabic-wikipedia-dump-2021) on Kaggle, and below are the training, testing, and dev sets we have used after performing the proper transformations.

| Data      | Link |
| ----------- | ----------- |
| Train      | [Here](https://drive.google.com/file/d/1uu_Ga7MZ6sYHhxfuPBH3rVPglxmIqL0O/view?usp=sharing)       |
| Test   | [Here](https://drive.google.com/file/d/1YwrWfISPXHQDaTtf3h6K-1-8bO3kH5lB/view?usp=sharing)        |
| Dev   | [Here](https://drive.google.com/file/d/18As7vbgveFWsjt6wGqlgvjw8ax9FxJbF/view?usp=sharing)        |

To generate the mixed dataset use the below for both ```train.csv``` and ```test.csv```

```bash
cat train.csv | awk -F , '{print $2","$3}' | sed "s/clean\,distorted_0\.05/clean,distorted_mix/g" > train_mix.csv
cat train.csv | awk -F , '{if (NR>1) print $2","$4}' >> train_mix.csv
```

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
You can try it out using the jupyter notebook provided ```infer.ipynb``` and download the tokenizer and one of the published models

# Results
| Model | CER (%) on 5%  | CER (%) on 10% | WER (%) on 5%  | WER (%) on 10% |
| ---------------------- | ----------- | ----------- | ----------- | ----------- |
| Transformer_0.05 | 1.24 | 4.15 | 5.35 | 18.38 |
| Transformer_0.1 | 1.45 | 2.82 | 5.95 | 10.36 |
| Transformer_mixed | 1.11 | 2.8 | 4.8 | 10.65 |
| Transformer_varied | 1.22 | 3.16 | 5.41 | 12.35 |
| RNNB_0.05 | 1.74 | 4.45 | 7.76 | 18.8 |
| RNNB_0.1 | 1.86 | 4.01 | 7.8 | 15.41 |
| RNNB_mixed | 1.67 | 4.06 | 7.51 | 16.55 |
| RNNB_varied | 1.77 | 4.36 | 8.14 | 18.01 |
| VanillaRNN_0.05 | 1.89 | 4.99 | 8.33 | 20.67 |
| VanillaRNN_0.1 | 2.01 | 4.52 | 18.19 | 16.95 |
| VanillaRNN_mixed | 1.86 | 4.53 | 7.84 | 17.36 |
| VanillaRNN_varied | 2.01 | 4.97 | 8.76 | 19.49 |

#### Pre-trained Models
All Pre-trained models are available [here](https://drive.google.com/drive/folders/113w5z1fMBVGLFAZu2IlCrRLWTIw37PE0)

# Fine tune the published models
* Download one of the published model
* Prepare the dataset as mentioned in [here](#ready)
* run the following command to train the model

```bash
python train.py --pre_trained_path path/to/checkpoint.pt
```

# Training on Other Languages
There are 2 different ways you can train on your data:
1. Your data is ready: 
<a name="ready"></a>
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
