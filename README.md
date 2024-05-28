# AI-Face-FairnessBench
This repository is the official implementation of our paper "AI-Face: A Million-Scale Demographically Annotated AI-Generated Face Dataset and Fairness Benchmark"
## License
The AI-Face Dataset is licensed under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode)
## Download
If you would like to access the AI-Face Dataset, please download and sign the [EULA](https://indiana-my.sharepoint.com/:b:/g/personal/sant_iu_edu/ETSPPGORgnVNqgTWaHjhMQkBe5nd2eMHRBN74JGa2R1n8g?e=DeK5cr). Please upload the signed EULA to the [Google Form](https://forms.gle/Wci1hsZCz6Rgnvw57) and fill the required details. Once the form is approved, the download link will be sent to you.
If you have any questions, please send an email to lin1785@purdue.edu, hu968@purdue.edu

## 1. Installation

You can run the following script to configure the necessary environment:

```
cd AI-Face-FairnessBench
conda create -n FairnessBench python=3.9.0
conda activate FairnessBench
pip install -r requirements.txt
```
## 2. Dataset Preparation
After getting our AI-Face dataset, put train.csv and test.csv under  [`./dataset`](./dataset).

## 3. Load Pretrained Weights
Before running the training code, make sure you load the pre-trained weights. We provide pre-trained weights under [`./training/pretrained`](./training/pretrained). You can also download *Xception* model trained on ImageNet (through this [link](http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth)) or use your own pretrained *Xception*.

## 4. Train
To run the training code, you should first go to the [`./training/`](./training/) folder, then run [`train_test.py`](training/train_test.py):

```
cd training

python train.py 
```

You can adjust the parameters in [`train_test.py`](training/train_test.py) to specify the parameters, *e.g.,* training dataset, batchsize, learnig rate, *etc*.

`--lr`: learning rate, default is 0.0005. 

`--train_batchsize`: batchsize for training, default is 128.

`--test_batchsize`: batchsize for testing, default is 32.

` --datapath`: /path/to/[`./dataset`](./dataset).

`--model`: detector name, default is 'xception'.

If you use the AI-face dataset in your research, please cite our paper as:
