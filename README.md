# DCGAN CelebA
A DCGAN that learns to generate celebrity faces based off of the CelebA dataset.

 ![Training plots](training-plots.gif)

## Install

```
python -m venv env

.\env\Scripts\activate

pip install -r requirements.txt
```

## Dataset

Download [CelebA dataset](https://www.kaggle.com/jessicali9530/celeba-dataset).

Extract directory `img_align_celeba` to working directory.

Run `loadDataset(True)` from `data.py` to prepare training data.

## Reference

https://machinelearningmastery.com/generative_adversarial_networks/