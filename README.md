# TF2 GANs

Implementation of GANs in TensorFlow 2.x. Ensure you have the dependencies installed
if you want to train these GANs.

```shell
$ pip install -r requirements.txt 
```


## Training GauGAN with [Facades](https://cmp.felk.cvut.cz/~tylecr1/facade/) dataset

```shell
$ gdown https://drive.google.com/uc?id=1zs2RPZkLeB5QDRVQtpqUtMwepR_Qfufp
$ unzip -q facades.zip
$ python train_gaugan.py
```