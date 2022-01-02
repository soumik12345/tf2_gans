# TF2 GANs

Implementation of GANs in TensorFlow 2.x. Ensure you have the dependencies installed
if you want to train these GANs.

```shell
$ pip install -r requirements.txt 
```


## Training GauGAN with [Facades](https://cmp.felk.cvut.cz/~tylecr1/facade/) dataset

```shell
$ gdown https://drive.google.com/uc?id=1q4FEjQg1YSb4mPx2VdxL7LXKYu3voTMj
$ unzip -q facades_data.zip
$ python train_facades.py --facades_configs ./configs/facades.py
```