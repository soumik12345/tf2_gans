# TF2 GANs

Implementation of GANs in TensorFlow 2.x. Ensure you have the dependencies installed
if you want to train these GANs.

```shell
$ pip install -r requirements.txt 
```


## Training GauGAN with [Facades](https://cmp.felk.cvut.cz/~tylecr1/facade/) dataset

Blog post: [GauGAN for conditional image generation](https://keras.io/examples/generative/gaugan/)

```shell
$ gdown https://drive.google.com/uc?id=1q4FEjQg1YSb4mPx2VdxL7LXKYu3voTMj
$ unzip -q facades_data.zip
$ python train_facades.py --facades_configs ./configs/facades.py
```

You can find the pre-trained checkpoints [here](https://github.com/soumik12345/tf2_gans/releases/tag/v0.2) and
the [training logs on Weights and Biases](https://wandb.ai/tf2_gans/GauGAN/runs/1vvw7cpw). You are also welcome to
checkout [this inference notebook](https://github.com/soumik12345/tf2_gans/blob/gaugan/notebooks/gaugan_facades_inference.ipynb) (runnable
on Colab) to play with the model. 

## [WIP] Training GauGAN with [Coco-Stuff 10k](https://github.com/nightrome/cocostuff10k)

```shell
$ wget -q http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/cocostuff-10k-v1.1.zip
$ unzip -q cocostuff-10k-v1.1.zip -d cocostuff10k_data
$ python train_cocostuff10k.py --cocostuff10k_configs ./configs/cocostuff10k.py
```

## Acknowledgements

* [ML-GDE](https://developers.google.com/programs/experts/) program that provided GCP credits for the execution of the experiments. 
