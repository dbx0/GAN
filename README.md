# GAN
A simple GAN using the CIFAR-10 dataset to generate new images

*CIFAR-10*
*https://www.cs.toronto.edu/~kriz/cifar.html*

## Requirements
* Linux or Mac
* Python 3.6
* Pip

## Instalation
Get into the project folder

Setting up the virtual env
```bash
pip install virtualenv
virtualenv venv_gan
source venv_gan/bin/activate
```

Installing dependencies
```bash
pip install -r requirements.txt
```

Downloading dataset 
```bash
wget -qO- http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz | tar xvz -C data/
```

## Example

**Input with real images**
<img src="https://i.imgur.com/BrtvDmt.png" width="500">

**Output generated with 25 epochs**
<img src="https://i.imgur.com/vbOhdSo.png" width="500">

