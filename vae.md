# Inspiration
- [Pytorch tutorial](https://github.com/pytorch/examples/blob/main/vae/main.py)
- [Stanford class](https://github.com/scpd-proed/XCS236-PS2/blob/main/src/submission/models/vae.py)
- [Clone of simo VQgan](https://github.com/cloneofsimo/vqgan-training/blob/main/ae.py)
- [Paul nvae](https://github.com/pauldb89/ml/blob/master/nvae/model.py)

- [NYU vae](https://atcold.github.io/NYU-DLSP21/)
- [Sebastian VAE](https://sebastianraschka.com/blog/2021/dl-course.html#l17-variational-autoencoders)

# Notes
- dataloader
  - add pairs?
  - use bernoulli noise
- example
  - argparse
  - utils.data.Dataloader(download=True)
  - Vae class
    - exp(0.5 * logvar)!
    - kl uses logvar, might as well use it, backprop will work better 
- loss reduction is sum which is weird
- easier to plot if z_dim == 2 but we can use tsne like nyu does
- sebi has alexnet in pt and other stuff, may be worth doing it
- [traverse latent space](https://medium.com/@rekalantar/variational-auto-encoder-vae-pytorch-tutorial-dce2d2fe0f5f)
