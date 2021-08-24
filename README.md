# pytorch-adversarial-training
A PyTorch implementation of adversarial training.

## Benchmarks

1. Basic and adversarial training on CIFAR-10 dataset.

    The image shape of CIFAR-10 dataset is 32x32, which is much smaller than the image shape used by Resnet, so we replace the first 7x7 convolution layer with 3x3 convolution layer with stride 1 and padding 1, and we remove the first MaxPooling layer. (Following SimCLR)

    We use modified ResNet18 as backbone model to run the following benchmarks.

    | **Training mode** | **Attack**                                     | **Batch size**    | **Learning rate** | **Scheduler**                    | **Clean dataset acc.** | **Adversarial acc.** |
    |:-----------------:|:----------------------------------------------:|:-----------------:|:-----------------:|:--------------------------------:|:----------------------:|:--------------------:|
    | Clean train       | LinfPGD($\epsilon$=8/255, step=2/255, iters=7) | 64*2(Distributed) | 1e-1              | MultiStep([100, 150], gamma=0.1) | 0.9483                 | 0.0125               |
    | Adversarial train | LinfPGD($\epsilon$=8/255, step=2/255, iters=7) | 64*2(Distributed) | 1e-1              | MultiStep([100, 150], gamma=0.1) | 0.8438                 | 0.4750               |
    | Clean train       | L2PGD($\epsilon$=1.0, step=0.2, iters=10)      | 64*2(Distributed) | 1e-1              | MultiStep([100, 150], gamma=0.1) | 0.9500                 | 0.0018               |
    | Adversarial train | L2PGD($\epsilon$=1.0, step=0.2, iters=10)      | 64*2(Distributed) | 1e-1              | MultiStep([100, 150], gamma=0.1) | 0.8082                 | 0.4293               |
    |                   |                                                |                   |                   |                                  |                        |                      |
