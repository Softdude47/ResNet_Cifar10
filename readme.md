# ResNet on Cifar10 dataset

## Train Model

To start training on command line and run:
```cmd
python train_cifar10.py --start-at 0 --model <path to specific model> --checkpoint <path to checkpoint model during training>
```

## Experiment

On this experiment, I used a model comprising of `9 by 3` residual module (i.e 3 9-residual-modules stacked on together) with a `L2` regularization of `5e-1`. The during training I used a `learning rate` of `0.1` and a polyinomial `learning rate` with `power = 1`.
![img](https://drive.google.com/file/d/1F1tqcdxLIKlcdnB0vTyq7N3MLxyFjaYj/view?usp=sharing "model metrics on each epochs")

After training for 100 epochs, the model attained ~93% validation accuracy and a validation loss < 0.4.
