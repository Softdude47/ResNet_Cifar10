
import os
import argparse
import numpy as np
from mas_lib.nn.conv.resnet import ResNet
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
from mas_lib.callbacks.epochcheckpoint import EpochCheckpoint
from mas_lib.callbacks.trainingmonitor import TrainingMonitor
from mas_lib.callbacks.polynomialdecay import PolynomialDecay
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# configuring commandline arguments
ap = argparse.ArgumentParser()
ap.add_argument(
    "-c",
    "--checkpoint",
    required=True,
    help="path to checkpoint model training"
)
ap.add_argument(
    "-m",
    "--model",
    help="path to load specific model checkpoint"
)
ap.add_argument(
    "-start",
    "--start-at",
    type=int,
    default=0,
    help="epoch to restart  trainig from"
)
args = vars(ap.parse_args())

# load cifar10 datasets
(xtrain, ytrain), (xtest, ytest) = cifar10.load_data()

# some parameter of model train function
BS = 128
STEPS = len(xtrain) // BS
STEPS = (STEPS + 1) if (STEPS * BS) < len(xtrain) else STEPS
NUM_EPOCHS = 100

# path to store model metrics during training
FIG_PATH = os.path.sep.join([os.getcwd(), "outputs", "resnet-cifar10.png"])
JSON_PATH = os.path.sep.join([os.getcwd(), "outputs", "resnet-cifar10.json"])

# convert cifar10 images to float data type
xtrain = xtrain.astype("float")
xtest = xtest.astype("float")

# perform mean subtraction on cifar10 images
mean = np.mean(xtrain, axis=(0,1,2))
xtrain -= mean
xtest -= mean

# one-hot encode cifar10 dataset labels
lb = LabelBinarizer()
ytrain = lb.fit_transform(ytrain)
ytest = lb.transform(ytest)


# image augmentation function
aug = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# model trainig callbacks
callbacks = [
    EpochCheckpoint(args["checkpoint"], interval=5),
    TrainingMonitor(FIG_PATH, JSON_PATH, args["start_at"]),
    PolynomialDecay(0.1, 1)
]

if args["model"] is None:
    opt = SGD(learning_rate=0.1, momentum=0.9)
    model = ResNet.build(32, 32, 3, 10, (9, 9, 9), (64, 64, 128, 256), reg=5e-4)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy'])
    
else:
    model = load_model(args["model"])
    print(f"old learning rate: {K.get_value(model.optimizer.learning_rate)}")
    K.set_value(model.optimizer.learning_rate, 1e-4)
    print(f"new learning rate: {K.get_value(model.optimizer.learning_rate)}")
    
    
model.fit(
    aug.flow(xtrain, ytrain, batch_size=BS),
    validation_data=(xtest, ytest),
    max_queue_size=BS * 2,
    steps_per_epoch=STEPS,
    callbacks=callbacks,
    epochs=NUM_EPOCHS,
    verbose=1,
)