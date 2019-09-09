from train import Train
from options import *

if __name__ == "__main__":
    args = get_arguments()
    t = Train(args)
    train_steps, val_steps = t.get_training_steps()
    saved_epochs = t.get_global_step() // train_steps
    epochs = 1 if args.fast else args.epochs
    for e in range(saved_epochs, epochs):
        t.train(train_steps, True)
        t.train(val_steps, False)