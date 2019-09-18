from train import Train
from options import *
from tqdm import tqdm
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

if __name__ == "__main__":
    args = get_arguments()
    t = Train(args)
    train_steps, val_steps = t.get_training_steps()
    saved_epochs = t.get_global_step() // train_steps
    epochs = 1 if args.fast else args.epochs
    for e in tqdm(range(saved_epochs, epochs), dynamic_ncols=True, desc='EPOCH: '):
        t.train(train_steps, True)
        t.train(val_steps, False)
    t.close_files()