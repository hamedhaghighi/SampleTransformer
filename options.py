import argparse
from ops import optimizer_factory

BATCH_SIZE = 1
DATA_DIRECTORY = '/home/oem/.tensorflow/music'
LOGDIR_ROOT = './logdir'
PRINT_EVERY = 100
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
WAVENET_PARAMS = './wavenet_params.json'
SAMPLE_SIZE = 2**15
L2_REGULARIZATION_STRENGTH = 0  
SILENCE_THRESHOLD = 1e-4 # TODO: change it to 0.3
MOMENTUM = 0.9
MAX_TO_KEEP = 3
METADATA = False



def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser(description='WaveNet example network')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='How many wav files to process at once. Default: ' + str(BATCH_SIZE) + '.')
    parser.add_argument('--lbsa', type=int, default=4,
                        help='layers of block sparse attention')
    parser.add_argument('--lsa', type=int, default=8,
                        help='layers of sparse attention')
    parser.add_argument('--lfa', type=int, default=16,
                        help='layers of full sparse attention')
    parser.add_argument('--wl', type=int, default=8,
                        help='layers of wavenet')
    parser.add_argument('--n_memory', type=int, default=8,
                        help='layers of wavenet')                
    parser.add_argument('--channel_size', type=int, default=128,
                        help='layers of full sparse attention')
    parser.add_argument('--init_kernel_size', type=int, default=32,
                        help='layers of full sparse attention')
    parser.add_argument('--q_levels', type=int, default=2**8,
                        help='layers of full sparse attention')
    parser.add_argument('--q_type', type=str, default='linear',
                        help='layers of full sparse attention')
    parser.add_argument('--scalar_input', action="store_true", default=False,
                        help='layers of full sparse attention')
    parser.add_argument('--data_dir', type=str, default=DATA_DIRECTORY,
                        help='The directory containing the VCTK corpus.')
    parser.add_argument('--logdir', type=str, default=None,
                        help='Directory in which to store the logging '
                        'information for TensorBoard. '
                        'If the model already exists, it will restore '
                        'the state and will continue training. '
                        'Cannot use with --logdir_root and --restore_from.')
    parser.add_argument('--logdir_root', type=str, default=LOGDIR_ROOT,
                        help='Root directory to place the logging '
                        'output and generated model. These are stored '
                        'under the dated subdirectory of --logdir_root. '
                        'Cannot use with --logdir.')
    parser.add_argument('--load_type', type=str, default=None,
                        help='Directory in which to restore the model from. '
                        'This creates the new model under the dated directory '
                        'in --logdir_root. '
                        'Cannot use with --logdir.')
    parser.add_argument('--print_every', type=int,
                        default=PRINT_EVERY,
                        help='How many steps to save each checkpoint after. Default: ' + str(PRINT_EVERY) + '.')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help='Number of training steps. Default: ' + str(NUM_EPOCHS) + '.')                    
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help='Learning rate for training. Default: ' + str(LEARNING_RATE) + '.')
    parser.add_argument('--sample_size', type=int, default=SAMPLE_SIZE,
                        help='Concatenate and cut audio samples to this many '
                        'samples. Default: ' + str(SAMPLE_SIZE) + '.')
    parser.add_argument('--l2_regularization_strength', type=float,
                        default=L2_REGULARIZATION_STRENGTH,
                        help='Coefficient in the L2 regularization. '
                        'Default: False')
    parser.add_argument('--silence_threshold', type=float,
                        default=SILENCE_THRESHOLD,
                        help='Volume threshold below which to trim the start '
                        'and the end from the training set samples. Default: ' + str(SILENCE_THRESHOLD) + '.')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=optimizer_factory.keys(),
                        help='Select the optimizer specified by this option. Default: adam.')
    parser.add_argument('--momentum', type=float,
                        default=MOMENTUM, help='Specify the momentum to be '
                        'used by sgd or rmsprop optimizer. Ignored by the '
                        'adam optimizer. Default: ' + str(MOMENTUM) + '.')
    parser.add_argument('--histograms', type=_str_to_bool, default=False,
                        help='Whether to store histogram summaries. Default: False')
    parser.add_argument('--gc_channels', type=int, default=None,
                        help='Number of global condition channels. Default: None. Expecting: Int')
    parser.add_argument('--max_checkpoints', type=int, default=MAX_TO_KEEP,
                        help='Maximum amount of checkpoints that will be kept alive. Default: '
                             + str(MAX_TO_KEEP) + '.')
    parser.add_argument('--fast', action="store_true", default=False)

    return parser.parse_args()