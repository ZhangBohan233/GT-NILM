from nilmtk.api import API
import warnings

warnings.filterwarnings("ignore")
from nilmtk.disaggregate import DM_GATE2
import nilmtk.utils as utils
from dataset_info import *
import torch

USE_GPU = True
device = torch.device("cuda:0" if USE_GPU and torch.cuda.is_available() else "cpu")
print(torch.__version__, device)

torch.set_float32_matmul_precision('high')

USING_DATASET = 'redd'
MAIN_POWER = 'apparent' if USING_DATASET == 'redd' else 'active'
APPLIANCES = ['dish washer', 'fridge', 'microwave', 'washing machine'] \
    if USING_DATASET == 'redd' \
    else ['dish washer', 'fridge', 'kettle', 'microwave', 'washing machine']

e = {
    # Specify power type, sample rate and disaggregated appliance
    'power': {
        'mains_train': [MAIN_POWER],  # problem: ukdale active, redd apparent
        'mains_test': [MAIN_POWER],
        'appliance': ['active']
    },
    'sample_rate': 3 if USING_DATASET == 'redd' else 6,
    # 'appliances': ['washing machine'],
    'app_meta': utils.GENERAL_APP_META,
    'appliances': APPLIANCES,
    # Universally no pre-training
    'pre_trained': False,
    # Specify algorithm hyper-parameters
    'save_note': '',
    'methods': {
        "DM_GATE2": DM_GATE2(
            {
                'n_epochs': 5 if USING_DATASET == "redd" else 5,
                'batch_size': 128,
                # size of the sliding window
                'sequence_length': 720,
                # step-size of the sliding window when training
                'overlapping_step': 1,
                # name suffix of the state-dict file.
                'note': USING_DATASET + '',
                # conduct test only, no training
                'test_only': False,
                # training or fine-tuning
                'fine_tune': False,
                'lr': 3e-5,
                # ddpm or ddim
                "sampler": "ddim",
                'patience': 5 if USING_DATASET == "redd" else 3,
                "app_meta": utils.GENERAL_APP_META,
                # whether to train the DM on active windows only
                'filter_train': False
            }
        )
    },
    # Specify train and test data
    'train': {
        'datasets':
            {'redd': REDD_TRAIN_STD} if USING_DATASET == "redd" else {'ukdale': UKDALE_TRAIN_STD}

    },
    'test': {
        'datasets':
            {'redd': REDD_TEST_STD} if USING_DATASET == "redd" else {'ukdale': UKDALE_TEST_STD}
        ,
        # Specify evaluation metrics
        'metrics': ['accuracy', 'f1score', 'mae', 'sae', 'wssim']
    }
}

if __name__ == '__main__':
    API(e)
