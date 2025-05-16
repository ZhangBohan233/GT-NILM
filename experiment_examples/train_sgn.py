from nilmtk.api import API
import warnings

warnings.filterwarnings("ignore")
from nilmtk.disaggregate import AttentionCNN, SGN
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

# A special case for training and testing on REDD within same households
REDD_TRAIN_SEEN = {
    'path': 'mnt/redd.h5',
    'buildings': {
        1: {
            'start_time': '2011-04-19',
            'end_time': '2011-04-26'
        },
        2: {
            'start_time': '2011-04-18',
            'end_time': '2011-04-25'
        },
        3: {
            'start_time': '2011-04-17',
            'end_time': '2011-04-24'
        },
        5: {
            'start_time': '2011-04-19',
            'end_time': '2011-05-30'
        },
        # 6: {
        #     'start_time': '2011-05-22',
        #     'end_time': '2011-06-13'
        # },
    }
}

REDD_TEST_SEEN = {
    'path': 'mnt/redd.h5',
    'buildings': {
        # 1: {
        #     'start_time': '2011-04-24',
        #     'end_time': '2011-05-26'
        # },
        # 2: {
        #     'start_time': '2011-04-25',
        #     'end_time': '2011-05-22'
        # },
        3: {
            'start_time': '2011-04-24',
            'end_time': '2011-05-30'
        },
    }
}

HOUSE = 3
CSV_PATH = f"./csv_ft/redd_seen_sgn_{'+'.join(APPLIANCES)}_{HOUSE}_.csv"

e = {
    # Specify power type, sample rate and disaggregated appliance
    'power': {
        'mains_train': [MAIN_POWER],
        'mains_test': [MAIN_POWER],
        'appliance': ['active']
    },
    'sample_rate': 3 if USING_DATASET == 'redd' else 6,
    'app_meta': utils.GENERAL_APP_META,
    'appliances': APPLIANCES,
    # Universally no pre-training
    'pre_trained': False,
    # 'save_note': 'seen',  # for the special case, save the models and csvs with suffix 'seen'
    # Specify algorithm hyper-parameters
    # 'csv_path': CSV_PATH,
    'methods': {
        "SGN": SGN(
            {'n_epochs': 15 if USING_DATASET == "redd" else 5, 'batch_size': 256,
             'sequence_length': 400 if USING_DATASET == 'redd' else 200,
             'appliance_length': 64 if USING_DATASET == 'redd' else 32,
             'test_only': False,
             'patience': 5 if USING_DATASET == "redd" else 3,
             # 'note': USING_DATASET + '_seen',
             'note': USING_DATASET
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
            {'redd': REDD_TEST_STD} if USING_DATASET == "redd" else {'ukdale': UKDALE_TEST_STD},
        # Specify evaluation metrics
        'metrics': ['accuracy', 'f1score', 'mae', 'sae', 'wssim']
    }
}

if __name__ == '__main__':
    API(e)
