from nilmtk.api import API
import warnings

warnings.filterwarnings("ignore")
from nilmtk.disaggregate import GaterCNN, DM_GATE2
import nilmtk.utils as utils

import torch

USE_GPU = True
device = torch.device("cuda:0" if USE_GPU and torch.cuda.is_available() else "cpu")
print(torch.__version__, device)

torch.set_float32_matmul_precision('high')

REDD_AVAIL = {
    'fridge': [1, 2, 3, 5, 6],
    'washing machine': [1, 2, 3, 4, 5, 6],
    'microwave': [1, 2, 3, 5]
}

REDD_TRAIN_STD = {
    # 'path': '../MultiDiffNILM/mnt/redd.h5',
    'path': 'mnt/redd.h5',
    'buildings': {
        2: {
            'start_time': '2011-04-18',
            'end_time': '2011-05-21'
        },
        3: {
            'start_time': '2011-04-17',
            'end_time': '2011-05-29'
        },
        # 4: {
        #     'start_time': '2011-04-17',
        #     'end_time': '2011-06-02'
        # },
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

REDD_TEST_STD = {
    # 'path': '../MultiDiffNILM/mnt/redd.h5',
    'path': 'mnt/redd.h5',
    # 'buildings': {
    #     2: {
    #         'start_time': '2011-04-26',
    #         'end_time': '2011-04-30'
    #     }
    # }
    'buildings': {
        1: {
            'start_time': '2011-04-19',
            'end_time': '2011-05-23'
        }
        # 1: {
        #     'start_time': '2011-04-28 05:57',
        #     'end_time': '2011-05-01'
        # }
    }
}

UKDALE_TRAIN_STD = {
    'path': 'mnt/ukdale.h5',
    'buildings': {
        1: {
            'start_time': '2013-05-31',
            'end_time': '2014-12-31'
            # 'end_time': '2013-12-31'
        },
        # 2: {
        #     'start_time': '2013-05-22',
        #     'end_time': '2013-08-01'
        # },
        5: {
            'start_time': '2014-07-01',
            'end_time': '2014-09-05'
        },

    },
}

UKDALE_TEST_STD = {
    'path': 'mnt/ukdale.h5',
    'buildings': {
        2: {
            'start_time': '2013-05-25',
            'end_time': '2013-09-30'
        },
    },
}

UKDALE_TEST_SPECIAL_SAMPLE = {
    'path': 'mnt/ukdale.h5',
    'buildings': {
        2: {
            'start_time': '2013-06-01',
            'end_time': '2013-06-04'
        },
    },
}

USING_DATASET = 'redd'
MAIN_POWER = 'apparent' if USING_DATASET == 'redd' else 'active'

e = {
    # Specify power type, sample rate and disaggregated appliance
    'power': {
        'mains_train': [MAIN_POWER],
        'mains_test': [MAIN_POWER],
        'appliance': ['active']
    },
    'sample_rate': 3 if USING_DATASET == 'redd' else 6,
    # 'appliances': ['washing machine'],
    'app_meta': utils.APP_META[USING_DATASET],
    'appliances': ['dish washer', 'fridge', 'microwave', 'washing machine'],
    # Universally no pre-training
    'pre_trained': False,
    'save_note': 'gated',
    # Specify algorithm hyper-parameters
    # gater is a network that identify the on/off states of an appliance
    # if user does not specify this parameter, the DM's output will be the final result
    'gater': GaterCNN(
        {
            'n_epochs': 10 if USING_DATASET == "redd" else 5,
            'batch_size': 128,
            'sequence_length': 400 if USING_DATASET == 'redd' else 200,
            'appliance_length': 64 if USING_DATASET == 'redd' else 32,
            # conduct test only, no training
            'test_only': False,
            # name suffix of the state-dict file.
            'note': USING_DATASET
        }),
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
                "app_meta": utils.APP_META[USING_DATASET],
                # whether to train the DM on active windows only
                'filter_train': True
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
