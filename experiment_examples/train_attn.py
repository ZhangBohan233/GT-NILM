from nilmtk.api import API
import warnings

warnings.filterwarnings("ignore")
from nilmtk.disaggregate import AttentionCNN
import nilmtk.utils as utils
from dataset_info import *
import torch

USE_GPU = True
device = torch.device("cuda:0" if USE_GPU and torch.cuda.is_available() else "cpu")
print(torch.__version__, device)

torch.set_float32_matmul_precision('high')

USING_DATASET = 'ukdale'
MAIN_POWER = 'apparent' if USING_DATASET == 'redd' else 'active'
APPLIANCES = ['dish washer', 'fridge', 'microwave', 'washing machine'] \
    if USING_DATASET == 'redd' \
    else ['dish washer', 'fridge', 'kettle', 'microwave', 'washing machine']

e = {
    # Specify power type, sample rate and disaggregated appliance
    'power': {
        'mains_train': [MAIN_POWER],
        'mains_test': [MAIN_POWER],
        'appliance': ['active']
    },
    'sample_rate': 3 if USING_DATASET == 'redd' else 6,
    'appliances': APPLIANCES,
    'app_meta': utils.GENERAL_APP_META,
    # 'appliances': ['microwave'],
    # Universally no pre-training
    'pre_trained': False,
    # Specify algorithm hyper-parameters
    'methods': {
        "AttentionCNN": AttentionCNN({
            'n_epochs': 20 if USING_DATASET == 'redd' else 10,
            'note': USING_DATASET,
            'sequence_length': 259 if USING_DATASET == 'redd' else 129,
            'patience': 5,
            'test_only': False
        }),
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
