REDD_AVAIL = {
    'fridge': [1, 2, 3, 5, 6],
    'washing machine': [1, 2, 3, 4, 5, 6],
    'microwave': [1, 2, 3, 5]
}

REDD_TRAIN_STD = {
    'path': '../mnt/redd.h5',
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
    'path': '../mnt/redd.h5',
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
    'path': '../mnt/ukdale.h5',
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
    'path': '../mnt/ukdale.h5',
    'buildings': {
        2: {
            'start_time': '2013-05-25',
            'end_time': '2013-09-30'
        },
    },
}

UKDALE_TEST_SPECIAL_SAMPLE = {
    'path': '../mnt/ukdale.h5',
    'buildings': {
        2: {
            'start_time': '2013-06-01',
            'end_time': '2013-06-04'
        },
    },
}
