import os.path

from nilmtk.dataset import DataSet
import pandas as pd
from nilmtk.losses import *
import numpy as np
import matplotlib.pyplot as plt
import datetime
import nilmtk.utils as utils

all_type_power = ['active', 'apparent']


class API():
    # The API is designed for rapid experimentation with NILM Algorithms. 
    def __init__(self, params):
        # Initializes the API with default parameters
        self.power = {}
        self.sample_period = 1
        self.appliances = []
        self.methods = {}
        self.chunk_size = None
        self.pre_trained = False
        self.metrics = []
        self.train_datasets_dict = {}
        self.test_datasets_dict = {}
        self.artificial_aggregate = False
        self.train_submeters = []
        self.train_mains = pd.DataFrame()
        self.transfer_mains = pd.DataFrame()
        self.transfer_submeters = []
        self.test_submeters = []
        self.test_mains = pd.DataFrame()
        self.gt_overall = {}
        self.pred_overall = {}
        self.classifiers = []
        self.DROP_ALL_NANS = True
        self.mae = pd.DataFrame()
        self.rmse = pd.DataFrame()
        self.errors = []
        self.predictions = []
        self.errors_keys = []
        self.predictions_keys = []
        self.params = params
        if 'app_meta' in params:
            self.app_meta = params['app_meta']
        else:
            self.app_meta = utils.GENERAL_APP_META
        for elems in params['power']:
            self.power = params['power']
        self.sample_period = params['sample_rate']
        for elems in params['appliances']:
            self.appliances.append(elems)

        self.pre_trained = params['pre_trained']
        self.train_datasets_dict = params['train']['datasets']
        if 'transfer' in params:
            self.transfer_dataset_dict = params['transfer']['datasets']
        else:
            self.transfer_dataset_dict = None

        if 'save' in params:
            self.save = params['save']
        else:
            self.save = True
        if 'save_note' in params:
            self.save_note = params['save_note']
        else:
            self.save_note = None
        if 'csv_path' in params:
            self.csv_path = params['csv_path']
        else:
            self.csv_path = None
        self.gater = None
        self.gater_method = params['gater'] if 'gater' in params else None

        self.test_datasets_dict = params['test']['datasets']
        self.metrics = params['test']['metrics']
        self.methods = params['methods']
        self.artificial_aggregate = params.get('artificial_aggregate', self.artificial_aggregate)
        self.chunk_size = params.get('chunk_size', self.chunk_size)
        self.experiment(params)

    def experiment(self, params):
        # Calls the Experiments with the specified parameters

        self.store_classifier_instances()
        self.store_gater_instances()
        d = self.train_datasets_dict
        trans = self.transfer_dataset_dict

        if self.gater is not None:
            self.train_jointly(self.gater, d, None)

        for model_name, clf in self.classifiers:
            # If the model is a neural net, it has an attribute n_epochs, Ex: DAE, Seq2Point
            print("Started training for ", clf.MODEL_NAME)
            # If the model has the filename specified for loading the pretrained model, then we don't need to load training data
            if hasattr(clf, 'load_model_path'):
                if clf.load_model_path:
                    print(clf.MODEL_NAME, " is loading the pretrained model")
                    continue

            print("Joint training for ", clf.MODEL_NAME)
            self.train_jointly(clf, d, trans)
            print("Finished training for ", clf.MODEL_NAME)

        d = self.test_datasets_dict

        print("Joint Testing for all algorithms")
        self.test_jointly(d)

    def train_jointly(self, clf, d, trans_d=None):
        # This function has a few issues, which should be addressed soon
        print("............... Loading Data for training ...................")
        # Store the train_main readings for all buildings
        self.train_mains = pd.DataFrame()
        self.train_submeters = [pd.DataFrame() for i in range(len(self.appliances))]
        for dataset in d:
            print("Loading data for ", dataset, " dataset")
            train = DataSet(d[dataset]['path'])
            for building in d[dataset]['buildings']:
                print("Loading building ... ", building)
                train.set_window(start=d[dataset]['buildings'][building]['start_time'],
                                 end=d[dataset]['buildings'][building]['end_time'])
                train_df = next(
                    train.buildings[building].elec.mains().load(physical_quantity='power',
                                                                ac_type=all_type_power,
                                                                sample_period=self.sample_period,
                                                                resample=True,
                                                                resample_kwargs={
                                                                    'fill_method': None}
                                                                ))
                train_df.ffill(axis=0, inplace=True, limit=30)
                if 'active' in train_df['power']:
                    train_df['active'] = train_df['power']['active']
                if 'apparent' in train_df['power']:
                    train_df['apparent'] = train_df['power']['apparent']
                # train_df['reactive'] = np.sqrt(
                #     train_df['power']['apparent'] ** 2 - train_df['power']['active'] ** 2)
                train_df.drop(['power'], axis=1, inplace=True)
                train_df = train_df[self.power['mains_train']]

                appliance_readings = []
                for appliance_name in self.appliances:
                    appliance_df = next(train.buildings[building].elec[appliance_name].load(
                        physical_quantity='power', ac_type=self.power['appliance'],
                        sample_period=self.sample_period,
                        resample=True,
                        resample_kwargs={'fill_method': None}
                    ))
                    appliance_df.ffill(axis=0, inplace=True, limit=30)
                    # appliance_df = appliance_df[[list(appliance_df.columns)[0]]]
                    # appliance_df.clip(upper=self.app_meta[appliance_name]["max"])
                    appliance_readings.append(appliance_df)
                if self.DROP_ALL_NANS:
                    train_df, appliance_readings = self.dropna(train_df, appliance_readings)
                if self.artificial_aggregate:
                    print("Creating an Artificial Aggregate")
                    train_df = pd.DataFrame(np.zeros(appliance_readings[0].shape),
                                            index=appliance_readings[0].index,
                                            columns=appliance_readings[0].columns)
                    for app_reading in appliance_readings:
                        train_df += app_reading

                print("Train Jointly")
                self.train_mains = self.train_mains.append(train_df)
                for i, appliance_name in enumerate(self.appliances):
                    self.train_submeters[i] = self.train_submeters[i].append(appliance_readings[i])

        if trans_d is not None:
            self.transfer_mains = pd.DataFrame()
            self.transfer_submeters = [pd.DataFrame() for i in range(len(self.appliances))]
            for dataset in trans_d:
                print("Loading transfer data for ", dataset, " dataset")
                transfer = DataSet(trans_d[dataset]['path'])
                for building in trans_d[dataset]['buildings']:
                    print("Loading building ... ", building)
                    transfer.set_window(start=trans_d[dataset]['buildings'][building]['start_time'],
                                        end=trans_d[dataset]['buildings'][building]['end_time'])
                    transfer_df = next(
                        transfer.buildings[building].elec.mains().load(physical_quantity='power',
                                                                       ac_type=all_type_power,
                                                                       sample_period=self.sample_period,
                                                                       resample=True,
                                                                       resample_kwargs={
                                                                           'fill_method': None}
                                                                       ))
                    transfer_df.ffill(axis=0, inplace=True, limit=30)
                    if 'active' in transfer_df['power']:
                        transfer_df['active'] = transfer_df['power']['active']
                    if 'apparent' in transfer_df['power']:
                        transfer_df['apparent'] = transfer_df['power']['apparent']
                    # train_df['reactive'] = np.sqrt(
                    #     train_df['power']['apparent'] ** 2 - train_df['power']['active'] ** 2)
                    transfer_df.drop(['power'], axis=1, inplace=True)
                    transfer_df = transfer_df[self.power['mains_transfer']]

                    appliance_readings = []
                    for appliance_name in self.appliances:
                        appliance_df = next(transfer.buildings[building].elec[appliance_name].load(
                            physical_quantity='power', ac_type=self.power['appliance'],
                            sample_period=self.sample_period,
                            resample=True,
                            resample_kwargs={'fill_method': None}
                        ))
                        appliance_df.ffill(axis=0, inplace=True, limit=30)
                        # appliance_df.clip(upper=self.app_meta[appliance_name]["max"])
                        # appliance_df = appliance_df[[list(appliance_df.columns)[0]]]
                        appliance_readings.append(appliance_df)
                    if self.DROP_ALL_NANS:
                        transfer_df, appliance_readings = (
                            self.dropna(transfer_df, appliance_readings))

                    print("Transfer Jointly")
                    self.transfer_mains = self.transfer_mains.append(transfer_df)
                    for i, appliance_name in enumerate(self.appliances):
                        self.transfer_submeters[i] = self.transfer_submeters[i].append(
                            appliance_readings[i])

        appliance_readings = []
        transfer_app_readings = []
        for i, appliance_name in enumerate(self.appliances):
            appliance_readings.append((appliance_name, [self.train_submeters[i]]))
            if trans_d is not None:
                transfer_app_readings.append((appliance_name, [self.transfer_submeters[i]]))

        # print("Transfer main of", type(clf), self.transfer_mains, self.transfer_submeters)
        self.train_mains = [self.train_mains]
        self.transfer_mains = [self.transfer_mains]
        self.train_submeters = appliance_readings
        self.transfer_submeters = transfer_app_readings
        # print("Train and Transfer", self.train_mains, self.transfer_mains)
        if self.transfer_dataset_dict is not None:
            clf.partial_fit(self.train_mains, self.train_submeters, self.pre_trained, True,
                            dst_main=self.transfer_mains, dst_appliances=self.transfer_submeters)
        else:
            clf.partial_fit(self.train_mains, self.train_submeters, self.pre_trained, True)

    def test_jointly(self, d):
        # Store the test_main readings for all buildings
        for dataset in d:
            print("Loading data for ", dataset, " dataset")
            test = DataSet(d[dataset]['path'])
            for building in d[dataset]['buildings']:
                test.set_window(start=d[dataset]['buildings'][building]['start_time'],
                                end=d[dataset]['buildings'][building]['end_time'])
                test_mains = next(
                    test.buildings[building].elec.mains().load(physical_quantity='power',
                                                               ac_type=all_type_power,
                                                               sample_period=self.sample_period,
                                                               resample=True,
                                                               resample_kwargs={
                                                                   'fill_method': None}))
                test_mains.ffill(axis=0, inplace=True, limit=30)
                if 'active' in test_mains['power']:
                    test_mains['active'] = test_mains['power']['active']
                if 'apparent' in test_mains['power']:
                    test_mains['apparent'] = test_mains['power']['apparent']
                # test_mains['reactive'] = np.sqrt(
                #     test_mains['power']['apparent'] ** 2 - test_mains['power']['active'] ** 2)
                test_mains.drop(['power'], axis=1, inplace=True)
                test_mains = test_mains[self.power['mains_test']]
                '''
                train_df = next(train.buildings[building].elec.mains().load(physical_quantity='power', ac_type= all_type_power,sample_period = self.sample_period))
                train_df['active'] = train_df['power']['active']
                train_df['apparent'] = train_df['power']['apparent']
                train_df['reactive'] = np.sqrt(train_df['power']['apparent']**2 - train_df['power']['active']**2)
                train_df.drop(['power'], axis = 1,inplace = True)
                train_df = train_df[self.power['mains']]
                '''

                appliance_readings = []

                for appliance in self.appliances:
                    test_df = next((test.buildings[building].elec[appliance].load(
                        physical_quantity='power', ac_type=self.power['appliance'],
                        sample_period=self.sample_period,
                        resample=True,
                        resample_kwargs={'fill_method': None}
                    )))
                    test_df.ffill(axis=0, inplace=True, limit=30)
                    # test_df.clip(upper=self.app_meta[appliance]["max"])
                    appliance_readings.append(test_df)

                # print(test_mains.head(50))

                if self.DROP_ALL_NANS:
                    test_mains, appliance_readings = self.dropna(test_mains, appliance_readings)

                if self.artificial_aggregate:
                    print("Creating an Artificial Aggregate")
                    test_mains = pd.DataFrame(np.zeros(appliance_readings[0].shape),
                                              index=appliance_readings[0].index,
                                              columns=appliance_readings[0].columns)
                    for app_reading in appliance_readings:
                        test_mains += app_reading

                self.test_mains = [test_mains]
                for i, appliance_name in enumerate(self.appliances):
                    self.test_submeters.append((appliance_name, [appliance_readings[i]]))

                self.storing_key = str(dataset) + "_" + str(building)
                self.call_predict(self.classifiers, dataset)

    def dropna(self, mains_df, appliance_dfs):
        # Drop missing value
        print("Dropping missing values")

        # The below steps are for making sure that data is consistent by doing intersection across appliances
        mains_df = mains_df.dropna()
        # mains_df = mains_df.loc[(mains_df!=0).any(axis=1)]
        for i in range(len(appliance_dfs)):
            appliance_dfs[i] = appliance_dfs[i].dropna()
        ix = mains_df.index
        for app_df in appliance_dfs:
            ix = ix.intersection(app_df.index)
        mains_df = mains_df.loc[ix]
        new_appliances_list = []
        for app_df in appliance_dfs:
            new_appliances_list.append(app_df.loc[ix])
        return mains_df, new_appliances_list

    def store_gater_instances(self):
        if self.gater_method is not None:
            try:
                print(self.gater_method)
                self.gater = self.gater_method

            except Exception as e:
                print("\n\nThe method {model_name} specied does not exist. \n\n".format(
                    model_name=self.gater_method))
                print(e)

    def store_classifier_instances(self):
        # This function is reponsible for initializing the models with the specified model parameters
        for name in self.methods:
            try:
                print(name)
                clf = self.methods[name]
                self.classifiers.append((name, clf))

            except Exception as e:
                print("\n\nThe method {model_name} specied does not exist. \n\n".format(
                    model_name=name))
                print(e)

    def call_predict(self, classifiers, dataset):
        # This functions computers the predictions on the self.test_mains using all the trained models and then compares different learn't models using the metrics specified        
        pred_gate = None
        if self.gater is not None:
            # print("Test main shape", self.test_mains[0].values.shape)
            gt_gate, pred_gate = self.predict(self.gater, self.test_mains, self.test_submeters,
                                              self.sample_period, 'Europe/London')

        pred_overall = {}
        gt_overall = pd.DataFrame()

        mains_df = self.test_mains[0].copy()
        ix = mains_df.index

        for name, clf in classifiers:
            gt_overall, pred = self.predict(clf, self.test_mains, self.test_submeters,
                                            self.sample_period, 'Europe/London',
                                            pred_gate=pred_gate)
            path = "predict" + clf.MODEL_NAME + ".csv"
            pred_overall[name] = pred
            pred.to_csv(path)

            ix = ix.intersection(pred.index)
            if pred_gate is not None:
                ix = ix.intersection(pred_gate.index)

        ix = ix.intersection(gt_overall.index)

        mains_df = mains_df.loc[ix]
        self.gt_overall = gt_overall.loc[ix]
        for k in pred_overall:
            pred_overall[k] = pred_overall[k].loc[ix]

        if pred_gate is not None:
            pred_gate = pred_gate.loc[ix]

        self.pred_overall = pred_overall
        self.gt_overall.to_csv("truth.csv")

        for clf_name in pred_overall:
            general_df = pd.DataFrame({'mains': mains_df.iloc[:, 0].to_numpy()}, index=ix)
            pred = pred_overall[clf_name]
            for app in pred:
                app_pred = pred[app]
                if pred_gate is not None:
                    app_gate = pred_gate[app]

                    general_df[app + "_ungated"] = app_pred
                    general_df[app + "_gate"] = app_gate

                    plt.plot(app_pred, label=app + "_ungated")
                    plt.plot(app_gate * 1000, label="gate")
                    plt.legend()
                    plt.show()

                    app_pred *= app_gate

                general_df[app + "_truth"] = gt_overall[app]
                general_df[app + "_pred"] = app_pred

            path = f'{dataset}-{"+".join([str(app) for app in pred])}-{clf_name}.csv'
            if self.save_note is not None:
                path = self.save_note + "-" + path
            if self.save:
                if self.csv_path is not None:
                    os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
                    general_df.to_csv(self.csv_path)
                else:
                    general_df.to_csv("./" + path)

        if gt_overall.size == 0:
            print("No samples found in ground truth")
            return None

        for i in gt_overall.columns:

            plt.figure(figsize=(28, 10))
            plt.plot(self.test_mains[0], label='Mains reading')
            plt.plot(gt_overall[i], label='Truth')
            for clf in pred_overall:
                plt.plot(pred_overall[clf][i], label=clf)
            plt.title(i)
            plt.legend()
            plt.show()

        for metric in self.metrics:
            try:
                loss_function = globals()[metric]
            except:
                print("Loss function ", metric, " is not supported currently!")
                continue

            computed_metric = {}
            for clf_name, clf in classifiers:
                computed_metric[clf_name] = self.compute_loss(gt_overall, pred_overall[clf_name],
                                                              loss_function)
            computed_metric = pd.DataFrame(computed_metric)
            print("............ ", metric, " ..............")
            print(computed_metric)
            self.errors.append(computed_metric)
            self.errors_keys.append(self.storing_key + "_" + metric)

    def predict(self, clf, test_elec, test_submeters, sample_period, timezone, **kwargs):
        # Generates predictions on the test dataset using the specified classifier.        
        print("Generating predictions for :", clf.MODEL_NAME)
        # "ac_type" varies according to the dataset used. 
        # Make sure to use the correct ac_type before using the default parameters in this code.

        if 'pred_gate' in kwargs and kwargs['pred_gate'] is not None:
            pred_list = clf.disaggregate_chunk(test_elec, pred_gate=kwargs['pred_gate'])
        else:
            pred_list = clf.disaggregate_chunk(test_elec)
        concat_pred_df = pd.concat(pred_list, axis=0)

        gt = {}
        for meter, data in test_submeters:
            concatenated_df_app = pd.concat(data, axis=1)
            index = concatenated_df_app.index
            gt[meter] = pd.Series(concatenated_df_app.values.flatten(), index=index)

        gt_overall = pd.DataFrame(gt, dtype='float32')
        pred = {}
        for app_name in concat_pred_df.columns:
            app_series_values = concat_pred_df[app_name].values.flatten()
            # Neural nets do extra padding sometimes, to fit, so get rid of extra predictions
            length = min(len(gt_overall), len(gt_overall[app_name]))
            app_series_values = app_series_values[:length]
            if len(app_series_values) < length:
                app_series_values = np.append(app_series_values,
                                              np.zeros(length - len(app_series_values)))
            index = gt_overall.index[:length]
            # print(app_series_values)
            # print(gt_overall[app_name])
            print("length", length, len(index), len(app_series_values))
            pred[app_name] = pd.Series(app_series_values, index=index)
        pred_overall = pd.DataFrame(pred, dtype='float32')
        # print(pred_overall)
        return gt_overall, pred_overall

    # metrics
    def compute_loss(self, gt, clf_pred, loss_function):
        kwargs = {}
        if loss_function.__name__ == 'sae':
            kwargs['window_size'] = 3600 // self.sample_period

        error = {}
        for app_name in gt.columns:
            if kwargs:
                print("sae with ws", kwargs)
                error[app_name] = loss_function(app_name,
                                                gt[app_name].values,
                                                clf_pred[app_name].values,
                                                **kwargs)
            else:
                error[app_name] = loss_function(app_name,
                                                gt[app_name].values,
                                                clf_pred[app_name].values)
        return pd.Series(error)
