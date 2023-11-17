from collections import defaultdict
import operator
import numpy as np
import math
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score
from sklearn.model_selection import train_test_split

def preprocess_dataset(df, target_col, columns_to_bin=None, columns_to_encode=None):
    """    encodes values, uses 5bin discretization
    1. Load from .csv file
    2. Apply binning
    3. Encode values (and encode bins)
    """
    #df = pd.read_csv(path)
    encodings = dict()
    encodings_2 = dict()
    if columns_to_bin:
        for col in columns_to_bin:
            qc = pd.qcut(df[col], q=5)
            # create bin encodings
            bin_encodings = {k:v for v, k in enumerate(qc.unique())}
            encodings[col] = bin_encodings
            df[col] = df[col].map(bin_encodings)
    if columns_to_encode:
        for col in columns_to_encode:
            encodings_2 = {k:v for v, k in enumerate(df[col].unique())}
            df[col] = df[col].map(encodings_2)
    # unify nan values representation
    df = df.replace(['?', 'Nan', 'NaN'], 'nan')
    return df, encodings


def construct_freq_table(dataset):
    """
    constructs freqeuncy table according to the algorithm described in Selective NB paper
    """
    freq_table = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
    unique_targets =set() # w zagnieżdżonym słowniku klucze to wartości przyjmowane przez cechę
    for ind, instance in dataset.iterrows():
        #print(instance)
        y = instance[-1]
        unique_targets.add(y)
        for i, j in enumerate(instance[:-1]):
            if j in freq_table[i].keys():
                freq_table[i][j][y] += 1
            else:
                freq_table[i][j][y]=1
    return freq_table, unique_targets


# class Freq_table():
#     def __init__(self, dataset):
#         self.table, self.unique_targets = construct_freq_table(dataset)
#     def pretty_print(self):
#         for k in self.table:
#             print(f'Attribute: {k}')
#             for a in self.table[k]:
#                 print(f'Frequencies of value {a}')
#                 freqs = self.table[k][a]
#                 for j, m in freqs.items():
#                     print(f'Class: {j}, number of occurences: {m}')


def calc_p_xi_y(i,j,y,freq_table,m=1):
    try:
        nominator = freq_table[i][j][y] + m/(len(freq_table[i].keys()))
    except KeyError:
        nominator = m/(len(freq_table[i].keys()))
    # suma wystąpień wszystkich kategorii dla atrybutu i i class y
    sum_freq_table = 0
    try:
        for l in freq_table[i].keys():
            sum_freq_table += freq_table[i][l][y]
        denominator = sum_freq_table + m
    except:
        denominator = m
    return nominator/denominator


def remove_inst_from_freq_table(example, freq_table):
    y = example[-1]
    for attr_ind, attr_val in enumerate(example[:-1]):
        if freq_table[attr_ind][attr_val][y] >= 1:
            freq_table[attr_ind][attr_val][y] -= 1
    return freq_table


def add_inst_to_freq_table(example, freq_table):
    y = example[-1]
    for attr_ind, attr_val in enumerate(example[:-1]):
        if freq_table[attr_ind][attr_val][y] >= 0:
            freq_table[attr_ind][attr_val][y] += 1
    return freq_table
# def mutual_information(X,y,i, unique_targets, freq_table):
#     unique_X = set(X)
#     mi = 0
#     for targ in unique_targets:
#         for x in unique_X:
#             px = X.count(x)/len(y)
#             py = y.count(targ)/len(y)
#             pxy = freq_table[i][x][targ]/len(y)
#             print(px, py, pxy)
#             if pxy == 0:
#                 continue
#             else:
#                 mi_partial = pxy*math.log((pxy)/(px*py), 2)
#                 mi += mi_partial
#     return mi
class NB_classifier():
    def __init__(self, dataset, target_col, verbose=False):
        self.dataset = dataset
        self.freq_table, self.unique_targets = construct_freq_table(self.dataset) 
        self.target_col = target_col
        if verbose:
            print(self.freq_table) 
    def classify(self, example):
        #example to havee the same order as in training examples used to construct frequency matrix
        targets_probs = dict()
        prior_probs = {y_cat: len(self.dataset[self.dataset[self.target_col]==y_cat]) for y_cat in self.unique_targets}
        for y_cat in self.unique_targets:
            joint_prob = 1
            for i, attr in enumerate(example[:-1]):
                joint_prob = joint_prob*calc_p_xi_y(i, attr, y_cat, self.freq_table)
            targets_probs[y_cat] = joint_prob*prior_probs[y_cat]
        print(targets_probs)
        return  max(targets_probs, key=targets_probs.get)
    

class Selective_NB_classifier():
    def __init__(self, dataset, target_col='y'):
        self.dataset = dataset
        self.freq_table, self.unique_targets = construct_freq_table(dataset)
        self.freq_table_original, _ = construct_freq_table(dataset)
        # unique targets: unique values of target column
        self.attr_mapping = {k:v for v, k in enumerate(self.dataset.columns[:-1])} # initial attribute names to indexes mapping
        self.attr_mapping_reverse = {v:k for v, k in enumerate(self.dataset.columns[:-1])} # reverse: indexes to attribute names mapping
        self.target_col = target_col
        self.rmse_dict = defaultdict(lambda: []) # dictionary of model versions (key is a tuple of model attrs) and the list of RMSE values for instaces
    
    def calculate_mutual_information(self):
        attr_mi = dict()
        for i, attr in enumerate(self.dataset.columns[:-1]):
            #attr_mi[attr] = mutual_information(self.dataset[attr].tolist(), self.dataset[self.target_col].tolist(), i, self.unique_targets, self.freq_table)
            attr_mi[attr] = normalized_mutual_info_score(self.dataset[attr].tolist(), self.dataset[self.target_col].tolist())      
        self.attr_mi = attr_mi
    
    def order_by_mi(self):
        """
        order encoded attributes by mutual information
        """
        self.attr_mi = dict(sorted(self.attr_mi.items(), key=lambda item: item[1], reverse=True))
        self.ordered_attrs = [self.attr_mapping[attr] for attr in self.attr_mi.keys()]
        self.ordered_attrs_verbose = [attr for attr in self.attr_mi.keys()]
        # construct frequency table with ordered attributes
        pass
    
    def predict_using_all_models(self,example):
        """
        predict one example using all models.
        Modifies self.rmse_dict with every new example.
        """
        self.freq_table = remove_inst_from_freq_table(example,self.freq_table) # table with example removed and frequencies updated
        targets_probs = dict() # dictionary with possible
        models_dict = dict()
        temp_attrs = [] # temporary list of attributes based on which freq table is created
        temp_dict = dict()
        y_gold = example[-1] # true label
        models_dict = dict()
        #rmse_dict = dict()
        # prior probabilities - TODO change it so that the predicted example is not taken into account in prior calculation
        prior_probs = dict()
        prob_dict = {y_cat: 1 for y_cat in self.unique_targets}
        for y_cat in self.unique_targets:
            if y_cat == y_gold:
                prior_probs[y_cat] = (len(self.dataset[self.dataset[self.target_col]==y_cat])-1)/(len(self.dataset)-1)
            else:
                prior_probs[y_cat] = len(self.dataset[self.dataset[self.target_col]==y_cat])/(len(self.dataset)-1)
        attrs_to_consider_dict = dict()
        for i, attr in enumerate(self.ordered_attrs):
            # iterating over encoded attributes, ordered by mutual information
            accum_squared_error = 0 # holds the sum of squared errors for model predictions for all classes for the example
            temp_attrs.append(attr) # extending list of temporary attributes
            temp_dict[attr] = self.freq_table[attr]
            temp_attrs_tuple = tuple(m for m in temp_attrs) #dictionary which is a temporary table keeps expanding with iterations over attributes
            models_dict[temp_attrs_tuple] = {a: 0 for a in self.unique_targets}
            predictions = dict()
            # dictionary where tuple with attr ids is the key and dictionaries of predicted probabilities for each class are held
            # predict the removed instance based on the temporary table
            for y_cat in self.unique_targets:
                py = prior_probs[y_cat]
                j = example[self.attr_mapping_reverse[attr]] # access attribute value at correct index in the passed example, attr is already mapped to attribute encoding
                try:
                    prob_xi_y = calc_p_xi_y(attr,j,y_cat, temp_dict)
                    prob_dict[y_cat] = prob_dict[y_cat]*prob_xi_y
                except KeyError:
                    print('Zero probability error!')
                    prob_dict[y_cat] = 0
                finally:
                    models_dict[temp_attrs_tuple][y_cat] = prob_dict[y_cat]*py
            # normalize the probabilities by the sum of predicted probabilities
            sum_of_probs = sum(prob_dict.values())
            predictions = {y_c: prob_dict[y_c]/sum_of_probs for y_c in self.unique_targets}
            # calculate the squared error
            squared_err = (1-predictions[y_gold])**2
            accum_squared_error += squared_err
            self.rmse_dict[temp_attrs_tuple].append(accum_squared_error)
        self.freq_table = add_inst_to_freq_table(example, self.freq_table) # add the instance back to frequency table
        #print(models_dict)
        return models_dict, self.rmse_dict
        # remove instance from frequency table
    
    def select_best_model(self):
        """
        performs leave-one-out crossvalidation to accumulate errors in self.rmse_dict and chooses the best model
        """
        for ind, example in self.dataset.iterrows():
            #iterating over examples and updating rmse_dict
            print(f'Instance {ind}')
            models_dict, self.rmse_dict = self.predict_using_all_models(example)
            # print(f'Instance: {ind}')
            # print(models_dict)
            # print(self.rmse_dict)
        # select best model based on accumulated RMSE
        #print(self.rmse_dict)
        self.rmse_sum_dict = {key: math.sqrt(sum(rmse)/len(rmse))for key, rmse in self.rmse_dict.items()}
        self.best_model = min(self.rmse_sum_dict, key=self.rmse_sum_dict.get)
        print(self.rmse_sum_dict)
        print(f'Best model attributes: {self.best_model}')
        # attributes of the best model
        #best freqeuncy table in incorrectly constructed
        self.best_freq_table = dict()
        for attr in self.best_model:
            self.best_freq_table[attr] = self.freq_table[attr]
    
    def classify(self, example):
        targets_probs = dict()
        # choose only attributes of example which belong to the best model attributes
        example_prep = dict()
        for i in self.best_model:
            example_prep[i] = example[i]
        assert len(example_prep.keys()) == len(self.best_model)
        prior_probs = {y_cat: len(self.dataset[self.dataset[self.target_col]==y_cat])/len(self.dataset) for y_cat in self.unique_targets}
        for y_cat in self.unique_targets:
            joint_prob = 1
            for i, attr in example_prep.items():
                joint_prob = joint_prob*calc_p_xi_y(i, attr, y_cat, self.freq_table)
            targets_probs[y_cat] = joint_prob*prior_probs[y_cat]
        # for i, attr in example_prep.items():
        #     for y_cat in self.unique_targets:
        #         j = example_prep[i] # access attribute value at correct index in the passed example, attr is already mapped to attribute encoding
        #         # tutaj cos sie wywala
        #         #try:
        #         prob = calc_p_xi_y(i,j,y_cat, self.freq_table)
        #         # except KeyError:
        #         #     #print('Zero probability error!')
        #         #     prob = 0
        #         #finally:
        #         targets_probs[y_cat] = targets_probs[y_cat]*prob
        # #print(targets_probs)
        # for y_cat in self.unique_targets:
        #     targets_probs[y_cat] = targets_probs[y_cat]*prior_probs[y_cat]
        print(f'Predicted class: {max(targets_probs, key=targets_probs.get)}')
        return max(targets_probs, key=targets_probs.get)
    
    def describe(self):
        print(f'Model attributes ranked by mutual information: {self.ordered_attrs}')
        print(f'Best model attributes: {self.best_model}')


def test1():
    import pandas as pd
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    dataset = pd.read_csv('./lenses.csv', header=None)
    dataset.columns = ['ind', 'age', 'spectacle', 'astigmatic', 'tear', 'target']
    dataset = dataset[['age', 'spectacle', 'astigmatic', 'tear', 'target']]
    print("Number of NaN values:")
    print(dataset['target'].isnull().sum())
    train, test = train_test_split(dataset, test_size=0.2, random_state=11)
    preprocess_dataset(train, 'target')
    # f_table = Freq_table(train)
    # f_table.pretty_print()
    print(construct_freq_table(train))
    snb = Selective_NB_classifier(train, target_col='target')
    snb.calculate_mutual_information()
    snb.order_by_mi()
    print(snb.attr_mi)
    print(snb.ordered_attrs)
    snb.select_best_model()
    preds = []
    gold = []
    snb.classify(test.iloc[0])
    for ind, example in test.iterrows():
        prediction = snb.classify(example)
        preds.append(prediction)
        gold.append(example['target'])


def test2():
#     from ucimlrepo import fetch_ucirepo 
# # fetch dataset 
#     dataset = fetch_ucirepo(id=2) 
#     #dataset = pd.read_csv('./adult.csv', header=None)
#     #dataset.columns = [f'col_{n}' for n in range(1,15)] + ['target']
#     dataset = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
#     print("Number of NaN values:")
#     print(dataset['income'].isnull().sum())
#     dataset.to_csv('adult.csv')
    dataset = pd.read_csv('adult.csv')
    dataset = dataset.drop(columns=['Unnamed: 0'])
    train, test = train_test_split(dataset, test_size=0.9)
    columns_to_bin =  ['age', 'fnlwgt']
    preprocess_dataset(train, 'income', columns_to_bin=columns_to_bin, columns_to_encode=['income'])
    preprocess_dataset(test, 'income', columns_to_bin=columns_to_bin, columns_to_encode=['income'])
    # f_table = Freq_table(train)
    # f_table.pretty_print()
    snb = Selective_NB_classifier(train, target_col='income')
    snb.calculate_mutual_information()
    snb.order_by_mi()
    snb.select_best_model()
    preds = []
    gold = []
    for ind, example in test.iterrows():
        prediction = snb.classify(example)
        preds.append(prediction)
        gold.append(example['income'])

if __name__ == "__main__":
    #dataset = [[1,50,1], [0,44,2], [1, 33, 1], [1,22,1], [1, 11, 1]]
    #dataset = pd.DataFrame(dataset, columns = ['Category', 'Name', 'Target'])
    # dataset = pd.read_csv('./labor.csv', header=None)
    # dataset.columns = [f'col_{n}' for n in range(1,17)] + ['target']
    # #dataset = dataset[]
    # print("Number of NaN values:")
    # print(dataset['target'].isnull().sum())
    # preprocess_dataset(dataset, 'target', columns_to_encode=['target'])
    # f_table = Freq_table(dataset)
    # f_table.pretty_print()
    # print(construct_freq_table(dataset))
    # snb = Selective_NB_classifier(dataset, target_col='target')
    # snb.calculate_mutual_information()
    # snb.order_by_mi()
    # print(snb.attr_mi)
    # print(snb.ordered_attrs)
    # snb.select_best_model()
    # example = dataset.iloc[0]
    # snb.predict_using_all_models(example)
    test2()