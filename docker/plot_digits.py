from sklearn import datasets, svm, metrics, tree
import pdb
import argparse
from utils import (
    preprocess_digits,
    train_dev_test_split,
    data_viz,
    get_all_h_param_comb,
    tune_and_save,
    macro_f1
)
from joblib import dump, load
import os
train_frac, dev_frac, test_frac = 0.8, 0.1, 0.1
assert train_frac + dev_frac + test_frac == 1.0

parser = argparse.ArgumentParser(
        description='With Args')
parser.add_argument('--clf_name')
parser.add_argument('--random_state')
args=parser.parse_args()

# 1. set the ranges of hyper parameters
gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]

svm_params = {}
svm_params["gamma"] = gamma_list
svm_params["C"] = c_list
svm_h_param_comb = get_all_h_param_comb(svm_params)

max_depth_list = [2, 10, 20, 50, 100]

dec_params = {}
dec_params["max_depth"] = max_depth_list
dec_h_param_comb = get_all_h_param_comb(dec_params)

h_param_comb = {"svm": svm_h_param_comb, "decision_tree": dec_h_param_comb}

# PART: load dataset -- data from csv, tsv, jsonl, pickle
digits = datasets.load_digits()
data_viz(digits)
data, label = preprocess_digits(digits)
# housekeeping
del digits

# define the evaluation metric
metric_list = [metrics.accuracy_score, macro_f1]
h_metric = metrics.accuracy_score
path1 = r"/home/apujari/MLOPS-2022/models/"
path2 = r"/home/apujari/MLOPS-2022/results/"
n_cv = 1
results = {}
for n in range(n_cv):
    x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(
        data, label, train_frac,args.random_state,dev_frac
    )
    # PART: Define the model
    # Create a classifier: a support vector classifier
    models_of_choice = {
        "svm": svm.SVC(),
        "decision_tree": tree.DecisionTreeClassifier(),
    }
    
    #for clf_name in models_of_choice:
    if(args.clf_name=='svm'):
        clf_name='svm' 
        clf = models_of_choice[clf_name]
        print("[{}] Running hyper param tuning for {}".format(n,clf_name))
        actual_model_path = tune_and_save(
            clf, x_train, y_train, x_dev, y_dev, h_metric, h_param_comb[clf_name], model_path=path1
        )

        # 2. load the best_model
        best_model = load(actual_model_path)

        # PART: Get test set predictions
        # Predict the value of the digit on the test subset
        predicted = best_model.predict(x_test)
        if not clf_name in results:
            results[clf_name]=[]    

        results[clf_name].append({m.__name__:m(y_pred=predicted, y_true=y_test) for m in metric_list})
        # 4. report the test set accurancy with that best model.
        # PART: Compute evaluation metrics
        #print(
            #f"Classification report for classifier {clf}:\n"
           # f"{metrics.classification_report(y_test, predicted)}\n"
        #)
        print(f"model saved at "+ str(tune_and_save))
    elif(args.clf_name=='tree'):
        clf_name='decision_tree' 
        clf = models_of_choice[clf_name]
        print("[{}] Running hyper param tuning for {}".format(n,clf_name))
        actual_model_path = tune_and_save(
            clf, x_train, y_train, x_dev, y_dev, h_metric, h_param_comb[clf_name], model_path=path1
        )

        # 2. load the best_model
        best_model = load(actual_model_path)

        # PART: Get test set predictions
        # Predict the value of the digit on the test subset
        predicted = best_model.predict(x_test)
        if not clf_name in results:
            results[clf_name]=[]    

        results[clf_name].append({m.__name__:m(y_pred=predicted, y_true=y_test) for m in metric_list})
        # 4. report the test set accurancy with that best model.
        # PART: Compute evaluation metrics
        #print(
            #f"Classification report for classifier {clf}:\n"
            #f"{metrics.classification_report(y_test, predicted)}\n"
        #)
        print(f"model saved at "+ str(tune_and_save))
#with open('results/args.clf_name_args.random_state.txt', 'wb') as f:
    #dump(results, "results/args.clf_name_args.random_state.txt")
temp=args.clf_name + "_"+ args.random_state + ".txt"
model_path1="".join([path2, temp])
dump(results, model_path1)
#print(results)