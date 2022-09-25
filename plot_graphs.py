from sklearn import datasets, svm, metrics

from utils import preprocess_digits, train_dev_test_split,h_param_tuning1, h_param_tuning, data_viz, pred_image_viz


train_frac, dev_frac, test_frac = 0.8, 0.1 , 0.1
assert train_frac + dev_frac + test_frac == 1.

gamma_list = [ 0.05, 0.00112, 0.205, 0.01,0.00086]
c_list = [0.001, 0.03, 0.23,0.78, 1.5]

h_param_comb = [{"gamma": g, "C": c} for g in gamma_list for c in c_list]

assert len(h_param_comb) == len(gamma_list) * len(c_list)

# PART: load dataset -- data from csv, tsv, jsonl, pickle
digits = datasets.load_digits()
data_viz(digits)
data, label = preprocess_digits(digits)
# housekeeping
del digits


x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(
    data, label, train_frac, dev_frac
)

# PART: Define the model
# Create a classifier: a support vector classifier
clf = svm.SVC()
# define the evaluation metric
metric=metrics.accuracy_score

model1, metric1, h_params1 = h_param_tuning1(h_param_comb, clf, x_train, y_train, x_dev, y_dev, metric)

predicted1 = model1.predict(x_test)

pred_image_viz(x_test, predicted1)

print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted1)}\n"
)

best_model, best_metric, best_h_params = h_param_tuning(h_param_comb, clf, x_train, y_train, x_dev, y_dev, metric)
predicted = best_model.predict(x_test)
pred_image_viz(x_test, predicted)
print("Best hyperparameters were:")
print(best_h_params)
