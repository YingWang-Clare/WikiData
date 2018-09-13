import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import GridSearchCV
from time import time
import matplotlib.pyplot as plt

# ?human ?gender ?cause ?date_of_birth ?date_of_death ?country ?occupation ?religion
data = pd.read_csv('query.csv')
print(data.info())
print("The total number of data points is: {}".format(data.shape[0]))
data = data.drop_duplicates(subset='human')
print("After de-du, the total number of data points is: {}".format(data.shape[0]))

data_raw = data.drop('human', axis=1)
features_raw = data_raw.drop(['cause', 'date_of_birth', 'date_of_death'], axis=1)
label_raw = data['cause']  # cause
# print(features_raw.head(5))
print("# label: {}".format(label_raw.shape))
print("# data: {}".format(data_raw.shape))

x = label_raw.value_counts()
df = pd.DataFrame(x)
print(df)

features = pd.get_dummies(features_raw, columns=['gender', 'country', 'occupation', 'religion'])
label = pd.get_dummies(label_raw)
print(features.shape, label.shape)
# features: gender, country, occupation, religion

X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=.2, random_state=5)
print("Training set has {} samples.\nTesting set has {} samples."
      .format((X_train.shape[0]), X_test.shape[0]))

# DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=5)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)
print('The accuracy for Decision Tree is {}.\nThe f1_score for Decision Tree is {}.'
      .format(accuracy_score(y_test, y_pred_tree), f1_score(y_test, y_pred_tree, average='micro')))
# The accuracy for Decision Tree is 0.6308900523560209.
# The f1_score for Decision Tree is 0.6731843575418994.

# Naive Bayes
y_pred_NB = OneVsRestClassifier(GaussianNB()).fit(X_train, y_train).predict(X_test)
print('The accuracy for Naive Bayes is {}.\nThe f1_score for Naive Bayes is {}.' \
      .format(accuracy_score(y_test, y_pred_NB), f1_score(y_test, y_pred_NB, average='micro')))

# SVM
y_pred_svm = OneVsRestClassifier(SVC(kernel='linear', random_state=5)).fit(X_train, y_train).predict(X_test)
print('The accuracy for SVC is {}.\nThe f1_score for SVC is {}.' \
      .format(accuracy_score(y_test, y_pred_svm), f1_score(y_test, y_pred_svm, average='micro')))

# randomForest
random = RandomForestClassifier(random_state=5)
random.fit(X_train, y_train)
y_pred_random = random.predict(X_test)
print('The accuracy for Random Forest is {}.\nThe f1_score for Random Forest is {}.'
      .format(accuracy_score(y_test, y_pred_random), f1_score(y_test, y_pred_random, average='micro')))

# AdaBoost
y_pred_boost = OneVsRestClassifier(AdaBoostClassifier(random_state=5)).fit(X_train, y_train).predict(X_test)
print('The accuracy for AdaBoost is {}.\nThe f1_score for AdaBoost is {}.'
      .format(accuracy_score(y_test, y_pred_boost), f1_score(y_test, y_pred_boost, average='micro')))

# KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print('The accuracy for KNN is {}.\nThe f1_score for KNN is {}.'
      .format(accuracy_score(y_test, y_pred_knn), f1_score(y_test, y_pred_knn, average='micro')))

# MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=10)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)
print('The accuracy for MLP is {}.\nThe f1_score for MLP is {}.'
      .format(accuracy_score(y_test, y_pred_mlp), f1_score(y_test, y_pred_mlp, average='micro')))

# The accuracy for Decision Tree is 0.6308900523560209.
# The f1_score for Decision Tree is 0.6731843575418994.

# The accuracy for Naive Bayes is 0.15968586387434555.
# The f1_score for Naive Bayes is 0.18050541516245486.

# The accuracy for SVC is 0.6740837696335078.
# The f1_score for SVC is 0.7166666666666666.

# The accuracy for Random Forest is 0.6596858638743456.
# The f1_score for Random Forest is 0.7086614173228346.

# The accuracy for AdaBoost is 0.6897905759162304.
# The f1_score for AdaBoost is 0.731081081081081.

# The accuracy for KNN is 0.675392670157068.
# The f1_score for KNN is 0.7230941704035874.

# Model Optimization: KNN
n_neighbors = range(1, 30)
param = dict(n_neighbors=n_neighbors)
knn_opt = KNeighborsClassifier()
grid_knn = GridSearchCV(knn_opt, param, scoring='accuracy', cv=10)
t0 = time()
grid_knn.fit(X_train, y_train)
t1 = time()
print("The traning time is: {}".format(t1 - t0))
best_clf = grid_knn.best_estimator_
results = grid_knn.cv_results_
print("The best optimized KNN is: \n\n{}".format(best_clf))
print("\nThe best accuracy score is {}".format(grid_knn.best_score_))
grid_mean_scores = []
for i in results:
    grid_mean_scores.append(i.mean_validation_score)
plt.title('KNN Optimization', fontsize=20)
plt.plot(n_neighbors, grid_mean_scores)
plt.tick_params(labelsize=20)
plt.xlabel('n_neighbors', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.show()

y_pred_knn_opt = best_clf.predict(X_test)
print('The accuracy for optimized KNN is {}.\nThe f1_score for optimized KNN is {}.'
      .format(accuracy_score(y_test, y_pred_knn_opt), f1_score(y_test, y_pred_knn_opt, average='micro')))

# Model Optimizaiton : Decision Tree
max_depth = range(1, 20)
min_samples_split = range(2, 20)
param = dict(max_depth=max_depth, min_samples_split=min_samples_split)
tree_opt = DecisionTreeClassifier(random_state=5)
grid_tree = GridSearchCV(tree_opt, param, scoring='accuracy', cv=10)
t0 = time()
grid_tree.fit(X_train, y_train)
t1 = time()
print("The traning time is: {}".format(t1 - t0))

best_clf = grid_tree.best_estimator_
results = grid_tree.cv_results_
print("The best optimized Decision Tree is: \n\n{}".format(best_clf))
print("\nThe best accuracy score is: {}".format(grid_tree.best_score_))

grid_mean_scores = []
for i in results:
    grid_mean_scores.append(i.mean_validation_score)

count = 1
list_1 = []
list_2 = []
for i in grid_mean_scores:
    list_1.append(i)
    count += 1
    #     print count
    if count == 19:
        list_2.append(np.mean(list_1))
        list_1 = []
        count = 1

plt.title('Decision Tree Optimization', fontsize=20)
plt.plot(max_depth, list_2)
plt.tick_params(labelsize=22)
plt.xlabel('max_depth', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)

y_pred_tree_opt = best_clf.predict(X_test)
print('The accuracy for optimized Decision Tree is: {}.\nThe f1_score for optimized Decision Tree is: {}.' \
      .format(accuracy_score(y_test, y_pred_tree_opt), f1_score(y_test, y_pred_tree_opt, average='micro')))

# Model Optimization: RandomForest
max_depth = range(1, 20)
min_samples_split = range(2, 20)
param = dict(max_depth=max_depth, min_samples_split=min_samples_split)
random_opt = RandomForestClassifier(random_state=5)
grid_random = GridSearchCV(random_opt, param, scoring='accuracy', cv=10)
t0 = time()
grid_random.fit(X_train, y_train)
t1 = time()
print
"The traning time is: {}".format(t1 - t0)
best_clf = grid_random.best_estimator_
results = grid_random.grid_scores_
print
"The best optimized Random Forest is: \n\n{}".format(best_clf)
print
"\nThe best accuracy score is: {}".format(grid_random.best_score_)
list_1 = []
list_2 = []
count = 1
for i in results:
    list_1.append(i[1])
    count += 1
    #     print count
    if count == 19:
        list_2.append(np.mean(list_1))
        list_1 = []
        count = 1
plt.plot(max_depth, list_2)
plt.tick_params(labelsize=20)
plt.title("Random Forest Optimization", fontsize=20)
plt.xlabel("max_depth", fontsize=20)
plt.ylabel("Accuracy", fontsize=20)
plt.show()
y_pred_random_opt = best_clf.predict(X_test)
print('The accuracy for optimized Random Forest is: {}.\nThe f1_score for optimized Random Forest is: {}.'
      .format(accuracy_score(y_test, y_pred_random_opt), f1_score(y_test, y_pred_random_opt, average='micro')))

#  Model Optimization: MLP
hidden_layer_sizes = range(1,10)
learning_rate = ['constant', 'adaptive']
max_iter = [100, 200, 300]
params = dict(hidden_layer_sizes=hidden_layer_sizes, learning_rate=learning_rate, max_iter=max_iter)
mlp_optimal = MLPClassifier(hidden_layer_sizes=10)
grid_mlp = GridSearchCV(mlp_optimal, params, scoring='accuracy', cv=10)
t0 = time()
grid_mlp.fit(X_train, y_train)
t1 = time()
print("Training time is: {}".format(t1-t0))
best_clf = grid_mlp.best_estimator_
results = grid_mlp.grid_scores_
print("The best optimized Decision Tree found: \n\n{}".format(best_clf))
print("\nThe highest accuracy score is: {}".format(grid_mlp.best_score_))
grid_mean_scores = []
for i in results:
    grid_mean_scores.append(i.mean_validation_score)
count = 1
list1 = []
list2 = []
for i in grid_mean_scores:
    list1.append(i)
    count += 1
    if count == 9:
        list2.append(np.mean(list1))
        list1 = []
        count = 1

plt.title('Multi-layer Perceptron Optimization', fontsize=18)
plt.plot(max_depth, list_2)
plt.tick_params(labelsize=20)
plt.xlabel('max_depth', fontsize=18)
plt.ylabel('Accuracy', fontsize=18)
plt.show()

y_pred_mlp_optimal = best_clf.predict(X_test)
print('The accuracy for optimized MLP is: {}.\nThe f1_score for optimized MLP is: {}.'\
    .format(accuracy_score(y_test, y_pred_mlp_optimal),f1_score(y_test, y_pred_mlp_optimal, average='micro')))
