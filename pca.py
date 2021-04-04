'''
PCA analysis to find importance of various features.
'''
from train_test_split import my_train_test_split
from sklearn.decomposition import PCA as pca
import numpy as np

x_train, x_test, y_train, y_test, for_pca = my_train_test_split(symbol="MFC")

model = pca().fit(for_pca)
num_components = model.components_.shape[0]

most_important = [np.abs(model.components_[i]).argmax() for i in range(num_components)]
print(most_important)
print(model.components_)