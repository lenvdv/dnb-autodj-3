from sklearn.externals import joblib
import os

theme_pca      = joblib.load(os.path.join('song_theme_pca_model_2.pkl'))
print(theme_pca)

import pickle

with open('song_theme_pca_model_pickle.pkl', 'wb') as f:
	pickle.dump(theme_pca, f)