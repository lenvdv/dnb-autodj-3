from sklearn.externals import joblib
import os
import pickle

with open('song_theme_pca_model_pickle.pkl', 'rb') as f:
	theme_pca = pickle.load(f,  encoding='latin1')


print(theme_pca)

joblib.dump(theme_pca, os.path.join('song_theme_pca_model_python3.pkl'))
