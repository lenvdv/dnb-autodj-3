from setuptools import setup, find_packages

setup(
	name='autodj',
	version='0.1',
	description='Automatic DJ program for drum and bass music',
	url='https://bitbucket.org/ghentdatascience/dj/',
	author='Len Vande Veire',
	packages=find_packages('src'),
	package_dir={'': 'src'},
	install_requires=[
		'colorlog',
		'Essentia',
		'joblib',
		'librosa',
		'numpy',
		'pyAudio',
		'scikit-learn==0.20.3',
		'scipy',
		'yodel',
	],
	include_package_data=True,
)
