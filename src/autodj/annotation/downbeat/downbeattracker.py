# Copyright 2017 Len Vande Veire, IDLab, Department of Electronics and Information Systems, Ghent University
# This file is part of the source code for the Auto-DJ research project, published in Vande Veire, Len, and De Bie, Tijl, "From raw audio to a seamless mix: an artificial intelligence approach to creating an automated DJ system.", 2018 (submitted)
# Released under AGPLv3 license.

import numpy as np
import os
from . import features
from .features import loudness, mfcc, onsetflux, onsetcsd, onsethfc

from sklearn.externals import joblib	# Model persistence

feature_modules = [features.loudness, features.mfcc,
				   features.onsetflux, features.onsetcsd,
				   features.onsethfc]

class DownbeatTracker:
	'''
		Detects the downbeat locations given the beat locations and audio
	'''
	def __init__(self):
		# Load the feature modules	
		self.model = joblib.load(os.path.join(os.path.abspath(os.path.dirname(__file__)),'model.pkl')) 
		self.scaler = joblib.load(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'scaler.pkl'))
	
	def trimAudio(self, audio, beats):
			
		beats = np.array(beats) * 44100 # Beats in samples
		# Calculate beatwise RMS
		rms = []
		for i in range(len(beats) - 1):
			rms.append(np.sqrt(np.mean(np.square(audio[int(beats[i]) : int(beats[i+1])]))))
		def adaptive_mean(x, N):
			return np.convolve(x, [1.0]*int(N), mode='same')/N
		rms_adaptive = adaptive_mean(rms, 4)
		rms_adaptive_max = max(rms_adaptive)
		
		# Determine cut positions
		start, end, ratiox = 0,0,0
		ratios = [.9, .8, .7, .6, .5, .4, .3, .2, .1]
		for ratio in ratios:
			# Extract beginning and end
			for i in range(len(rms)):
				if rms[i] > ratio*rms_adaptive_max:
					start = i
					break # Go to trail cutting
			for i in range(len(rms)):
				if rms[len(rms) - i - 1] > ratio*rms_adaptive_max:
					end = len(rms) - i - 1
					break # Go to check if file is not too small now
			# If beginning and end cut not more than 50% of the song, then it is ok
			if end - start >= len(rms) * 0.4:
				ratiox = ratio
				break
				
		return start, end
		
	def getFeaturesForAudio(self, input_features):
		
		FRAME_INDEXER_MIN = 4
		FRAME_INDEXER_MAX = len(input_features['beats']) - 9 # -9 instead of -8 to prevent out-of-bound in featureLoudness
		trim_start_beat, trim_end_beat = self.trimAudio(input_features['audio'], input_features['beats'])
		indexer_start = max(FRAME_INDEXER_MIN, trim_start_beat)
		indexer_end = min(FRAME_INDEXER_MAX, trim_end_beat)
		frame_indexer = range(indexer_start, indexer_end) 
				
		# Calculate the features on every frame in the audio
		features_cur_file = None
		for module in feature_modules:
			absolute_feature_submatrix = module.feature_allframes(input_features, frame_indexer)
			if features_cur_file is None:
				features_cur_file = absolute_feature_submatrix
			else:
				features_cur_file = np.append(features_cur_file, absolute_feature_submatrix, axis=1)
		return features_cur_file, trim_start_beat

	def track(self, audio, beats, fft_mag, fft_phase, onset_curve):
		'''
		Track the downbeats of the given audio file

		:param audio: Numpy array in mono of the audio, sampled at 44100 Hz.
		:param beats: Beats of the music, as timestamps in seconds.
		:param fft_mag: The STFT magnitude spectrum; should be calculated at a hop size of 512 and frame length 2048.
		:param fft_phase: The STFT phase spectrum; should be calculated at a hop size of 512 and frame length 2048.
		:param onset_curve: Onset curve feature, calculated with a hop size of 512 over the input audio.

		:return: Downbeats of the music, as timestamps in seconds. Always is a subset of the beats.
		'''

		input_features = {
			'audio' : audio,
			'beats' : beats,
			'fft_mag' : fft_mag,
			'fft_ang' : fft_phase,
			'onset_curve' : onset_curve,
		}

		features, trim_start_beat = self.getFeaturesForAudio(input_features)
		probas = self.model.predict_log_proba(features)
			
		sum_log_probas = np.array([[0,0,0,0]], dtype='float64')		
		permuted_row = [0] * 4
		
		for i, j, row in zip(range(len(probas)), np.array(range(len(probas))) % 4, probas):
			permuted_row[:4-j] = row[j:]
			permuted_row[4-j:] = row[:j] # element i of permuted_row (i = 0,1,2,3) corresponds to the TRAJECTORY over the song starting with downbeat 0, 1, 2, 3
			perm_row_np = np.array([permuted_row])
			sum_log_probas = sum_log_probas + permuted_row
		
		# The audio got trimmed, so make sure the downbeat index is offset by the correct amount!
		downbeatIndex = ((4-np.argmax(sum_log_probas)) + trim_start_beat) % 4
		
		return beats[downbeatIndex::4]
