# Copyright 2017 Len Vande Veire, IDLab, Department of Electronics and Information Systems, Ghent University
# This file is part of the source code for the Auto-DJ research project, published in Vande Veire, Len, and De Bie, Tijl, "From raw audio to a seamless mix: an artificial intelligence approach to creating an automated DJ system.", 2018 (submitted)
# Released under AGPLv3 license.

import numpy as np
import sys
from essentia import *
from essentia.standard import Spectrum, Windowing, CartesianToPolar, OnsetDetection, FFT, FrameGenerator

class BeatTracker:
	'''Detects the BPM, phase and locations of the beats for the given input audio'''
	
	def __init__(self, minBpm = 160.0, maxBpm = 190.0, stepBpm = 0.01, FRAME_SIZE = 1024, HOP_SIZE = 512, SAMPLE_RATE = 44100.0):
		self.minBpm = minBpm
		self.maxBpm = maxBpm
		self.stepBpm = stepBpm
		self.FRAME_SIZE = FRAME_SIZE
		self.HOP_SIZE = HOP_SIZE
		self.SAMPLE_RATE = SAMPLE_RATE
		
		self.bpm = None
		self.phase = None
		self.beats = None
		
		self.onset_curve = None
		self.fft_mag_1024_512 = None # FFT magnitude on windows of 1024 samples, 512 hop size
		self.fft_phase_1024_512 = None # FFT phase on windows of 1024 samples, 512 hop size
		
	def getBpm(self):
		'''Returns the BPM for the analysed audio.
		
		:returns Beats per minute
		'''
		if self.bpm is None:
			raise Exception('No BPM detected yet, you must run the BeatTracker first!')
		return self.bpm
		
	def getPhase(self):
		'''Returns the beat phase for the analysed audio.
		
		:returns Phase in seconds
		'''
		if self.phase is None:
			raise Exception('No phase detected yet, you must run the BeatTracker first!')
		return self.phase
		
	def getBeats(self):
		'''Returns the beat locations for the analysed audio. These beats are all equidistant (constant BPM is assumed).
		
		:returns Array of beat locations in seconds
		'''
		if self.beats is None:
			raise Exception('No beats detected yet, you must run the BeatTracker first!')
		return self.beats
		
	def getOnsetCurve(self):
		'''Returns an array of onset values locations for the analysed audio.
		
		:returns Onset detection curve as a float array
		'''
		if self.onset_curve is None:
			raise Exception('No onset detection curve calculated yet, you must run the BeatTracker first!')
		return self.onset_curve

	def run(self, audio):

		# Calculate the melflux onset detection function

		pool = Pool()
		w = Windowing(type='hann')
		fft = np.fft.fft
		od_flux = OnsetDetection(method='melflux')

		for frame in FrameGenerator(audio, frameSize=self.FRAME_SIZE, hopSize=self.HOP_SIZE):
			pool.add('audio.windowed_frames', w(frame))

		fft_result = fft(pool['audio.windowed_frames']).astype('complex64')
		fft_result_mag = np.absolute(fft_result)
		fft_result_ang = np.angle(fft_result)
		self.fft_mag_1024_512 = fft_result_mag
		self.fft_phase_1024_512 = fft_result_ang

		for mag, phase in zip(fft_result_mag, fft_result_ang):
			pool.add('onsets.complex', od_flux(mag, phase))

		odf = pool['onsets.complex']

		# Given the ODF, calculate the tempo and the phase
		tempo, tempo_curve, phase, phase_curve = BeatTracker.get_tempo_and_phase_from_odf(odf, self.HOP_SIZE)

		# Calculate the beat annotations
		spb = 60./tempo #seconds per beat
		beats = (np.arange(phase, (np.size(audio)/self.SAMPLE_RATE) - spb + phase, spb).astype('single'))
		
		# Store all the results
		self.bpm = tempo
		self.phase = phase
		self.beats = beats
		self.onset_curve = BeatTracker.hwr(pool['onsets.complex'])

	@staticmethod
	def autocorr(x):
		result = np.correlate(x, x, mode='full')
		return result[result.size // 2:]

	@staticmethod
	def adaptive_mean(x, N):
		return np.convolve(x, [1.0] * int(N), mode='same') / N

	@staticmethod
	def frames_per_beat(beats_per_minute, sample_rate, hop_size):
		return (60.0 * sample_rate) / (hop_size * beats_per_minute)

	@staticmethod
	def hwr(x, N=16):
		x_mean = BeatTracker.adaptive_mean(x, N)
		x_hwr = (x - x_mean).clip(min=0)
		return x_hwr

	@staticmethod
	def sum_curve_at_intervals(x, valid_hop_sizes, valid_offsets):
		# valid_phases = np.arange(0.0, 60.0 / bpm, 0.001)  # Valid phases in SECONDS
		activities = np.zeros((len(valid_hop_sizes), len(valid_offsets)))
		for idx_hop, hop_size in enumerate(valid_hop_sizes):
			for idx_offset, offset in enumerate(valid_offsets):
				# Discard last value to prevent reading beyond array (last value rounded up for example)
				frames = (np.round(np.arange(offset, np.size(x), hop_size)).astype('int'))[:-1]
				activities[idx_hop, idx_offset] = np.sum(x[frames]) / np.size(frames)
		return activities.squeeze() / np.max(activities)

	@staticmethod
	def get_tempo_and_phase_from_odf(x, odf_hop_size, min_bpm=160.0, max_bpm=185.0, step_bpm=0.01,
									 step_phase_s=0.001, phase_level='beat', sr=44100):
		# Calculate the tempo of a piece of music, given it's onset detection curve x
		# Default parameter values are a reasonable range for drum and bass music
		#
		# One joint method for tempo and phase because autocorrelation is shared

		x_hwr = BeatTracker.hwr(x)
		x_corr = BeatTracker.autocorr(x_hwr)

		tempo_range_bpm = np.arange(min_bpm, max_bpm, step_bpm)
		tempo_range_odf = sr * 60.0 / (tempo_range_bpm * odf_hop_size)

		tempo_detection_curve_ = BeatTracker.sum_curve_at_intervals(x_corr, tempo_range_odf, [0.0, ])
		tempo_idx = np.argmax(tempo_detection_curve_)
		tempo_bpm, tempo_odf = tempo_range_bpm[tempo_idx], tempo_range_odf[tempo_idx]

		if phase_level == 'beat':
			phase_level_mult = 1
		else:
			phase_level_mult = 8
		phase_range_s = np.arange(0, phase_level_mult * 60.0 / tempo_bpm, step_phase_s)
		phase_range_odf = sr * phase_range_s / odf_hop_size

		phase_detection_curve = BeatTracker.sum_curve_at_intervals(
			x_hwr, [phase_level_mult * tempo_odf], phase_range_odf)
		phase_idx = np.argmax(phase_detection_curve)
		phase_s = phase_range_s[phase_idx]

		return tempo_bpm, tempo_detection_curve_, phase_s, phase_detection_curve
