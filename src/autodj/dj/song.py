# Copyright 2017 Len Vande Veire, IDLab, Department of Electronics and Information Systems, Ghent University
# This file is part of the source code for the Auto-DJ research project, published in Vande Veire, Len, and De Bie, Tijl, "From raw audio to a seamless mix: an artificial intelligence approach to creating an automated DJ system.", 2018 (submitted)
# Released under AGPLv3 license.

from ..annotation.util import *
from .annotators.wrappers import *
from .timestretching import * # Data augmentation
import json

import logging
logger = logging.getLogger('colorlogger')

def normalizeAudioGain(audio, rgain, target = -10):
	# Normalize to a level of -16, which is often value of signal here
	factor = 10**((-(target - rgain)/10.0) / 2.0) # Divide by 2 because we want sqrt (amplitude^2 is energy)
	audio *= factor	
	return audio

class Song:

	def __init__(self, path_to_file, annotation_modules = None):

		self.filepath = path_to_file
		self.dir_, self.title = os.path.split(os.path.abspath(path_to_file))
		self.title, self.extension = os.path.splitext(self.title)
		self.dir_annot = os.path.join(self.dir_, ANNOT_SUBDIR)
		
		if not os.path.isdir(self.dir_annot):
			logger.debug('Creating annotation directory : ' + self.dir_annot)
			os.mkdir(self.dir_annot)
		
		self.audio = None

		self.songBeginPadding = 0	# Number of samples to pad the song with, if first segment index < 0
		# Features shared over different components (e.g. beat and downbeat tracking)
		self.fft_phase_1024_512 = None
		self.fft_mag_1024_512 = None

		self.annotation_modules = annotation_modules if annotation_modules is not None else []
		self.json_features = {}
		self.json_file_path = os.path.join(self.dir_annot, f'{self.title}.json')


	def _add_features_to_song(self, dict_):
		for k,v in dict_.items():
			setattr(self, k, v)

	def getSegmentType(self, dbeat):
		''' Get the segment type ('H' or 'L') of the segment the dbeat falls in '''
		for i in range(len(self.segment_types)-1):
			if self.segment_indices[i] <= dbeat and self.segment_indices[i+1] > dbeat:
				return self.segment_types[i]
		raise Exception('Invalid downbeat ' + str(dbeat) + ', should be between ' + str(self.segment_indices[0]) + ', ' + str(self.segment_indices[-1]))

	def hasAnnot(self, prefix):
		return os.path.isfile(pathAnnotationFile(self.dir_, self.title, prefix))

	def hasAllAnnot(self):
		'''
		Check if this file has annotation files
		'''
		for annot_module_wrapper in self.annotation_modules:
			if not annot_module_wrapper.is_annotated_in(self):
				return False
		return True

	def annotate(self):
		# This doesn't store the annotations and audio in memory yet, this would cost too much memory: writes the annotations to disk and evicts the data from main memory until the audio is loaded for playback
		loader = MonoLoader(filename = os.path.join(self.dir_, self.title + self.extension))
		self.audio = loader()

		# ----------------------------------------
		# Load all annotations that already exist
		# ----------------------------------------
		self.open()

		# ----------------------------------------
		# Calculate all annotations that don't exist yet, and save them
		# ----------------------------------------
		for annot_module_wrapper in self.annotation_modules:
			if not annot_module_wrapper.is_annotated_in(self):
				logger.debug(f'Calculating {annot_module_wrapper} annotations of {self.title}')
				calculated_features = annot_module_wrapper.process(self)
				self._add_features_to_song(calculated_features)
				additional_features = annot_module_wrapper.calculate_supplimentary_features(self)
				self._add_features_to_song(additional_features)

				# Save the new features
				try:
					annot_module_wrapper.save_annotations_to_file(self, self.dir_annot)  # Features are passed using song
				except NotImplementedError:
					self.json_features.update(calculated_features)

			else:
				additional_features = annot_module_wrapper.calculate_supplimentary_features(self)
				self._add_features_to_song(additional_features)

		self._save_json_features(self.json_features, self.json_file_path)
		self.json_features = None

		# Clean up memory: do not keep annotations or audio in RAM until song is actually used
		self.close()

	def _save_json_features(self, data, path_to_file):
		with open(path_to_file, 'w+') as jsonfile:
			json.dump(data, jsonfile)

	def _load_json_features(self, path_to_file):
		with open(path_to_file, 'r+') as jsonfile:
			return json.load(jsonfile)

	def open(self):

		try:
			self.json_features = self._load_json_features(self.json_file_path)
			self._add_features_to_song(self.json_features)
		except FileNotFoundError as e:
			print(e)
			pass

		# Load all custom song annotations
		for annot_module_wrapper in self.annotation_modules:
			if annot_module_wrapper.is_annotated_in(self):
				try:
					loaded_features = annot_module_wrapper.load_annotations_from_file(self, self.dir_annot)
					self._add_features_to_song(loaded_features)
				except NotImplementedError:
					pass # Not implemented means that it has already been loaded from the json file
				except FileNotFoundError:  # TODO should this be silently ignored?
					pass

				additional_features = annot_module_wrapper.calculate_supplimentary_features(self)
				self._add_features_to_song(additional_features)

			
	def openAudio(self):
		filename = os.path.join(self.dir_, self.title + self.extension)
		audio, sr = librosa.load(filename, sr=44100, mono=False)
		audio = audio.astype('single')
		self.audio_left, self.audio_right = audio[0, :], audio[1, :]
		self.audio = librosa.to_mono(audio)
		self.audio = normalizeAudioGain(self.audio, self.replaygain)
		if self.songBeginPadding > 0:
			self.audio = np.append(np.zeros((1,self.songBeginPadding),dtype='single'), self.audio)
	
	def closeAudio(self):
		# Garbage collector will take care of this later on
		self.audio = None
		
	# Close the audio file and reset all buffers to None
	def close(self):
		self.audio = None
		self.beats = None
		self.onset_curve = None
		self.tempo = None
		self.downbeats = None
		self.segment_indices = None
		self.segment_types = None
		self.replaygain = None
		self.key = None
		self.scale = None
		self.spectralContrast = None
		self.fft_phase_1024_512 = None
		self.fft_mag_1024_512 = None
		
	def getOnsetCurveFragment(self, start_beat_idx, stop_beat_idx, target_tempo = 175):
		# Cut out a section of the onset detection curve with beat granularity
		# Stretch the onset curve to the target BPM (175 by default) to ensure that
		# curves of different songs are comparable
		
		# Parameters of the onset detection function calculation
		HOP_SIZE = 512
		SAMPLE_RATE = 44100
		start_frame = int(SAMPLE_RATE * self.beats[start_beat_idx] / HOP_SIZE)
		stop_frame = int(SAMPLE_RATE * self.beats[stop_beat_idx] / HOP_SIZE)
		return self.onset_curve[start_frame:stop_frame]
		
	def markedForAnnotation(self):
		# Songs can be marked for manual annotation fixing. This is stored in this annotation file
		return self.title in loadCsvAnnotationFile(self.dir_, ANNOT_MARKED_PREFIX)
		
	def markForAnnotation(self):
		writeCsvAnnotation(self.dir_, ANNOT_MARKED_PREFIX, self.title, 1)
		
	def unmarkForAnnotation(self):
		deleteCsvAnnotation(self.dir_, ANNOT_MARKED_PREFIX, self.title)
		
