import essentia
import essentia.standard as ess
import numpy as np

from ...annotation.beat.beattracker import *
from ...annotation.downbeat.downbeattracker import *
from ...annotation.key.keyestimation import KeyEstimator
from ...annotation.segmentation.structuralsegmentation import *
from ...annotation.singing.singing_voice_detector import *
from ...annotation.style.theme_descriptor import *
from ...annotation import util as annot_util

import essentia
import essentia.standard


class BaseAnnotationWrapper():

    def is_annotated_in(self, song):
        ''' This function should return true if the annotation exists (on disk or in memory). '''
        raise NotImplementedError()

    def process(self, song):
        raise NotImplementedError()

    def save_annotations_to_file(self, song, annotation_dir, file_handle=None):
        '''
        If implemented, then this function should save the annotations in a certain file,
         and return the path to that file.
        Otherwise, it should raise an exception, which will be caught by the song or songcollection. Then, the song or
         songcollection instance will take care of saving the annotation
         (in a .json file per song, alongside all other annotations for that song).
        '''
        raise NotImplementedError()

    def load_annotations_from_file(self, song, annotation_dir):
        raise NotImplementedError()

    def calculate_supplimentary_features(self, song):
        '''
        If implemented, then this function allows to add more features to the song that are not stored in the annotation files.
        For example, a (tempo, phase) annotation can be expanded and explicit beat features can be added to the song, even though
        these are not stored explicitly on disk.
        '''
        return {}

    def __str__(self):

        return self.__class__.__name__


class BeatAnnotationWrapper(BaseAnnotationWrapper):

    def __init__(self):
        super(BeatAnnotationWrapper, self).__init__()
        self.beattracker = BeatTracker()

    def process(self, s):
        self.beattracker.run(s.audio)
        return {
            'tempo' : self.beattracker.bpm,
            'phase' : self.beattracker.phase,
            'audio_length_samples' : len(s.audio),
        }

    def calculate_supplimentary_features(self, song):
        result = {}

        tempo, phase = song.tempo, song.phase
        spb = 60./tempo #seconds per beat
        result['tempo'] = np.around(song.tempo, 2)
        result['beats'] = (np.arange(phase, (song.audio_length_samples/self.beattracker.SAMPLE_RATE) - spb + phase, spb).astype('single'))
        # TODO (maybe) remove passing around of fft_mag and fft_phase
        #  Currently, the STFT magnitude and phase are passed around between the beat tracker, the onset curve annotation
        #  and the downbeat tracker. This is a bit more efficient as they all reuse the spectrogram to calculate things.
        #  However, this does introduce a dependency between all annotation modules. Maybe this should be rewritten to
        #  avoid this.
        if self.beattracker.fft_mag_1024_512 is not None:
            result['fft_mag_1024_512'] = self.beattracker.fft_mag_1024_512
            result['fft_phase_1024_512'] = self.beattracker.fft_phase_1024_512

        return result

    def is_annotated_in(self, song):
        # return song.hasAnnot(annot_util.ANNOT_BEATS_PREFIX)
        return hasattr(song, 'tempo') and hasattr(song, 'phase') and hasattr(song, 'audio_length_samples')


class OnsetCurveAnnotationWrapper(BaseAnnotationWrapper):

    def __init__(self):
        super(OnsetCurveAnnotationWrapper, self).__init__()

    def process(self, song):

        if song.fft_mag_1024_512 is None or song.fft_phase_1024_512 is None:
            pool = essentia.Pool()
            w = essentia.standard.Windowing(type='hann')
            fft = np.fft.fft
            FRAME_SIZE, HOP_SIZE = 2048, 512
            for frame in essentia.standard.FrameGenerator(song.audio, frameSize=FRAME_SIZE, hopSize=HOP_SIZE):
                pool.add('audio.windowed_frames', w(frame))

            fft_result = fft(pool['audio.windowed_frames']).astype('complex64')
            fft_result_mag = np.absolute(fft_result)
            fft_result_ang = np.angle(fft_result)
            song.fft_mag_1024_512 = fft_result_mag
            song.fft_phase_1024_512 = fft_result_ang

        fft_result_mag = song.fft_mag_1024_512
        fft_result_ang = song.fft_phase_1024_512

        od_hfc = essentia.standard.OnsetDetection(method='hfc')
        pool = essentia.Pool()

        for mag, phase in zip(fft_result_mag, fft_result_ang):
            pool.add('onsets', od_hfc(mag, phase))

        # Normalize and half-rectify onset detection curve
        def adaptive_mean(x, N):
            return np.convolve(x, [1.0] * int(N), mode='same') / N

        novelty_mean = adaptive_mean(pool['onsets'], 16.0)
        novelty_hwr = (pool['onsets'] - novelty_mean).clip(min=0)
        novelty_hwr = novelty_hwr / np.average(novelty_hwr)

        # Save the novelty function, as it is used later for song matching as well
        return {'onset_curve' : novelty_hwr.tolist()}

    def is_annotated_in(self, song):
        return hasattr(song, 'onset_curve')


class DownbeatAnnotationWrapper(BaseAnnotationWrapper):

    def __init__(self):
        super(DownbeatAnnotationWrapper, self).__init__()
        self.dbeattracker = DownbeatTracker()

    def process(self, song):
        # TODO rename names from run and track to something consistent... (__call__?)
        downbeats = self.dbeattracker.track(
            song.audio, song.beats, song.fft_mag_1024_512, song.fft_phase_1024_512, song.onset_curve)
        return {
            'downbeats' : downbeats.tolist(),
        }

    def is_annotated_in(self, song):
        # return song.hasAnnot(annot_util.ANNOT_DOWNB_PREFIX) and song.hasAnnot(annot_util.ANNOT_ODF_HFC_PREFIX)
        return hasattr(song, 'downbeats')


class StructuralSegmentationWrapper(BaseAnnotationWrapper):

    def __init__(self):
        super(StructuralSegmentationWrapper, self).__init__()
        self.structural_segmentator = StructuralSegmentator()

    def process(self, song):
        # TODO rename names from run and track and analyse to something consistent... (__call__?)

        segment_indices, segment_types = self.structural_segmentator.analyse(
            song.audio, song.downbeats, song.onset_curve, song.tempo, )
        return {
            'segment_indices' : segment_indices.tolist(),
            'segment_types' : segment_types.tolist(),
        }

    def is_annotated_in(self, song):
        # return song.hasAnnot(annot_util.ANNOT_SEGMENT_PREFIX)
        return hasattr(song, 'segment_indices') and hasattr(song, 'segment_types')

    def calculate_supplimentary_features(self, song):
        result = {}
        # Some songs have a negative first segment index because the first measure got cropped a bit
        # The song is therefore extended artificially by introducing silence at the beginning
        if song.segment_indices[0] < 0:
            # Calculate the amount of padding
            beat_length_s = 60.0 / song.tempo
            songBeginPaddingSeconds = (-song.segment_indices[0] * 4 * beat_length_s - song.downbeats[0])
            result['songBeginPadding'] = int(songBeginPaddingSeconds * 44100)
            downbeats = [dbeat + songBeginPaddingSeconds for dbeat in song.downbeats]
            result['downbeats'] = [i * 4 * beat_length_s for i in range(-song.segment_indices[0])] + downbeats
            beats = [beat + songBeginPaddingSeconds for beat in song.beats]
            result['beats'] = [i * beat_length_s for i in range(int(beats[0] / beat_length_s))] + beats
            result['onset_curve'] = np.append(np.zeros((1, int(song.songBeginPadding / 512))),
                                         song.onset_curve)  # 512 is hop size for OD curve calculation
            offset = song.segment_indices[0]
            result['segment_indices'] = [idx - offset for idx in song.segment_indices]
        return result


class ReplayGainWrapper(BaseAnnotationWrapper):

    def __init__(self):
        super(ReplayGainWrapper, self).__init__()
        self.replay_gain = essentia.standard.ReplayGain()

    def process(self, song):
        rgain = self.replay_gain(song.audio)
        return {'replaygain' : rgain}

    def is_annotated_in(self, song):
        # return song.title in annot_util.loadCsvAnnotationFile(song.dir_, annot_util.ANNOT_GAIN_PREFIX)
        return hasattr(song, 'replaygain')


class KeyEstimatorWrapper(BaseAnnotationWrapper):

    def __init__(self):
        super(KeyEstimatorWrapper, self).__init__()
        self.key_estimator = KeyEstimator()

    def process(self, song):
        key, scale = self.key_estimator(song.audio)
        return {'key' : key, 'scale' : scale}

    def is_annotated_in(self, song):
        return hasattr(song, 'key') and hasattr(song, 'scale')


class ThemeDescriptorWrapper(BaseAnnotationWrapper):

    def __init__(self):
        super(ThemeDescriptorWrapper, self).__init__()
        self.theme_annotator = ThemeDescriptorEstimator()

    def process(self, song):

        segments_high = [i for i in range(len(song.segment_types)) if song.segment_types[i] == 'H']
        slices = []
        for i in segments_high:
            start_sample = int(44100 * song.downbeats[song.segment_indices[i]])
            end_sample = int(44100 * song.downbeats[song.segment_indices[i + 1]])
            slices.append((start_sample, end_sample))

        song_theme_descriptor = self.theme_annotator(song.audio, slices)
        return {'song_theme_descriptor' : song_theme_descriptor.tolist()}

    def is_annotated_in(self, song):
        return hasattr(song, 'song_theme_descriptor')
        # return song.hasAnnot(annot_util.ANNOT_THEME_DESCR_PREFIX)

    def calculate_supplimentary_features(self, song):
        # When loaded from JSON, it is a normal array, not a numpy array.
        return {'song_theme_descriptor' : np.array(song.song_theme_descriptor)}


class SingingVoiceWrapper(BaseAnnotationWrapper):

    def __init__(self):
        super(SingingVoiceWrapper, self).__init__()
        self.singing_voice_detector = SingingVoiceDetector()

    def process(self, song):
        is_singing = self.singing_voice_detector(song.audio, song.downbeats)
        return {'singing_voice' : is_singing.tolist()}

    def is_annotated_in(self, song):
        return hasattr(song, 'singing_voice')
        # return song.hasAnnot(annot_util.ANNOT_SINGINGVOICE_PREFIX)

    def calculate_supplimentary_features(self, song):
        return {'singing_voice' : np.array(song.singing_voice)}