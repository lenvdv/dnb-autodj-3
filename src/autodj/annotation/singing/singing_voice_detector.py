import essentia
import essentia.standard as ess
import numpy as np
import os
import scipy
from sklearn.externals import joblib

class SingingVoiceDetector:

    def __init__(self, ):

        basepath = os.path.dirname(os.path.abspath(__file__))
        self.singing_model  = joblib.load(os.path.join(basepath, 'singingvoice_model.pkl'))
        self.singing_scaler = joblib.load(os.path.join(basepath, 'singingvoice_scaler.pkl'))

    def _calculate_features_for_audio(self, audio):

        FRAME_SIZE, HOP_SIZE = 2048, 1024
        features = []

        low_f = 100
        high_f = 7000

        w = ess.Windowing(type='hann')
        spec = ess.Spectrum(size=FRAME_SIZE)
        mfcc = ess.MFCC(lowFrequencyBound=low_f, highFrequencyBound=high_f)
        spectralContrast = ess.SpectralContrast(lowFrequencyBound=low_f, highFrequencyBound=high_f)
        pool = essentia.Pool()

        for frame in ess.FrameGenerator(audio, frameSize=FRAME_SIZE, hopSize=HOP_SIZE):
            frame_spectrum = spec(w(frame))
            spec_contrast, spec_valley = spectralContrast(frame_spectrum)
            mfcc_bands, mfcc_coeff = mfcc(frame_spectrum)
            pool.add('spec_contrast', spec_contrast)
            pool.add('spec_valley', spec_valley)
            pool.add('mfcc_coeff', mfcc_coeff)

        def add_moment_features(array):
            avg = np.average(array, axis=0)
            std = np.std(array, axis=0)
            skew = scipy.stats.skew(array, axis=0)
            deltas = array[1:, :] - array[:-1, :]
            avg_d = np.average(deltas, axis=0)
            std_d = np.std(deltas, axis=0)

            features.extend(avg)
            features.extend(std)
            features.extend(skew)
            features.extend(avg_d)
            features.extend(std_d)

        add_moment_features(pool['spec_contrast'])
        add_moment_features(pool['spec_valley'])
        add_moment_features(pool['mfcc_coeff'])

        return np.array(features, dtype='single')

    def __call__(self, audio, downbeats):

        features = []
        for dbeat_idx in range(len(downbeats) - 1):
            start = int(downbeats[dbeat_idx] * 44100)
            stop = int(downbeats[dbeat_idx + 1] * 44100)
            if start >= len(audio):
                break
            features.append(self._calculate_features_for_audio(audio[start:stop]))
        X = np.array(features)

        return np.array(self.singing_model.decision_function(self.singing_scaler.transform(X)), dtype='single')