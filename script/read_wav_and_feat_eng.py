import librosa
import numpy as np
from librosa.feature import zero_crossing_rate, mfcc, spectral_centroid, spectral_rolloff, spectral_bandwidth, rmse


def read_wav_and_feat_eng(data, context):
    """Background Cloud Function to be triggered by Cloud Storage.
       This generic function logs relevant data when a file is changed.

    Args:
        data (dict): The Cloud Functions event payload.
        context (google.cloud.functions.Context): Metadata of triggering event.
    Returns:
        None; the output is written to Stackdriver Logging
    """

    print('Event ID: {}'.format(context.event_id))
    print('Event type: {}'.format(context.event_type))
    print('Bucket: {}'.format(data['bucket']))
    print('File: {}'.format(data['name']))
    print('Metageneration: {}'.format(data['metageneration']))
    print('Created: {}'.format(data['timeCreated']))
    print('Updated: {}'.format(data['updated']))


def read_audio_file(file_name):
    """
    Read audio file using librosa package. librosa allows resampling to desired sample rate and convertion to mono.

    :return:
    * play_list: a list of audio_data as numpy.ndarray. There are 5 overlapping signals, each one is 5-second long.
    """

    play_list = list()

    for offset in range(5):
        audio_data, _ = librosa.load(file_name, sr=44100, mono=True, offset=offset, duration=5.0)
        play_list.append(audio_data)

    return play_list


def feature_engineer(audio_data):
    """
    Extract features using librosa.feature.

    Each signal is cut into frames, features are computed for each frame and averaged [median].
    The numpy array is transformed into a data frame with named columns.

    :param audio_data: the input signal samples with frequency 44.1 kHz
    :return: a numpy array (numOfFeatures x numOfShortTermWindows)
    """

    zcr_feat = compute_librosa_features(audio_data=audio_data, feat_name='zero_crossing_rate')
    rmse_feat = compute_librosa_features(audio_data=audio_data, feat_name='rmse')
    mfcc_feat = compute_librosa_features(audio_data=audio_data, feat_name= 'mfcc')
    spectral_centroid_feat = compute_librosa_features(audio_data=audio_data, feat_name='spectral_centroid')
    spectral_rolloff_feat = compute_librosa_features(audio_data=audio_data, feat_name='spectral_rolloff')
    spectral_bandwidth_feat = compute_librosa_features(audio_data=audio_data, feat_name='spectral_bandwidth')

    concat_feat = np.concatenate((zcr_feat,
                                  rmse_feat,
                                  mfcc_feat,
                                  spectral_centroid_feat,
                                  spectral_rolloff_feat,
                                  spectral_bandwidth_feat
                                  ), axis=0)

    return np.mean(concat_feat, axis=1, keepdims=True).transpose()


def compute_librosa_features(audio_data, feat_name):
    """
    Compute feature using librosa methods

    :param audio_data: signal
    :param feat_name: feature to compute
    :return: np array
    """
    rate = 44100   # All recordings in ESC are 44.1 kHz
    frame = 512    # Frame size in samples

    if feat_name == 'zero_crossing_rate':
        return zero_crossing_rate(y=audio_data, hop_length=frame)
    elif feat_name == 'rmse':
        return rmse(y=audio_data, hop_length=frame)
    elif feat_name == 'mfcc':
        return mfcc(y=audio_data, sr=rate, n_mfcc=13)
    elif feat_name == 'spectral_centroid':
        return spectral_centroid(y=audio_data, sr=rate, hop_length=frame)
    elif feat_name == 'spectral_rolloff':
        return spectral_rolloff(y=audio_data, sr=rate, hop_length=frame, roll_percent=0.90)
    elif feat_name == 'spectral_bandwidth':
        return spectral_bandwidth(y=audio_data, sr=rate, hop_length=frame)
