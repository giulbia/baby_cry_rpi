import numpy as np
import soundfile as sf
import io
import json
from google.cloud import storage
import googleapiclient.discovery
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

    print('gs://{}/{}'.format(data['bucket'], data['name']))

    BUCKET = data['bucket']

    # Create a Cloud Storage client.
    gcs = storage.Client()

    # Get the bucket that the file will be uploaded to.
    bucket = gcs.get_bucket(BUCKET)

    # READ AUDIO FILE

    # specify a filename
    file_name = data['name']

    # read a blob
    blob = bucket.blob(file_name)
    file_as_string = blob.download_as_string()

    play_list = read_audio_file(file_as_string)

    print("\nREADING DONE\n")

    print(len(play_list))
    print([play_list[i].size for i in range(len(play_list))])

    # FEATURE ENGINEERING
    play_list_processed = list()

    for signal in play_list:
        play_list_processed.append(feature_engineer(signal))

    print("\nFEATURE ENGINEERING DONE\n")

    # CREATE JSON WITH FEATURES
    play_list_json = features_to_json(play_list_processed)

    print("\nDATA READY FOR PREDICTION: JSON FORMAT DONE\n")

    # GET PREDICTIONS WITH ML ENGINE
    predictions = predict_json(project="parenting-3", model="model", instances=json.loads(play_list_json)["instances"],
                               version="v11")

    print("\nPREDICTION DONE\n")
    print(predictions)

    # MAJORITY VOTE


def read_audio_file(file_as_string):
    """
    Read audio file using librosa package. librosa allows resampling to desired sample rate and convertion to mono.

    :return:
    * play_list: a list of audio_data as numpy.ndarray. There are 5 overlapping signals, each one is 5-second long.
    """

    # convert the string to bytes and then finally to audio samples as floats

    play_list = list()
    sr = 44100
    n_samples = sr * 5

    for offset in range(5):
        audio_data, _ = sf.read(file=io.BytesIO(file_as_string), start=offset*sr, stop=n_samples+offset*sr)
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

    return np.mean(concat_feat, axis=1, keepdims=True).transpose()[0].tolist()


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


def features_to_json(processed_data):

    #Create dictionary
    dic = dict()
    dic['instances'] = processed_data

    #Dump data dict to jason
    return json.dumps(dic)


def predict_json(project, model, instances, version=None):
    """
    Send json data to a deployed model for prediction.
    Args:
        project (str): project where the Cloud ML Engine Model is deployed.
        model (str): model name.
        instances ([[float]]): List of input instances, where each input
           instance is a list of floats.
        version: str, version of the model to target.
    Returns:
        Mapping[str: any]: dictionary of prediction results defined by the
            model.
    """

    # Create the ML Engine service object.
    # To authenticate set the environment variable
    # GOOGLE_APPLICATION_CREDENTIALS=<path_to_service_account_file>

    service = googleapiclient.discovery.build('ml', 'v1')
    name = 'projects/{}/models/{}'.format(project, model)

    if version is not None:
        name += '/versions/{}'.format(version)

    response = service.projects().predict(
        name=name,
        body={'instances': instances}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    return response['predictions']
