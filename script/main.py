import numpy as np
import io
import json
import re
import base64

import soundfile as sf
from librosa.feature import zero_crossing_rate, mfcc, spectral_centroid, spectral_rolloff, spectral_bandwidth, rmse

from google.cloud import storage
import googleapiclient.discovery
from google.oauth2 import service_account
from googleapiclient.errors import HttpError


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

    # READ AUDIO FILE

    # Get the bucket where the wav file will be uploaded to.
    bucket = gcs.get_bucket(BUCKET)

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
    num_predictions = list()
    for predicted_class in predictions:
        num_predictions.append(is_baby_cry(predicted_class))

    final_prediction = majority_vote(num_predictions)

    print("\nFinal prediction is: {}.\n".format(final_prediction))

    # SAVE PREDICTION AS TEXT FILE IN RESERVED BUCKET

    bucket_name = "parenting-3-prediction"

    pred_bucket = gcs.get_bucket(bucket_name)

    blob = pred_bucket.blob("prediction.txt")

    blob.upload_from_string(str(final_prediction))

    # Make the file publicly accessible so that a device can download it
    blob.make_public()

    print('File {} is publicly accessible at {}'.format(blob.name, blob.public_url))

    print('Sending file location to device...')
    send_to_device(bucket_name="parenting-3-prediction",
                   gcs_file_name="prediction.txt",
                   destination_file_name="prediction.txt",
                   project_id="parenting-3",
                   registry_id="raspberry-pi",
                   device_id="rpi",
                   service_account_json=None,
                   cloud_region="europe-west1")


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
    """
    Put features into json format
    :param processed_data: list of list of features [[feat_1, feat_2, feat_3, ...], [feat_1, feat_2, feat_3, ...]]
    :return: json {"instances": [[feat_1, feat_2, feat_3, ...], [feat_1, feat_2, feat_3, ...]]}
    """

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


def is_baby_cry(predicted_class):
    """
    Make prediction with trained model

    :param predicted_class: str of the kind '004 - Baby cry'
    :return: 1 (it's baby cry); 0 (it's not a baby cry)
    """

    match = re.search('Crying baby', predicted_class)

    if match:
        return 1
    else:
        return 0


def majority_vote(prediction_list):
    """
    Overall prediction

    :param prediction_list: numeric list of 0's and 1's

    :return: 1 if more than half predictions are 1s
    """

    if sum(prediction_list) > len(prediction_list)/2.0:
        return 1
    else:
        return 0


def get_client(service_account_json):
    """Returns an authorized API client by discovering the IoT API and creating
    a service object using the service account credentials JSON."""
    api_scopes = ['https://www.googleapis.com/auth/cloud-platform']
    api_version = 'v1'
    discovery_api = 'https://cloudiot.googleapis.com/$discovery/rest'
    service_name = 'cloudiotcore'

    credentials = service_account.Credentials.from_service_account_file(service_account_json)
    scoped_credentials = credentials.with_scopes(api_scopes)

    discovery_url = '{}?version={}'.format(
        discovery_api, api_version)

    return googleapiclient.discovery.build(
        service_name,
        api_version,
        discoveryServiceUrl=discovery_url,
        credentials=scoped_credentials)


def send_to_device(
        bucket_name,
        gcs_file_name,
        destination_file_name,
        project_id,
        cloud_region,
        registry_id,
        device_id,
        service_account_json):
    """Sends the configuration to the device."""

    client = get_client(service_account_json)

    device_name = 'projects/{}/locations/{}/registries/{}/devices/{}'.format(
        project_id, cloud_region, registry_id, device_id)

    config_data = ({
        'bucket_name': bucket_name,
        'gcs_file_name': gcs_file_name,
        'destination_file_name': destination_file_name
    })

    config_data_json = json.dumps(config_data, separators=(',', ': '))

    body = {
        # The device configuration specifies a version to update, which
        # can be used to avoid having configuration updates race. In this
        # case, you use the special value of 0, which tells Cloud IoT Core to
        # always update the config.
        'version_to_update': 0,
        # The data is passed as raw bytes, so we encode it as base64. Note
        # that the device will receive the decoded string, and so you do not
        # need to base64 decode the string on the device.
        'binary_data': base64.b64encode(config_data_json.encode('utf-8'))
            .decode('ascii')
    }

    request = client.projects().locations().registries().devices(
    ).modifyCloudToDeviceConfig(name=device_name, body=body)

    try:
        request.execute()
        print('Successfully sent file to device: {}'.format(device_id))
    except HttpError as e:
        # If the server responds with an HtppError, most likely because
        # the config version sent differs from the version on the
        # device, log it here.
        print('Error executing ModifyCloudToDeviceConfig: {}'.format(e))
