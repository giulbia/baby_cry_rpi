# -*- coding: utf-8 -*-

import argparse
import os
import pickle
import sys
import warnings

egg_path = '{}/../lib/baby_cry_detection-1.1-py2.7.egg'.format(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(egg_path)

from baby_cry_detection.rpi_methods import Reader
from baby_cry_detection.rpi_methods.baby_cry_predictor import BabyCryPredictor
from baby_cry_detection.rpi_methods.feature_engineer import FeatureEngineer
from baby_cry_detection.rpi_methods.majority_voter import MajorityVoter


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path_data',
                        default='{}/../recording/'.format(os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument('--load_path_model',
                        default='{}/../model/'.format(os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument('--save_path',
                        default='{}/../prediction/'.format(os.path.dirname(os.path.abspath(__file__))))

    # Arguments
    args = parser.parse_args()
    load_path_data = os.path.normpath(args.load_path_data)
    load_path_model = os.path.normpath(args.load_path_model)
    save_path = os.path.normpath(args.save_path)

    ####################################################################################################################
    # READ RAW SIGNAL
    ####################################################################################################################

    # Read signal
    file_name = 'signal_9s.wav'       # only one file in the folder
    file_reader = Reader(os.path.join(load_path_data, file_name))
    play_list = file_reader.read_audio_file()

    ####################################################################################################################
    # iteration
    ####################################################################################################################

    # iterate on play_list for feature engineering and prediction

    ####################################################################################################################
    # FEATURE ENGINEERING
    ####################################################################################################################

    # Feature extraction
    engineer = FeatureEngineer()

    play_list_processed = list()

    for signal in play_list:
        tmp = engineer.feature_engineer(signal)
        play_list_processed.append(tmp)

    ####################################################################################################################
    # MAKE PREDICTION
    ####################################################################################################################

    # https://stackoverflow.com/questions/41146759/check-sklearn-version-before-loading-model-using-joblib
    with warnings.catch_warnings():
      warnings.simplefilter("ignore", category=UserWarning)

      with open((os.path.join(load_path_model, 'model.pkl')), 'rb') as fp:
          model = pickle.load(fp)

    predictor = BabyCryPredictor(model)

    predictions = list()

    for signal in play_list_processed:
        tmp = predictor.classify(signal)
        predictions.append(tmp)

    ####################################################################################################################
    # MAJORITY VOTE
    ####################################################################################################################

    majority_voter = MajorityVoter(predictions)
    majority_vote = majority_voter.vote()

    ####################################################################################################################
    # SAVE
    ####################################################################################################################

    # Save prediction result
    with open(os.path.join(save_path, 'prediction.txt'), 'wb') as text_file:
        text_file.write("{0}".format(majority_vote))

if __name__ == '__main__':
    main()
