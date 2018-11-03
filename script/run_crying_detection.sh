#!/usr/bin/env bash

PREDICTION=0
PLAYING=0
CPT=0

PROJECT_PATH=/opt/baby_cry_rpi

function clean_up {

	# Perform program exit housekeeping
	echo ""
	echo "Thank you for using parenting 2.1"
	echo "Good Bye."
	stop_playing
	exit
}

trap clean_up SIGHUP SIGINT SIGTERM

function recording(){
	echo -n "Start Recording..."
	arecord -D plughw:1,0 -d 9 -f S16_LE -c1 -r44100 -t wav ${PROJECT_PATH}/recording/signal_9s.wav
}

function interacting_with_gcp(){
    echo -n "Sending wav to bucket..."
    gsutil cp ${PROJECT_PATH}/recording/signal_9s.wav gs://parenting-3-recording/
    echo -n "Waiting for answer..."
}

function predict() {
	echo -n "Predicting..."
	echo -n "What is the prediction? "
	python ${PROJECT_PATH}/script/make_prediction.py
	PREDICTION=$(cat ${PROJECT_PATH}/prediction/prediction.txt)
	echo "Prediction is $PREDICTION"
}

function start_playing() {
	if [[ ${PLAYING} == 0 ]]; then
		echo "start playing"
                aplay -D plughw:0,0 ${PROJECT_PATH}/lullaby/lullaby_classic.wav
		PLAYING=1
	fi
}

function stop_playing(){
	if [[ ${PLAYING} == 1 ]]; then
		echo "stop playing"
		PLAYING=0
	fi
}

echo "Welcome to Parenting 2.1"
echo ""
while true; do
	recording
	interacting_with_gcp
	predict
	if [[ ${PREDICTION} == 0 ]]; then
		stop_playing
	else
		CPT=$(expr ${CPT} + 1)
		start_playing
	fi
echo "State of the Process PREDICTION = $PREDICTION, PLAYING=$PLAYING, # TIMES MY BABY CRIED=$CPT"
done
clean_up
