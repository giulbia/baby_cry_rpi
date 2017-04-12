docker run --rm --device /dev/snd -v `pwd`:/opt/baby_cry_rpi -w /opt/baby_cry_rpi -it mikkl/rpi-raspbian-baby-cry-detection:pi2 bash script/run_crying_detection.sh

