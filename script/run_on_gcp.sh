docker run --rm --device /dev/snd --volumes-from gcloud-config -v `pwd`:/opt/baby_cry_rpi -w /opt/baby_cry_rpi -it giulbia/gcp-rpi:latest bash script/run_crying_detection.sh
