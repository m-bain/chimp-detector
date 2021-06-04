#!/bin/bash
# Parameters:
# $1 -> video_file: Full path to input video
# $2 -> video_log_file_name: Full path to the pipeline log file
# $3 -> video_result_file_name: Full path to the pipeline result file
# $4 -> video_processing_folder_name: Full path to the pipeline temporary processing folder
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<additional paths>
BASEDIR=$(dirname "$0")
cd "$BASEDIR"
source face_detector/bin/activate
if [ "$#" -eq 5 ]; then
    DATE=`date '+%d-%m-%Y %H:%M:%S'`
    echo "[$DATE]: Executing start_pipeline $1 $2 $3 $4" > "$2"
    # get video name
    VIDEONAME=$(basename "$1")
    # check variable is not empty
    if  [ ! -z "$VIDEONAME" ]; then
        # remove previous temporary folder/file, if present
        rm -rf "$4"
        # make new temporary folder and subfolder for the frames
        mkdir -p "$4" "$4/${VIDEONAME}"
        # download VIA application
        wget http://www.robots.ox.ac.uk/~vgg/software/via/via.html -P "$4"
        # extract video frames, but only 1 frame per second
        DATE=`date '+%d-%m-%Y %H:%M:%S'`
        echo "[$DATE]: Extracting video frames ..." >> "$2"
        echo $VIDEONAME
        "ffmpeg" -i "${1}" -vsync vfr -q:v 1 -start_number 0 -vf scale=iw:ih*\(1/sar\) -loglevel panic "$4/${VIDEONAME}/%06d.jpg" >> "$2" 2>&1
        DATE=`date '+%d-%m-%Y %H:%M:%S'`
        echo "[$DATE]: Finished extracting video frames" >> "$2"
        # call the rest of the pipeline
        echo "[$DATE]: --Starting BodyBissau DETECTION PIPELINE--" >> "$2"
        echo $5
        python face_detector/data_pipeline.py "${2}" None "$4/${VIDEONAME}" -r -c -m face_detector/ssd.pytorch/weights/$5 -t 0.37 -v ${VIDEONAME}
        # create results file
        echo "[$DATE]: --Creating result file--" >> "$2"
        zip -r "$3" "$4" >> "$2" 2>&1
        # clean up
        rm -rf "$4"
        #rm -rf "$1" # keep the video for now{
    else
        echo "start_pipeline: video file name missing"
    fi
else
   echo "start_pipeline: Invalid number of parameters"
fi
