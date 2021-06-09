# Seebibyte Visual Tracker

[SVT](http://www.robots.ox.ac.uk/~vgg/projects/seebibyte/software/svt/) is a visual tracking software that can track any object in a video. The initial location of the object is provided by the user by manually drawing a box around this object. This object is then tracked automatically in subsequent frames of the video.

This software is developed and maintained by <a href="http://www.robots.ox.ac.uk/~adutta/">Abhishek Dutta</a> and the deep neural network model used by this software was developed and trained by <a href="http://www.robots.ox.ac.uk/~lz/">Li Zhang</a>.

The development and maintenance of Seebibyte Visual Tracker (SVT) is supported by EPSRC programme grant Seebibyte: Visual Search for the Era of Big Data (EP/M013774/1).

## Installation

 * Install [anaconda](https://conda.io/docs/user-guide/install/index.html)
 * Install `ffmpeg` (needed only if you want to use GUI, most Linux systems already have it)
```
sudo apt-get install ffmpeg       # required to read and write video frames
```

 * Install `git`
```
sudo apt-get install git          # needed to install svt from gitlab source repo.
```

 * Create a new conda environment and install the dependencies as follows:
```
conda update -n base conda        # to update conda
conda create -n svt python=3.7    # source: conda-forge
conda activate svt                # to deactivate this environment, use `conda deactivate`
pip install opencv-python
conda install pytorch torchvision -c pytorch    # around 480MB of new software
python3 -m pip install git+https://gitlab.com/vgg/svt/ # install svt
svt --help                        # success if you see help messages
```

## User Guide: Graphical User Interface
```
$ conda activate svt
$ svt.py
```

Now open `/home/tlm/dev/svt/src/ui/svt.html` in a web browser (Chrome or Firefox) and

  * Input the location of a video file and click Submit
  * Click on the video to pause the video and draw a region that you wish to track
  * Now click the "Start Tracking" (⊳, a small play button in the control panel on top left corner) button to track the object

You can also upload or download the track annotations using buttons in the control panel.

### User Guide: Command Line Interface (for Advanced Users)
```
$ conda activate svt
$ svt --help
usage: svt [-h] [--cmd {match_detections}] [--infile INFILE] [--indir INDIR]
           [--outdir OUTDIR] [--othreshold OTHRESHOLD] [--outfile OUTFILE]
           [--outfmt {plain_csv,via_annotation,via_project}] [--update]
           [--verbose] [--gpu GPU]

Seebibyte Visual Tracker (SVT)
[http://www.robots.ox.ac.uk/~vgg/projects/seebibyte/software/svt/]

optional arguments:
  -h, --help            show this help message and exit
  --cmd {match_detections}
                        See SVT documentation for a detailed description of
                        each command.
  --infile INFILE       input file containing metadata (e.g. filenames,
                        bounding boxes, etc) that will be processed by SVT
  --indir INDIR         path location where all the assets (frames, video,
                        etc) that are described in the infile
  --outdir OUTDIR       location where svt application data (e.g.tracker model
                        file, tracks, etc) are written. Default:
                        $HOME/svt_app_data/
  --othreshold OTHRESHOLD
                        overlap threshold for positive detection [0,1]
  --outfile OUTFILE     the object tracking data will be exported to this file
                        in JSON format
  --outfmt {plain_csv,via_annotation,via_project}
                        output format, see SVT documentation for a detailed
                        description of each format.
  --update              forces download of latest tracker model file before
                        tracking can begin
  --verbose             show progress
  --gpu GPU             use this GPU (uses GPU 0 by default, set -1 to disable
                        GPU usage)
```

Here is a sample command to match automatic face detections:
```
svt --cmd match_detections --infile /data/initial_detections.csv --indir /data/all_frames/ --outdir /home/me/vgg/svt/ --othreshold 0.6 --outfile /data/matched_detections.csv --outfmt plain_csv --verbose --gpu 0
```
For this command to run, we need the following assets:
  * a set of initial detections saved in the `/data/initial_detections.csv` file
  * all the frame images stored in `/data/all_frames/` folder
  * a GPU (if you don't have a GPU, set `--gpu -1`)

If you want to visualize the matched track detections, use `--outfmt via_project` 
and `--outfile /data/matched_detections_via_project.json`. This will generate a 
[VIA](www.robots.ox.ac.uk/~vgg/software/via/) project containing all frames, their 
automatic detections and a corresponding `track_id`. To visualize the matched tracks, 
open [VIA](www.robots.ox.ac.uk/~vgg/software/via/), click <code>Project &rarr; Load</code> 
and point to the `/data/matched_detections_via_project.json` file. Now, click 
<code>View &rarr; Toggle Image Grid View</code> and create a new group using 
<code>Group by</code> with `track_id` attribute.

A sample of `/data/initial_detections.csv` is shown below. Initially, the `track_id` 
is unknown. When `svt --cmd match_detections` command is complete, the `track_id` 
will contain a globally unique identifier for each matching detections in consecutive frames.

```
"shot_id","frame_id","frame_filename","track_id","box_id","x","y","width","height"
2,3,"00003.jpg",-1,1,520.34,76.677,41.669,51.005
2,4,"00004.jpg",-1,1,524.07,81.6,40.936,55.243
2,5,"00005.jpg",-1,1,498.4,79.521,41.646,58.98
2,6,"00006.jpg",-1,1,499.59,78.697,41.514,56.18
2,7,"00007.jpg",-1,1,502.9,83.906,39.9,58.278
2,8,"00008.jpg",-1,1,504.99,77.887,39.652,56.498
2,9,"00009.jpg",-1,1,502.17,75.613,39.794,56.526
2,10,"00010.jpg",-1,1,505.44,80.003,38.938,56.438
2,11,"00011.jpg",-1,1,500.09,77.46,40.499,58.613
2,12,"00012.jpg",-1,1,501.16,86.252,38.778,57.66
2,13,"00013.jpg",-1,1,500.89,85.825,39.086,57.768
3,14,"00014.jpg",-1,1,495.46,47.232,35.617,50.696
3,14,"00014.jpg",-1,2,281.41,59.037,31.216,49.609
3,14,"00014.jpg",-1,3,131.55,21.857,34.817,52.069
3,14,"00014.jpg",-1,4,307.36,148.58,42.175,55.213
3,14,"00014.jpg",-1,5,231.01,103.83,26.411,46.58
3,15,"00015.jpg",-1,1,461.5,11.899,32.614,52.105
3,15,"00015.jpg",-1,2,257.38,13.018,37.787,46.698
3,15,"00015.jpg",-1,3,349.87,151.17,40.916,53.872
3,15,"00015.jpg",-1,4,157.79,-7.4872,39.838,36.486
3,16,"00016.jpg",-1,1,494.87,-10.776,32.163,42.302
3,16,"00016.jpg",-1,2,406.67,136.33,41.633,46.173
3,17,"00017.jpg",-1,1,376.36,105.12,42.241,52.559
3,17,"00017.jpg",-1,2,442.79,-11.862,38.468,36.625
3,18,"00018.jpg",-1,1,31.854,95.624,30.07,48.337
3,18,"00018.jpg",-1,2,330.44,73.499,43.183,60.744
...
```

### User Guide: SVT as a Python Library (for Advanced Users)
See [svt/svt_api_example.py] for a full code example on how to access SVT as 
a library in python. Here is the code with some description.

```
from svt.detections import detections
from svt.siamrpn_tracker import siamrpn_tracker

#### initialize a tracker
gpu_id = 0    # set to -1 to use CPU only
svt_model_path = '/home/tlm/data/svt/latest.pth' # this file will be downloaded and saved (if missing)
tracker_config = {'gpu_id':gpu_id,
                  'verbose':args.verbose,
                  'preload_model':True,
                  'model_url':'http://www.robots.ox.ac.uk/~vgg/projects/seebibyte/software/svt/download/model/latest',
                  'force_model_download':args.update,
                  'download_model_if_missing':True }
tracker = siamrpn_tracker(model_path=svt_model_path, config=tracker_config)

#### initialize detections
frame_img_dir = '/data/all_frames'
detections_match_config = {'match_overlap_threshold':0.6,
                           'UNKNOWN_TRACK_ID_MARKER':-1,
                           'frame_img_dir':frame_img_dir,
                           'verbose':True,
                           'via_project_name':'my_via_project' }
## see below for a description of this data format
detection_data = { '2': {'3': { '1': [-1, 520.34,76,677,41,669,51,005], ... }, ... }, ... }
frame_id_to_filename_map = { "3": "00003.jpg", "4": "00004.jpg", "5": "00005.jpg", ... }

my_detections = detections()
my_detections.read(detection_data) # if you already have detections in a python dict, use this method
# if detections are saved in a csv file, use the following method
#my_detections.read_from_file(data_filename=args.infile, data_format='csv')

#### match detections using the tracker and detection data
my_detections.match(tracker=tracker, config=detections_match_config)

#### save matched detections to a VIA project (for visualization)
via_project_filename = '/data/matched_detections_via_project.json'
output_format = 'via_project'    # allowed values {'plain_csv', 'via_annotation', 'via_project'}
my_detections.export(outfile=via_project_filename, outfmt=output_format, config=detections_match_config)
```

The `detection_data` variable is a Python dictionary
```
{
  "2": {          # shot_id  = '2'
    "3": {        # frame_id = '3'
      "1": [      # bounding box id = '1' (first detection in a frame)
        -1,       # track_id = -1 (to indicate that track is unknown)
        520.34,   # x coordinate of the bounding box
        76.677,   # y coordinate of the bounding box
        41.669,   # width of the bounding box
        51.005    # height of the bounding box
      ],
      "2": [      # bounding box id = '2' (second detection in the frame)
        -1,       # track_id = -1 (to indicate that track is unknown)
        520.34,   # x coordinate of the bounding box
        76.677,   # y coordinate of the bounding box
        41.669,   # width of the bounding box
        51.005    # height of the bounding box
      ],
      ...
    },
    "4": {
      "1": [
        -1,
        524.07,
        81.6,
        40.936,
        55.243
      ],
      ...
    }
    ...
  },
  ...
}
```

The `frame_id_to_filename_map` variable is also a Python dictionary which builds 
a correspondence between `frame_id` and the related frame image filename.
```
{ 
  "3": "00003.jpg", ## frame_id=3 corresponds to the frame image '00003.jpg'
  "4": "00004.jpg",
  ...
}
```

To visualize the matched tracks, open [VIA](www.robots.ox.ac.uk/~vgg/software/via/), 
click <code>Project &rarr; Load</code> and point to the `/data/matched_detections_via_project.json` 
file. Now, click <code>View &rarr; Toggle Image Grid View</code> and create a 
new group using <code>Group by</code> with `track_id` attribute.
## Contact
If you encounter issues while installing or using this software, please report it at https://gitlab.com/vgg/svt/issues (requires an account at gitlab.com)

