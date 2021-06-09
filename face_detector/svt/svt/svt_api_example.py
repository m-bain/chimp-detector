from svt.detections import detections
from svt.siamrpn_tracker import siamrpn_tracker

#### initialize a tracker
gpu_id = 0    # set to -1 to use CPU only
svt_model_path = '/data/svt/pretrained_model_data/latest.pth' # this file will be downloaded and saved (if missing)
tracker_config = {'gpu_id':gpu_id,
                  'verbose':True,
                  'preload_model':True,
                  'model_url':'http://www.robots.ox.ac.uk/~vgg/projects/seebibyte/software/svt/download/model/latest',
                  'force_model_download':False,
                  'download_model_if_missing':True }
tracker = siamrpn_tracker(model_path=svt_model_path, config=tracker_config)

#### initialize detections
frame_img_dir = '/data/svt/all_frames'
detections_match_config = {'match_overlap_threshold':0.6,
                           'UNKNOWN_TRACK_ID_MARKER':-1,
                           'frame_img_dir':frame_img_dir,
                           'verbose':True,
                           'via_project_name':'my_via_project' }
## see below for a description of this data format
detection_data = {"2": {"3": {"1": [-1, 520.34, 76.677, 41.669, 51.005]}, "4": {"1": [-1, 524.07, 81.6, 40.936, 55.243]}, "5": {"1": [-1, 498.4, 79.521, 41.646, 58.98]}, "6": {"1": [-1, 499.59, 78.697, 41.514, 56.18]}, "7": {"1": [-1, 502.9, 83.906, 39.9, 58.278]}, "8": {"1": [-1, 504.99, 77.887, 39.652, 56.498]}, "9": {"1": [-1, 502.17, 75.613, 39.794, 56.526]}, "10": {"1": [-1, 505.44, 80.003, 38.938, 56.438]}, "11": {"1": [-1, 500.09, 77.46, 40.499, 58.613]}, "12": {"1": [-1, 501.16, 86.252, 38.778, 57.66]}, "13": {"1": [-1, 500.89, 85.825, 39.086, 57.768]}}, "3": {"14": {"1": [-1, 495.46, 47.232, 35.617, 50.696], "2": [-1, 281.41, 59.037, 31.216, 49.609], "3": [-1, 131.55, 21.857, 34.817, 52.069], "4": [-1, 307.36, 148.58, 42.175, 55.213], "5": [-1, 231.01, 103.83, 26.411, 46.58]}, "15": {"1": [-1, 461.5, 11.899, 32.614, 52.105], "2": [-1, 257.38, 13.018, 37.787, 46.698], "3": [-1, 349.87, 151.17, 40.916, 53.872], "4": [-1, 157.79, -7.4872, 39.838, 36.486]}, "16": {"1": [-1, 494.87, -10.776, 32.163, 42.302], "2": [-1, 406.67, 136.33, 41.633, 46.173]}, "17": {"1": [-1, 376.36, 105.12, 42.241, 52.559], "2": [-1, 442.79, -11.862, 38.468, 36.625]}}}

frame_id_to_filename_map = {"3": "00003.jpg", "4": "00004.jpg", "5": "00005.jpg", "6": "00006.jpg", "7": "00007.jpg", "8": "00008.jpg", "9": "00009.jpg", "10": "00010.jpg", "11": "00011.jpg", "12": "00012.jpg", "13": "00013.jpg", "14": "00014.jpg", "15": "00015.jpg", "16": "00016.jpg", "17": "00017.jpg", "18": "00018.jpg"}

my_detections = detections()
my_detections.read(detection_data, frame_id_to_filename_map) # if you already have detections in a python dict, use this method
# if detections are saved in a csv file, use the following method
#my_detections.read_from_file(data_filename=args.infile, data_format='csv')

#### match detections using the tracker and detection data
my_detections.match(tracker=tracker, config=detections_match_config)

#### save matched detections to a VIA project (for visualization)
via_project_filename = '/data/svt/matched_detections_via_project.json'
output_format = 'via_project'    # allowed values {'plain_csv', 'via_annotation', 'via_project'}
my_detections.export(outfile=via_project_filename, outfmt=output_format, config=detections_match_config)
