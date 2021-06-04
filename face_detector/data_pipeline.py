from __future__ import print_function

__author__      = 'Ernesto Coto'
__copyright__   = 'April 2019'

import sys
import os
import argparse
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
import io
import cv2
import numpy as np
import pickle
from collections import OrderedDict
from create_detection_vid import create_det_vid

# import face detection stuff
sys.path.append(os.path.join(os.path.dirname(__file__), 'ssd.pytorch'))
from data import BaseTransform
from ssd import build_ssd

# import tracking stuff
sys.path.append(os.path.join(os.path.dirname(__file__), 'svt'))
from svt.detections import detections
from svt.siamrpn_tracker import siamrpn_tracker

# import face identification stuff
import torchvision.models as models

# import customizes export functions for VIA
import customized_via_exports
import utils

# face detection constants
DEFAULT_FACE_DETECT_MODEL = os.path.join(os.path.dirname(__file__), 'ssd.pytorch', 'weights', 'ssd300_CF_115000.pth' )
DEFAULT_FACE_DETECT_VISUAL_THRESHOLD = 0.6
DEFAULT_DETECTIONS_FILENAME = 'detections.pkl'  # note that since April 2019, this file will also include face identification info
DEFAULT_ID_TO_FILE_FILENAME = 'frame_id_to_filename.pkl'

# face tracking constants
DEFAULT_MATCH_OVERLAP_THRESHOLD = 0.6
UNKNOWN_TRACK_ID_MARKER = -1

# face identification constants
DEFAULT_FACE_ID_MODEL = os.path.join(os.path.dirname(__file__), 'ssd.pytorch', 'weights', 'Resnet18_w_ctai_many_model_best_21March19.pth.tar' )
FACE_ID_CLASSES = 94
FACE_ID_MAP = {
    0: 'NOTFACE',
    1: 'YO',
    2: 'YOLO',
    3: 'VELU',
    4: 'JOYA',
    5: 'JEJE',
    6: 'JIRE',
    7: 'PELEY',
    8: 'PAMA',
    9: 'FANA',
    10: 'FOAF',
    11: 'FLANLE',
    12: 'FANLE',
    13: 'TUA',
    14: 'KAI',
    15: 'NINA',
    16: 'NTO',
    17: 'FOTAIU',
    18: 'JURU',
    19: 'VUAVUA',
    20: 'PONI',
    21: 'PILI',
    22: 'POKURU',
    23: 'FANWA',
}
face_id_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Enable/Disable reducing the number of images per track
DEFAULT_REDUCE_IMAGES_PER_TRACK = False

# Number of retries when detection/identification fails on an image due to 'out of memory'
OOM_MAX_NUM_RETRIES = 2

if __name__ == '__main__':

    # Check arguments before doing anything
    parser = argparse.ArgumentParser(description='data ingestion pipeline')
    parser.add_argument('video_log_file_name', metavar='video_log_file_name', type=str, help='Full path to the pipeline log file')
    parser.add_argument('video_result_file_name', metavar='video_result_file_name', type=str, help='Full path to the pipeline result file')
    parser.add_argument('video_processing_folder_name', metavar='video_processing_folder_name', type=str, help='Full path to the pipeline temporary processing folder')
    parser.add_argument('-m', dest='detection_model', default=DEFAULT_FACE_DETECT_MODEL, type=str, help='DETECTION model to load')
    parser.add_argument('-f', dest='face_id_model', default=DEFAULT_FACE_ID_MODEL, type=str, help='FACE IDENTIFICATION model to load')
    parser.add_argument('-d', dest='previous_detections_file', type=str, help='Full path to file containing previous detections data (it must have been produce with this script)')
    parser.add_argument('-t', dest='visual_threshold', default=DEFAULT_FACE_DETECT_VISUAL_THRESHOLD, type=float, help='Confidence threshold for detection')
    parser.add_argument('-o', dest='overlap_threshold', default=DEFAULT_MATCH_OVERLAP_THRESHOLD, type=float, help='Overlap threshold for building tracks')
    parser.add_argument('-c', dest='use_cuda', default=False, action='store_true', help='Use cuda to evaluate model')
    parser.add_argument('-i', dest='run_face_id', default=False, action='store_true', help='Run face identification process')
    parser.add_argument('-ld', dest='load_detections', default=False, action='store_true', help='Skips the DETECTION process and load the detections from previous data (use the -d option)')
    parser.add_argument('-li', dest='load_face_ids', default=False, action='store_true', help='Skips the FACE IDENTIFICATION process and loads the identification scores from previous data (use the -d option)')
    parser.add_argument('-r', dest='reduce_tracks', default=DEFAULT_REDUCE_IMAGES_PER_TRACK, action='store_true', help='Reduces the images saved per track to the output VIA project')
    parser.add_argument('-v', dest='original_video', required=True, type=str)

    args = parser.parse_args()

    # Create/clear the log file
    log_out = open(args.video_log_file_name, 'a', buffering=1)
    output = ""
    err = ""

    try:
        ########################################################
        #### INPUT PHASE
        ########################################################

        # Check parameters
        if args.load_face_ids and args.previous_detections_file is None:
            raise Exception("Can't load the face identifications because no previous detection data file has been specified")

        if args.load_detections and args.previous_detections_file is None:
            raise Exception("Can't load the face detections because no previous detection data file has been specified")

        if args.run_face_id and args.load_face_ids:
            raise Exception("Can't load the face identifications and run the face identification process at the same time")

        # Start PyTorch pipeline
        log_out.write('[%s]: Starting PyTorch pipeline ...\n' %  time.strftime("%d-%m-%Y %H:%M:%S") )
        log_out.write('[%s]: %s\n' %  (time.strftime("%d-%m-%Y %H:%M:%S"), str(sys.argv) ))

        # Check if previous detection data was provided
        loaded_detections = None
        frame_id_to_filename = None
        previous_detections_file = None
        current_detections = None
        if args.load_detections or args.load_face_ids:
            previous_detections_file = args.previous_detections_file
            if os.sep not in args.previous_detections_file:
                previous_detections_file = os.path.join( os.path.dirname(args.video_processing_folder_name), args.previous_detections_file)
            if os.path.exists(previous_detections_file):
                with open(previous_detections_file, 'rb') as fin:
                    loaded_detections = pickle.load(fin)
                frame_id_to_filename = loaded_detections['frame_id_to_filename']
                loaded_detections = loaded_detections['detections_and_scores']
                log_out.write('[%s]: Loaded previous detections data from %s\n' %  (time.strftime("%d-%m-%Y %H:%M:%S"), previous_detections_file) )
            else:
                raise Exception("Could not find file %s with previous detections data" % previous_detections_file)

        # Acquire list of images
        video_frames_list = os.listdir(args.video_processing_folder_name)
        video_frames_list.sort()
        video_frames_list_size = len(video_frames_list)
        if len(video_frames_list) == 0:
            raise Exception('ERROR: There are no frames in the video frames path. Aborting !\n')
        log_out.write('[%s]: The pipeline will process %d images\n' %  (time.strftime("%d-%m-%Y %H:%M:%S"), video_frames_list_size ) )

        # Set the default tensor type
        # If cuda was requested, check that it is really available
        cuda_enable = False
        if args.use_cuda and torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            cuda_enable = True
            log_out.write('[%s]: CUDA is enable\n' %  time.strftime("%d-%m-%Y %H:%M:%S") )
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        ########################################################
        #### DETECTION PHASE
        ########################################################

        # If we have to run the detection process...
        if not args.load_detections:

            num_classes = 2 # +1 background
            net = build_ssd('test', 300, num_classes) # initialize SSD
            if cuda_enable:
                net.load_state_dict(torch.load(args.detection_model))
                net.eval()
                net = net.cuda()
                cudnn.benchmark = True
            else:
                net.load_state_dict(torch.load(args.detection_model, map_location=torch.device('cpu')))
                net.eval()

            log_out.write('[%s]: Finished loading model %s\n' %  (time.strftime("%d-%m-%Y %H:%M:%S"), args.detection_model) )
            log_out.write('[%s]: Starting detection phase\n' %  time.strftime("%d-%m-%Y %H:%M:%S") )

            # init this important variables
            frame_id_to_filename = {}
            current_detections = { '0': {} } # so far, we use only one shot '0'

            # Go through frame list
            for im_index in range(video_frames_list_size):

                if im_index % 1000 == 0 or im_index == video_frames_list_size-1:
                    log_out.write('[%s]: Starting to process images %d to %d \n' %  (time.strftime("%d-%m-%Y %H:%M:%S"), im_index, min(im_index+1000-1, video_frames_list_size-1)))

                # Acquire image
                frame_fname = video_frames_list[im_index]
                frame_full_path = os.path.join(args.video_processing_folder_name, frame_fname)
                img = cv2.imread(frame_full_path)

                # Apply transforms to the image
                transform = BaseTransform(net.size, (104, 117, 123))
                transformed = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
                transformed = Variable(transformed.unsqueeze(0))

                # Send image through network
                if cuda_enable:
                    transformed = transformed.cuda()

                oom_retry = OOM_MAX_NUM_RETRIES
                while oom_retry>0:
                    try:
                        if oom_retry != OOM_MAX_NUM_RETRIES:
                            log_out.write('[%s]: OUT OF MEMORY - Retrying detection at image %s\n' %  (time.strftime("%d-%m-%Y %H:%M:%S"), frame_full_path ) )

                        net_out = net(transformed)      # run detector
                        net_detections = net_out.data

                        if oom_retry != OOM_MAX_NUM_RETRIES:
                            log_out.write('[%s]: OUT OF MEMORY - Success retrying detection at image %s\n' %  (time.strftime("%d-%m-%Y %H:%M:%S"), frame_full_path ) )

                        break # leave the while
                    except Exception as e:
                        if 'out of memory' in str(e):
                            log_out.write('[%s]: OUT OF MEMORY while running the detection at image %s. EXCEPTION %s\n' %  (time.strftime("%d-%m-%Y %H:%M:%S"), frame_full_path, str(e) ) )
                            oom_retry = oom_retry - 1
                        else:
                            log_out.write('[%s]: Error while running the detection at image %s. EXCEPTION %s\n' %  (time.strftime("%d-%m-%Y %H:%M:%S"), frame_full_path, str(e) ) )
                            oom_retry = 0

                if oom_retry <=0:
                    log_out.write('[%s]: Giving up on detection at image %s\n' %  (time.strftime("%d-%m-%Y %H:%M:%S"), frame_full_path ) )
                    continue

                # Scale each detection back up to the image
                scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
                pred_num = 0
                for i in range(net_detections.size(1)):
                    j = 0
                    while net_detections[0, i, j, 0] >= args.visual_threshold:
                        score = net_detections[0, i, j, 0]
                        pt = (net_detections[0, i, j, 1:]*scale).cpu().numpy()
                        coords = ( max(float(pt[0]), 0.0),
                            max(float(pt[1]), 0.0),
                            min(float(pt[2]),img.shape[1]),
                            min(float(pt[3]),img.shape[0]))
                        if coords[2]-coords[0] >= 1 and coords[3]-coords[1] >= 1:
                            if pred_num == 0:
                                # We found at least one prediction, so add the image to the filename dict and the current detections dict
                                frame_id_to_filename[ str(im_index) ] = frame_fname
                                current_detections['0'][ str(im_index) ] = {}
                            # Save detections to list ...
                            if not args.load_face_ids or not args.run_face_id:
                                # In this case, the face_id_scores might be generated later depending on args.run_face_id
                                a_detection = [ UNKNOWN_TRACK_ID_MARKER, coords[0], coords[1], # [track_id, x, y,
                                                coords[2]-coords[0], coords[3]-coords[1], #  w, h,
                                                None ]  # dummy-face_id_scores ]
                            else:
                                 # In this case, load the score ids from the previous data file.
                                 # Note that for this work the previous data file must have been the output of this same
                                 # script over the same data, or the internal structure of 'loaded_detections' and
                                 # 'current_detections' will not match and the command below must likely will cause an error.
                                a_detection = [ UNKNOWN_TRACK_ID_MARKER, coords[0], coords[1], # [track_id, x, y,
                                                coords[2]-coords[0], coords[3]-coords[1], #  w, h,
                                                loaded_detections['0'][ str(im_index) ][ str(pred_num) ][5] ] # face_id_scores ]

                            current_detections['0'][ str(im_index) ][ str(pred_num) ] = a_detection
                            pred_num += 1
                        j += 1

            log_out.write('[%s]: Finished detections\n' %  time.strftime("%d-%m-%Y %H:%M:%S") )

        else:
            # At this point, we have skipped the detection process, so we can just copy the previous data file
            current_detections = loaded_detections
            # However, we should reset the face_id_scores, if present
            if args.run_face_id:
                for image_info in current_detections['0']:
                    for prediction_info in current_detections['0'][image_info]:
                        current_detections['0'][image_info][prediction_info][5] = None

        ########################################################
        #### FACE IDENTIFICATION PHASE
        ########################################################

        # If we have to run the face identification process...
        if args.run_face_id and not args.load_face_ids:

            # Load models
            face_id_model = models.__dict__['resnet18']()
            face_id_model.fc =  nn.Linear(512, FACE_ID_CLASSES)
            face_id_checkpoint = torch.load(args.face_id_model)
            if cuda_enable:
                face_id_model = nn.DataParallel(face_id_model).cuda()
                face_id_model.load_state_dict(face_id_checkpoint['state_dict'])
            else:
                state_dict = face_id_checkpoint['state_dict']
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k.replace('module.', '')
                    new_state_dict[name] = v
                face_id_model.load_state_dict(new_state_dict)

            face_id_model.eval()
            log_out.write('[%s]: Finished loading model %s\n' %  (time.strftime("%d-%m-%Y %H:%M:%S"), args.face_id_model) )
            log_out.write('[%s]: Starting face identification phase\n' %  time.strftime("%d-%m-%Y %H:%M:%S") )

            if cuda_enable:
                cudnn.benchmark = True

            im_index_c = 0
            frame_id_num = len(frame_id_to_filename.keys())
            for im_index in frame_id_to_filename.keys():

                if im_index_c % 1000 == 0 or im_index_c == frame_id_num-1:
                    log_out.write('[%s]: Starting to process images %d to %d \n' %  (time.strftime("%d-%m-%Y %H:%M:%S"), im_index_c, min(im_index_c+1000-1, frame_id_num-1)))

                # Acquire image (yes, again)
                frame_fname = frame_id_to_filename[ im_index ] # remember here im_index should be a string
                frame_full_path = os.path.join(args.video_processing_folder_name, frame_fname)
                img = cv2.imread(frame_full_path)

                im_index_c = im_index_c + 1

                for prediction in current_detections['0'][ im_index ]:
                    coords = current_detections['0'][ im_index ][prediction][1:5]
                    coords[2] = coords[2] + coords[0] # change width to coordinates
                    coords[3] = coords[3] + coords[1] # change height to coordinates
                    oom_retry = OOM_MAX_NUM_RETRIES
                    while oom_retry>0:
                        try:
                            if oom_retry != OOM_MAX_NUM_RETRIES:
                                log_out.write('[%s]: OUT OF MEMORY - Retrying identification at image %s\n' %  (time.strftime("%d-%m-%Y %H:%M:%S"), frame_full_path ) )

                            # crop image to detection
                            crop_img = img[ int(coords[1]):int(coords[3]), int(coords[0]):int(coords[2]), :]
                            # prepare face image for classification
                            crop_img= cv2.resize(crop_img, (224, 224)).astype(np.float32)
                            zero_one_range_img = cv2.normalize(crop_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                            torch_permuted_img = torch.from_numpy(zero_one_range_img).permute(2, 0, 1)
                            crop_img_normalized = face_id_normalize(torch_permuted_img)
                            transformed = Variable(crop_img_normalized.unsqueeze(0))
                            # run face classifier
                            face_id_vector = face_id_model(transformed)
                            m = nn.Softmax() # Max's hack for bringing the results down to 0-1 range
                            face_id_vector = m(face_id_vector)
                            # keep only relevant output + move to cpu + detach
                            face_id_vector_keep = face_id_vector[0][:len(FACE_ID_MAP)].cpu().detach().numpy()

                            if oom_retry != OOM_MAX_NUM_RETRIES:
                                log_out.write('[%s]: OUT OF MEMORY - Success retrying identification at image %s\n' %  (time.strftime("%d-%m-%Y %H:%M:%S"), frame_full_path ) )

                            break # leave the while
                        except Exception as e:
                            if 'out of memory' in str(e):
                                log_out.write('[%s]: OUT OF MEMORY while running the face identification at image %s. EXCEPTION %s\n' %  (time.strftime("%d-%m-%Y %H:%M:%S"), frame_full_path, str(e) ) )
                                oom_retry = oom_retry - 1
                            else:
                                log_out.write('[%s]: Error while running the face identification at image %s. EXCEPTION %s\n' %  (time.strftime("%d-%m-%Y %H:%M:%S"), frame_full_path, str(e) ) )
                                oom_retry = 0

                        if oom_retry <=0:
                            log_out.write('[%s]: Giving up on identification at image %s\n' %  (time.strftime("%d-%m-%Y %H:%M:%S"), frame_full_path ) )
                            continue

                    # save face_id_scores
                    current_detections['0'][ im_index ][prediction][5] = face_id_vector_keep

            log_out.write('[%s]: Finished face identification\n' %  time.strftime("%d-%m-%Y %H:%M:%S") )

        #log_out.write('[%s]: -----------------\n' %  time.strftime("%d-%m-%Y %H:%M:%S") )
        #log_out.write('[%s]:current_detections: %s\n' %  (time.strftime("%d-%m-%Y %H:%M:%S"), str(current_detections) ) )
        #log_out.write('[%s]: -----------------\n' %  time.strftime("%d-%m-%Y %H:%M:%S") )
        #log_out.write('[%s]:frame_id_to_filename: %s\n' %  (time.strftime("%d-%m-%Y %H:%M:%S"), str(frame_id_to_filename) ) )
        #log_out.write('[%s]: -----------------\n' %  time.strftime("%d-%m-%Y %H:%M:%S") )

        # save face detections and filename mapping
        detections_pkl = os.path.join( os.path.dirname(args.video_processing_folder_name), DEFAULT_DETECTIONS_FILENAME)
        if not os.path.exists(detections_pkl):
            pickle.dump( { 'detections_and_scores': current_detections, 'frame_id_to_filename': frame_id_to_filename }, open( detections_pkl , "wb" ) )

        ########################################################
        #### TRACKING PHASE
        ########################################################

        log_out.write('[%s]: Starting tracking phase\n' %  time.strftime("%d-%m-%Y %H:%M:%S") )

        # redirect sys.stdout to a buffer to capture the prints() in the code below
        stdout = sys.stdout
        sys.stdout = io.StringIO()

        # initialize a tracker
        gpu_id = -1   # set to -1 to use CPU only
        if cuda_enable:
            gpu_id = 0
        svt_model_path = 'face_detector/svt/model/latest.pth' # this file will be downloaded and saved (if missing)
        tracker_config = {'gpu_id':gpu_id,
                  'verbose': True,
                  'preload_model':True,
                  'model_url':'http://www.robots.ox.ac.uk/~vgg/projects/seebibyte/software/svt/download/model/latest',
                  'force_model_download': False,
                  'download_model_if_missing': True }
        tracker = siamrpn_tracker(model_path=svt_model_path, config=tracker_config)

        # initialize detections
        frame_img_dir = args.video_processing_folder_name
        detections_match_config = {'match_overlap_threshold': args.overlap_threshold,
                                   'UNKNOWN_TRACK_ID_MARKER': UNKNOWN_TRACK_ID_MARKER,
                                   'frame_img_dir': frame_img_dir,
                                   'verbose': True,
                                   'via_project_name':'my_via_project' }

        my_detections = detections()
        my_detections.read(current_detections, frame_id_to_filename)

        # match detections using the tracker and detection data
        my_detections.match(tracker=tracker, config=detections_match_config)

        # get output and restore sys.stdout
        output = sys.stdout.getvalue()
        sys.stdout = stdout
        log_out.write('[%s]: %s\n' %  (time.strftime("%d-%m-%Y %H:%M:%S"), output))
        log_out.write('[%s]: Finished tracking\n' %  time.strftime("%d-%m-%Y %H:%M:%S") )

        ########################################################
        #### OUTPUT PHASE
        ########################################################
        images_subfolder = os.path.basename(args.video_processing_folder_name)
        via_project_filename = os.path.join( os.path.dirname(args.video_processing_folder_name), images_subfolder + '.json' )

        if args.run_face_id or args.load_face_ids:
            # save average vectors to CSV before choosing the face representatives
            avg_vectors_csv_filename = via_project_filename.replace( '.json', '.avg_vectors.csv')
            log_out.write('[%s]: Saving average vectors to %s\n' %  (time.strftime("%d-%m-%Y %H:%M:%S"), avg_vectors_csv_filename))
            customized_via_exports.export_average_vectors_csv(my_detections, avg_vectors_csv_filename)
            # replace face_id_scores in the my_detections internal structure by a single face representative
            # taken from FACE_ID_MAP. This is so that the VIA export saves only the face representative string
            customized_via_exports.choose_track_face_representative(my_detections, FACE_ID_MAP)

        frame_track_map_keep = None
        if args.reduce_tracks:
            # If we are going to keep only a few images per track, then get the list of images
            # to keep so as to specify them during the exporting
            log_out.write('[%s]: Preparing output...\n' %  time.strftime("%d-%m-%Y %H:%M:%S") )
            frame_track_map_keep = customized_via_exports.get_representative_images_per_track(my_detections)

        # save matched detections to a VIA project (for visualization)
        log_out.write('[%s]: Saving VIA project\n' %  time.strftime("%d-%m-%Y %H:%M:%S") )
        if args.run_face_id or args.load_face_ids:
            customized_via_exports_config = {   'frame_img_dir': detections_match_config['frame_img_dir'],
                                                'via_project_name': detections_match_config['via_project_name'],
                                                'extra_region_attributes': { "face_id": {
                                                        "type": "text",
                                                        "description": "face identification corresponding to shape",
                                                        "default_value": "not_defined" }
                                                }
                                            }
            customized_via_exports.export_via_project(detections=my_detections, outfile=via_project_filename,
                            config=customized_via_exports_config, frames_to_keep_per_track=frame_track_map_keep)
        else:
            customized_via_exports.export_via_project(detections=my_detections, outfile=via_project_filename,
                            config=detections_match_config, frames_to_keep_per_track=frame_track_map_keep)

        # save all track information in CSV format (for safekeeping)
        all_tracks_csv_filename = via_project_filename.replace( '.json', '.full.csv')
        if args.run_face_id or args.load_face_ids:
            customized_via_exports.export_plain_csv(detections=my_detections, outfile=all_tracks_csv_filename)
        else:
            my_detections.export(outfile=all_tracks_csv_filename, outfmt='plain_csv', config=detections_match_config)

        # create video with bounding boxes visualised
        create_det_vid(all_tracks_csv_filename, args.video_processing_folder_name, args.original_video)


        # remove the images that were not kept
        if args.reduce_tracks and len(frame_track_map_keep)>0:
            log_out.write('[%s]: Removing unnecessary images\n' %  time.strftime("%d-%m-%Y %H:%M:%S") )
            utils.remove_unused_images(args.video_processing_folder_name, frame_track_map_keep )

        # create zip result file
        if args.video_result_file_name != "None":
            log_out.write('[%s]: Creating ZIP result file\n' %  time.strftime("%d-%m-%Y %H:%M:%S") )
            utils.zipdir( args.video_result_file_name, os.path.dirname(args.video_processing_folder_name))
            log_out.write('[%s]: Output saved to %s\n' %  (time.strftime("%d-%m-%Y %H:%M:%S"), args.video_result_file_name) )
        else:
            log_out.write('[%s]: WARNING the output will not be saved!\n' %  time.strftime("%d-%m-%Y %H:%M:%S") )

        log_out.write('[%s]: PyTorch pipeline finished\n' %  time.strftime("%d-%m-%Y %H:%M:%S") )

    except Exception as e:
        # log the exception and leave
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        log_out.write('[%s]: Failed to invoke the data ingestion python script\n' %  time.strftime("%d-%m-%Y %H:%M:%S") )
        log_out.write('[%s]: EXCEPTION %s\n' % (time.strftime("%d-%m-%Y %H:%M:%S"), str(e)))
        log_out.write('[%s]: exc_type %s\n' % (time.strftime("%d-%m-%Y %H:%M:%S"), str(exc_type)))
        log_out.write('[%s]: fname %s\n' % (time.strftime("%d-%m-%Y %H:%M:%S"), fname))
        log_out.write('[%s]: exc_tb.tb_lineno %s\n' % (time.strftime("%d-%m-%Y %H:%M:%S"), str(exc_tb.tb_lineno)))
        log_out.write(err)
        pass

    log_out.close()
