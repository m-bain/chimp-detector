##
## Entry point for Seebibyte Visual Tracker
##
## Author:  Abhishek Dutta <adutta@robots.ox.ac.uk>
## Date:    23 Nov. 2018
##

from svt.svt_server import run_svt_server
from svt.svt_task_manager import svt_task_manager
from svt.detections import detections
from svt.siamrpn_tracker import siamrpn_tracker

import argparse
from pathlib import Path  # to get location of user's home folder
import shutil             # to copy files
import os
import webbrowser         # to open svt ui in a web browser

def svt_start_info(verbose):
  global name, shortname, version, page, model_url, SVT_SERVER_HOSTNAME, SVT_SERVER_PORT

  name      = 'Seebibyte Visual Tracker'
  shortname = 'SVT'
  version   = '2.0.1'
  page      = 'http://www.robots.ox.ac.uk/~vgg/projects/seebibyte/software/svt/'
  model_url = 'http://www.robots.ox.ac.uk/~vgg/projects/seebibyte/software/svt/download/model/latest'
  SVT_SERVER_HOSTNAME = '0.0.0.0'
  SVT_SERVER_PORT = 10001

  if verbose:
    print('')
    print(name + '(' + shortname + ')')
    print('ver. ' + version)
    print('see [' + page + '] for details.')
    print('')

def main():
  global task_manager, SVT_SERVER_HOSTNAME, SVT_SERVER_PORT

  ## initialize and define the command line parser
  parser = argparse.ArgumentParser(prog='svt',
                                   description='Seebibyte Visual Tracker (SVT) [http://www.robots.ox.ac.uk/~vgg/projects/seebibyte/software/svt/]')
  parser.add_argument('--cmd',
                      required=False,
                      type=str,
                      choices=['match_detections'],
                      help='See SVT documentation for a detailed description of each command.')
  parser.add_argument('--infile',
                      required=False,
                      help='input file containing metadata (e.g. filenames, bounding boxes, etc) that will be processed by SVT')
  parser.add_argument('--indir',
                      required=False,
                      help='path location where all the assets (frames, video, etc) that are described in the infile')
  parser.add_argument('--outdir',
                      required=False,
                      help='location where svt application data (e.g.tracker model file, tracks, etc) are written. Default: $HOME/svt_app_data/')
  parser.add_argument('--othreshold',
                      required=False,
                      type=float,
                      help='overlap threshold for positive detection [0,1]')
  parser.add_argument('--outfile',
                      required=False,
                      help='the object tracking data will be exported to this file in JSON format')
  parser.add_argument('--outfmt',
                      required=False,
                      type=str,
                      default='plain_csv',
                      choices=['plain_csv','via_annotation','via_project'],
                      help='output format, see SVT documentation for a detailed description of each format.')
  parser.add_argument('--update',
                      required=False,
                      help='forces download of latest tracker model file before tracking can begin',
                      action='store_true')
  parser.add_argument('--verbose',
                      required=False,
                      help='show progress',
                      action='store_true')
  parser.add_argument('--gpu',
                      required=False,
                      type=int,
                      default=0,
                      help='use this GPU (uses GPU 0 by default, set -1 to disable GPU usage)')
  args = parser.parse_args()

  ## show start info and define constants
  svt_start_info(args.verbose)

  ## locate where the svt source is located
  svt_src_dir = os.path.dirname(os.path.realpath(__file__))

  ## create a folder to store all data generated and used by svt
  svt_app_data_dir = os.path.join(Path.home(), 'vgg', 'svt') # write all track data to user's home folder
  if args.outdir is not None:
    svt_app_data_dir = args.outdir
  if not os.path.isdir(svt_app_data_dir):
    os.makedirs(svt_app_data_dir)

  ## download tracker model file (if missing)
  pretrained_model_folder = os.path.join( svt_app_data_dir, 'pretrained_model_data' )
  model_path = os.path.join(pretrained_model_folder, 'latest.pth')

  if args.cmd is None:
    ## if command line arguments are missing,
    ## start svt_server and listen for requests over HTTP
    ## initialize the svt task manager

    ## copy svt.html (a GUI to use SVT tracker)
    svt_src_html_filename = os.path.join(svt_src_dir, 'ui', 'svt.html')
    svt_dst_html_filename = os.path.join(svt_app_data_dir, 'svt.html')
    if os.path.isfile(svt_dst_html_filename):
      if os.path.getsize(svt_src_html_filename) != os.path.getsize(svt_dst_html_filename):
        shutil.copy2(svt_src_html_filename, svt_dst_html_filename)
    else:
      shutil.copy2(svt_src_html_filename, svt_dst_html_filename)

    svt_track_dir = os.path.join(svt_app_data_dir, 'track_data')
    if not os.path.isdir(svt_track_dir):
      os.makedirs(svt_track_dir)

    task_manager = svt_task_manager(model_path=model_path, track_data_dir=svt_track_dir, gpu_id=args.gpu)
    task_manager.start()
    svt_html_ui = 'file://' + os.path.realpath(svt_dst_html_filename)
    print( 'Opening [ %s ] in default web browser ...' % (svt_html_ui))
    webbrowser.open( svt_html_ui, new=0) # open svt ui in a web browser
    print('Ready to track, listening for request at %s:%d' %
          (SVT_SERVER_HOSTNAME, SVT_SERVER_PORT))
    run_svt_server(task_manager, SVT_SERVER_HOSTNAME, SVT_SERVER_PORT)
  else:
    if args.cmd == 'match_detections':
      tracker_config = {'gpu_id':args.gpu,
                        'verbose':args.verbose,
                        'preload_model':True,
                        'model_url':'http://www.robots.ox.ac.uk/~vgg/projects/seebibyte/software/svt/download/model/latest',
                        'force_model_download':args.update,
                        'download_model_if_missing':True }
      tracker = siamrpn_tracker(model_path='/home/tlm/latest.pth', config=tracker_config)

      detections_match_config = {'match_overlap_threshold':0.5,
                                 'UNKNOWN_TRACK_ID_MARKER':-1,
                                 'frame_img_dir':args.indir,
                                 'verbose':args.verbose,
                                 'via_project_name':'my_via_project' }
      my_detections = detections()
      my_detections.read_from_file(data_filename=args.infile, data_format='csv')
      #my_detections.read(detection_data) # if you already have detections in a python dict, use this method
      my_detections.match(tracker=tracker, config=detections_match_config)
      my_detections.export(args.outfile, args.outfmt, config=detections_match_config)

if __name__ == '__main__':
  main()
