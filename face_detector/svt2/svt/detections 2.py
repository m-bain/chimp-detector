##
## denotes automatically detected regions obtained from standard object
## detectors like face detector, eye detector, etc.
##
## Author:  Abhishek Dutta <adutta@robots.ox.ac.uk>
## Date:    12 Dec. 2018
##

import threading
import os
import csv
import json # for debug
import cv2
import numpy as np
import math
import copy # to clone dict

import svt.models as models
import svt.siamrpn_tracker as siamrpn_tracker

import torch
from functools import partial
import pickle
from torch.autograd import Variable
import torch.nn.functional as F

class detections():
  def __init__(self, detection_data=None, frame_id_to_filename_map=None):
    if detection_data is not None and frame_id_to_filename_map is not None:
      self.detection_data = detection_data
      self.frame_id_to_filename_map = frame_id_to_filename_map

  def read(self, detection_data, frame_id_to_filename_map):
    self.detection_data = copy.deepcopy(detection_data)
    self.frame_id_to_filename_map = copy.deepcopy(frame_id_to_filename_map)

  def read_from_file(self, data_filename, data_format):
    if os.path.isfile(data_filename) and data_format == 'csv':
      self._read_detections_from_csv(data_filename)

  def _read_detections_from_csv(self, filename):
    self.detection_data = {}
    self.frame_id_to_filename_map = {}
    with open(filename) as csvfile:
      filedata = csv.DictReader(csvfile)
      for row in filedata:
        bounding_box = [ int(row['track_id']),
                         float(row['x']),
                         float(row['y']),
                         float(row['width']),
                         float(row['height']) ]
        if row['frame_id'] not in self.frame_id_to_filename_map:
          self.frame_id_to_filename_map[ row['frame_id'] ] = row['frame_filename']

        if row['shot_id'] in self.detection_data:
          if row['frame_id'] in self.detection_data[ row['shot_id'] ]:
            if row['box_id'] not in self.detection_data[ row['shot_id'] ][ row['frame_id'] ]:
              ## append new box to existing (shot_id,frame_id) pair
              self.detection_data[ row['shot_id'] ][ row['frame_id'] ][ row['box_id'] ] = bounding_box
            else:
              ## box_id must be unique for each (shot_id,frame_id) pair
              raise ValueError('box_id=%d is not unique for shot_id=%d and frame_id=%d' %
                               (row['box_id'], row['shot_id'], row['frame_id']))
          else:
            ## add new frame_id and then a new box to this shot_id
            self.detection_data[ row['shot_id'] ][ row['frame_id'] ] = {}
            self.detection_data[ row['shot_id'] ][ row['frame_id'] ][ row['box_id'] ] = bounding_box
        else:
          ## create new (shot_id,frame_id,box_id)
          self.detection_data[ row['shot_id'] ] = {}
          self.detection_data[ row['shot_id'] ][ row['frame_id'] ] = {}
          self.detection_data[ row['shot_id'] ][ row['frame_id'] ][ row['box_id'] ] = bounding_box

  def match(self, tracker, config):
    next_track_id = 0 # intialize globally unique track id
    for shot_id in self.detection_data:
      if config['verbose']:
        print('Processing shot_id=%s' % (shot_id))

      #### retrieve a sorted list of all frame_id for a given shot_id
      frame_id_list = sorted( self.detection_data[shot_id], key=int ) # key=int ensures frame_id is treated as number

      #### run a forward matching pass for each pair of consecutive frames
      for frame_id_index in range(0, len(frame_id_list) - 1):
        template_frame_id = frame_id_list[frame_id_index]
        search_frame_id   = frame_id_list[frame_id_index + 1]
        template_fn       = self.frame_id_to_filename_map[ template_frame_id ]
        search_fn         = self.frame_id_to_filename_map[ search_frame_id ]
        search_bbox_list  = self.detection_data[shot_id][ search_frame_id ]
        #print('  %s -> %s' % (template_frame_id, search_frame_id))

        #### Preload template and search image
        template_abs_fn = os.path.join(config['frame_img_dir'], template_fn)
        search_abs_fn = os.path.join(config['frame_img_dir'], search_fn)
        template_img = self.load_image(template_abs_fn)
        search_img = self.load_image(search_abs_fn)

        for box_id in self.detection_data[shot_id][template_frame_id]:
          #print('    box_id=%s' % (box_id))
          b = self.detection_data[shot_id][template_frame_id][box_id]
          template_bbox = [ b[0], int(b[1]), int(b[2]), int(b[3]), int(b[4]) ] # we don't need float

          #### initialize tracker using frame k
          tracker.init_tracker(template_img, template_bbox)
          #### track the object in frame (k+1)
          pos, size, score = tracker.track(search_img);
          tracked_search_bbox = [ template_bbox[0],
                                  int(pos[0] - size[0]/2),
                                  int(pos[1] - size[1]/2),
                                  int(size[0]),
                                  int(size[1]) ]

          max_overlap_search_box_id, max_overlap = self.find_most_overlapping_bbox(tracked_search_bbox, search_bbox_list)
          #print('      overlap=%f, search bbox_id=%s' % (max_overlap, max_overlap_search_box_id))
          if max_overlap >= config['match_overlap_threshold']:
            # propagate the track_id of template's bbox to the matched search bbox
            if template_bbox[0] == config['UNKNOWN_TRACK_ID_MARKER']:
              self.detection_data[shot_id][template_frame_id][box_id][0] = next_track_id
              next_track_id = next_track_id + 1

            self.detection_data[shot_id][ search_frame_id ][max_overlap_search_box_id][0] = self.detection_data[shot_id][template_frame_id][box_id][0]
            #print('    %s is track %d' % (search_frame_id, template_bbox[0]))

  def export(self, outfile, outfmt, config):
    if outfmt == 'plain_csv':
      self.export_plain_csv(outfile, config)
      return
    if outfmt == 'via_annotation':
      self.export_via_annotation(outfile, config)
      return
    if outfmt == 'via_project':
      self.export_via_project(outfile, config)
      return

    print('Unknown export format %s' % (outfmt))

  def export_plain_csv(self, outfile, config):
    with open(outfile, 'w') as csvfile:
      csvfile.write('shot_id,frame_id,frame_filename,track_id,box_id,x,y,width,height\n')
      for shot_id in self.detection_data:
        for frame_id in self.detection_data[shot_id]:
          row_prefix = '%s,%s,"%s",' % (shot_id,
                                        frame_id,
                                        self.frame_id_to_filename_map[ frame_id ])
          for box_id in self.detection_data[shot_id][frame_id]:
            box = self.detection_data[shot_id][frame_id][box_id]
            row_suffix1 = '%d,%s,%.3f,%.3f,%.3f,%.3f\n' % (box[0], box_id, box[1], box[2], box[3], box[4])
            #row_suffix1 = '%d,%s,%d,%d,%d,%d\n' % (box[0], box_id, box[1], box[2], box[3], box[4])
            csvfile.write( row_prefix + row_suffix1 )

  def export_via_annotation(self, outfile, config):
    with open(outfile, 'w') as viafile:
      viafile.write('filename,file_size,file_attributes,region_count,region_id,region_shape_attributes,region_attributes\n')

      for shot_id in self.detection_data:
        for frame_id in self.detection_data[shot_id]:
          frame_abs_path = os.path.join(config['frame_img_dir'],
                                        self.frame_id_to_filename_map[ frame_id ])
          frame_filesize = os.path.getsize(frame_abs_path)
          row_prefix = '%s,%d,{},%d,' % (frame_abs_path,
                                         frame_filesize,
                                         len(self.detection_data[shot_id][frame_id]))
          for box_id in self.detection_data[shot_id][frame_id]:
            box = self.detection_data[shot_id][frame_id][box_id]
            row_suffix1 = '%s,"{""name"":""rect"",""x"":%d,""y"":%d,""width"":%d,""height"":%d}",' % (box_id, box[1], box[2], box[3], box[4])
            row_suffix2 = '"{""shot_id"":%s,""frame_id"":%s,""box_id"":%s,""track_id"":%d}"\n' % (shot_id, frame_id, box_id, box[0])
            viafile.write( row_prefix + row_suffix1 + row_suffix2 )

  def export_via_project(self, outfile, config):
    via_project = {}
    via_project['_via_settings'] = {
      "ui": {
        "annotation_editor_height": 25,
        "annotation_editor_fontsize": 0.8,
        "leftsidebar_width": 18,
        "image_grid": {
          "img_height": 80,
          "rshape_fill": "none",
          "rshape_fill_opacity": 0.3,
          "rshape_stroke": "yellow",
          "rshape_stroke_width": 2,
          "show_region_shape": True,
          "show_image_policy": "all"
        },
        "image": {
          "region_label": "__via_region_id__",
          "region_label_font": "10px Sans",
          "on_image_annotation_editor_placement": "NEAR_REGION"
        }
      },
      "core": {
        "buffer_size": 18,
        "filepath": {},
        "default_filepath": config['frame_img_dir'] + os.sep
      },
      "project": {
        "name": config['via_project_name']
      }
    }

    via_project['_via_attributes'] = {
      "file":{
        "shot_id":{
          "type": "text",
          "description": "video frames shot continually by a camera are grouped under a single unique shot_id",
          "default_value": "not_defined"
        },
        "frame_id":{
          "type": "text",
          "description": "unique id of each frame",
          "default_value": "not_defined"
        }
      },
      "region":{
        "track_id":{
          "type": "text",
          "description": "regions corresponding to same object have the same globally unique track_id",
          "default_value": "not_defined"
        },
        "box_id":{
          "type": "text",
          "description": "each region in a frame is assigned a unique box_id",
          "default_value": "not_defined"
        }
      }
    }

    via_project['_via_img_metadata'] = {}
    for shot_id in self.detection_data:
      for frame_id in self.detection_data[shot_id]:
        frame_filename = self.frame_id_to_filename_map[ frame_id ]
        frame_abs_path = os.path.join(config['frame_img_dir'],
                                      frame_filename)
        frame_filesize = os.path.getsize(frame_abs_path)
        fileid = '%s%d' % (frame_filename, frame_filesize)
        via_project['_via_img_metadata'][fileid] = {'filename':frame_filename,
                                                    'size':frame_filesize}
        via_project['_via_img_metadata'][fileid]['file_attributes'] = {'shot_id':shot_id,
                                                                       'frame_id':frame_id}
        via_project['_via_img_metadata'][fileid]['regions'] = []
        for box_id in self.detection_data[shot_id][frame_id]:
          box = self.detection_data[shot_id][frame_id][box_id]
          via_project['_via_img_metadata'][fileid]['regions'].append( {
            'shape_attributes':{'name':'rect', 'x':box[1], 'y':box[2], 'width':box[3], 'height':box[4]},
            'region_attributes':{'track_id':box[0], 'box_id':box_id}
          } )

    with open(outfile, 'w') as jsonfile:
      json.dump(via_project, jsonfile, indent=None, separators=(',',':'))

  def find_most_overlapping_bbox(self, new_bbox, existing_bbox_list):
    max_overlap = -1.0
    max_overlap_bbox_id = -1
    for bbox_id in existing_bbox_list:
      bbox_i = existing_bbox_list[bbox_id]
      overlap = self.compute_overlap(new_bbox, bbox_i)
      if overlap > max_overlap:
        max_overlap = overlap
        max_overlap_bbox_id = bbox_id
    return max_overlap_bbox_id, max_overlap

  # assumption: a, b = [track_id, x, y, width, height]
  # see: https://gist.github.com/vierja/38f93bb8c463dce5500c0adf8648d371
  def compute_overlap(self, a, b):
    x11 = a[1]
    y11 = a[2]
    x12 = a[1] + a[3]
    y12 = a[2] + a[4]

    x21 = b[1]
    y21 = b[2]
    x22 = b[1] + b[3]
    y22 = b[2] + b[4]

    intersect_area = 0
    union_area = 0

    ## check if we have nested rectangles
    if self.is_inside(x11, y11, x21, y21, x22, y22) and self.is_inside(x12, y12, x21, y21, x22, y22):
      intersect_area = (x12 - x11) * (y12 - y11)
      union_area = (x22 - x21) * ( y22 - y21)
    else:
      ## check if we have nested rectangles
      if self.is_inside(x21, y21, x11, y11, x12, y12) and self.is_inside(x22, y22, x11, y11, x12, y12):
        intersect_area = (x22 - x21) * ( y22 - y21)
        union_area = (x12 - x11) * (y12 - y11)
      else:
        ## rectangles overlap or they do not overlap
        x0_intersect = max(x11, x21)
        y0_intersect = max(y11, y21)
        x1_intersect = min(x12, x22)
        y1_intersect = min(y12, y22)
        intersect_area = max((x1_intersect - x0_intersect), 0) * max((y1_intersect - y0_intersect), 0)
        union_area = (x12-x11)*(y12-y11) + (x22-x21)*(y22-y21) - intersect_area

    return intersect_area / (union_area + 0.00001)

  def is_inside(self, x, y, x0, y0, x1, y1):
    if x >= x0 and x <= x1 and y >= y0 and y <= y1:
      return True
    else:
      return False

  def load_image(self, fn):
    im = cv2.imread(fn)
    return im
