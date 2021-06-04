__author__      = 'Ernesto Coto'
__copyright__   = 'April 2019'

import os
import json
import numpy as np

def export_via_project(detections, outfile, config, frames_to_keep_per_track=None):
    """
        Exports VIA project with images ordered by their filename, which should be a string with
        6 decimal digits, plus the extension. The decimal digits in the filename represent
        the number of the frame within a video, starting from 000000 and up to the last
        frame in the video. This forces VIA to load track images in order with respect to
        the filenames.

        This exporter also removes from the project those files are not in the
        frames_to_keep_per_track parameter. If frames_to_keep_per_track is 'None'
        then no files are removed.

        The 'default_filepath' of the project is modified to assume the subdirectory
        of images is contained in the same directory as the VIA application.
        Arguments:
            detections: structure containing VIA data
            outfile: out VIA project file
            config: dictionary with some configuration settings
            frames_to_keep_per_track: dictionary of filenames lists, grouped by track-ids
    """
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
        "default_filepath": '.' + os.sep + os.path.basename(config['frame_img_dir']) + os.sep
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
    # add any new extra region attributes
    if 'extra_region_attributes' in config.keys():
        via_project['_via_attributes']['region'].update(config['extra_region_attributes'])
    # now save the image data to json, but only for those frames in the frame_track_map_shorten dictionary
    via_project['_via_img_metadata'] = {}
    for shot_id in detections.detection_data:
      for frame_id in detections.detection_data[shot_id]:
        frame_filename = detections.frame_id_to_filename_map[ frame_id ]
        # do a brute-force search of the frame in frames_to_keep_per_track
        frame_found = False
        if frames_to_keep_per_track is None:
            frame_found = True
        else:
            for track_id in frames_to_keep_per_track.keys():
                if frame_filename in frames_to_keep_per_track[track_id]:
                    frame_found = True
                    break
        if frame_found:
            frame_abs_path = os.path.join(config['frame_img_dir'],
                                          frame_filename)
            frame_filesize = os.path.getsize(frame_abs_path)
            fileid = frame_filename # this is important for the sorting specified when saving the json
            via_project['_via_img_metadata'][fileid] = {'filename':frame_filename,
                                                        'size':frame_filesize}
            via_project['_via_img_metadata'][fileid]['file_attributes'] = {'shot_id':shot_id,
                                                                           'frame_id':frame_id}
            via_project['_via_img_metadata'][fileid]['regions'] = []
            for box_id in detections.detection_data[shot_id][frame_id]:
              box = detections.detection_data[shot_id][frame_id][box_id]
              region_attributes = {'track_id': box[0], 'box_id': box_id }
              if 'extra_region_attributes' in config.keys() and len(box) > 5:
                  extra_region_counter = 5
                  for key in config['extra_region_attributes'].keys():
                      if extra_region_counter < len(box):
                        region_attributes.update( { key: box[extra_region_counter] } )
                        extra_region_counter = extra_region_counter + 1
              via_project['_via_img_metadata'][fileid]['regions'].append( {
                'shape_attributes':{'name':'rect', 'x':box[1], 'y':box[2], 'width':box[3], 'height':box[4]},
                'region_attributes': region_attributes
              } )

    with open(outfile, 'w') as jsonfile:
      json.dump(via_project, jsonfile, indent=None, separators=(',',':'), sort_keys=True)


def export_plain_csv(detections, outfile):
    """
        Exports VIA project to CSV file.
        It differs from the SVT exporter in one column: face_id.
        Arguments:
            detections: structure containing VIA data
            outfile: exported CSV file name
    """
    with open(outfile, 'w') as csvfile:
      csvfile.write('shot_id,frame_id,frame_filename,track_id,box_id,x,y,width,height,face_id\n')
      for shot_id in detections.detection_data:
        for frame_id in detections.detection_data[shot_id]:
          row_prefix = '%s,%s,"%s",' % (shot_id,
                                        frame_id,
                                        detections.frame_id_to_filename_map[ frame_id ])
          for box_id in detections.detection_data[shot_id][frame_id]:
            box = detections.detection_data[shot_id][frame_id][box_id]
            row_suffix1 = '%d,%s,%.3f,%.3f,%.3f,%.3f,%s\n' % (box[0], box_id, box[1], box[2], box[3], box[4], box[5])
            csvfile.write( row_prefix + row_suffix1 )


def compute_average_vectors_per_track(detections):
    """
        For each detection D on a track in the VIA data structure, this method
        takes the vector of face classification scores (position 5 in
        in the array D), and computes the average vector per track
        Arguments:
            detections: structure containing VIA data
        Returns:
            A dictionary containing the average vector per track
    """
    # gather face identifications per track
    frame_track_map = {}
    frame_id_avg_map = {}
    for shot_id in detections.detection_data:
      for frame_id in detections.detection_data[shot_id]:
          for box_id in detections.detection_data[shot_id][frame_id]:
            box = detections.detection_data[shot_id][frame_id][box_id]
            track_id = box[0]
            face_id_vector = box[5] # scores vector
            if track_id not in frame_track_map.keys():
                frame_track_map[track_id] = (1.0, face_id_vector)
            else:
                n = frame_track_map[track_id][0]
                frame_track_map[track_id] = (n + 1.0, np.add(frame_track_map[track_id][1], face_id_vector) )

    # select best (most common) face_id per track
    for track_id in frame_track_map.keys():
        frame_id_avg_map[track_id] = frame_track_map[track_id][1]/frame_track_map[track_id][0]

    return frame_id_avg_map


def export_average_vectors_csv(detections, outfile):
    """
        Exports the average vectors per track to a CSV file.
        Arguments:
            detections: structure containing VIA data
            outfile: exported CSV file name
    """
    frame_id_avg_map = compute_average_vectors_per_track(detections)
    with open(outfile, 'w') as csvfile:
        csvfile.write('track_id,average_vector\n')
        for track_id in frame_id_avg_map.keys():
            avg_str = np.array2string(frame_id_avg_map[track_id]).replace('\n', '')
            csvfile.write('%d,"%s"\n' % ( track_id, avg_str))


def choose_track_face_representative(detections, class_map):
    """
        For each detection D on a track in the VIA data structure, this method
        takes the vector of face classification scores (position 5 in
        in the array D), computes the average vector per track and selects
        the face id corresponding to the maximum value on the average vector,
        this is the face id that "represents" the track. D[5] is then modified
        to store just the "representative" face id.
        Arguments:
            detections: structure containing VIA data
            class_map: map from the index of the maximum value on the average vector,
                       to a face id.
    """
    # get average vectors
    frame_id_avg_map = compute_average_vectors_per_track(detections)

    # select best (most common) face_id per track
    frame_track_map = {}
    for track_id in frame_id_avg_map.keys():
        frame_id_avg = frame_id_avg_map[track_id]
        best_class = np.argmax(frame_id_avg)
        frame_track_map[track_id] = class_map[best_class]

    # modify info in original VIA structure with best face_id per track
    for shot_id in detections.detection_data:
      for frame_id in detections.detection_data[shot_id]:
          for box_id in detections.detection_data[shot_id][frame_id]:
            box = detections.detection_data[shot_id][frame_id][box_id]
            track_id = box[0]
            box[5] = frame_track_map[track_id] # note that here we erase the scores vector and replace it with a string


def get_representative_images_per_track(detections):
    """
        Given a VIA data structure containing images assigned to tracks,
        this method chooses "representative" images per track (the first
        and last frame). Those frames that are not assigned to any track
        (track_id == -1) are kept intact.

        The intention is to reduce the number of images referenced in the
        VIA data structure that is to be exported later, so that less images
        are saved with the project.
        Arguments:
            detections: structure containing VIA data
        Returns
            The list of images to be kept in the VIA data structure
    """
    frame_track_map = {}
    for shot_id in detections.detection_data:
      for frame_id in detections.detection_data[shot_id]:
          frame_filename = detections.frame_id_to_filename_map[ frame_id ]
          for box_id in detections.detection_data[shot_id][frame_id]:
            box = detections.detection_data[shot_id][frame_id][box_id]
            track_id = box[0]
            if track_id not in frame_track_map.keys():
              frame_track_map[track_id] = []
            frame_track_map[track_id].append( frame_filename )
            frame_track_map[track_id].sort()

    frame_track_map_keep = {}
    for track_id in frame_track_map.keys():
        frame_track_map_keep[track_id] = []
        if track_id >=0:
            # add the first and last frame of each valid track
            frame_track_map_keep[track_id].append( frame_track_map[track_id][0] )
            frame_track_map_keep[track_id].append( frame_track_map[track_id][-1] )
        else:
            # keep all frames that are not in any track (track_id == -1)
            frame_track_map_keep[track_id] = frame_track_map[track_id]

    return frame_track_map_keep

