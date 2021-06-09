import json
import threading
import time
import random
from hashlib import blake2b
import copy
import os

from svt.buffered_input import buffered_input

# required by the tracker
import svt.models as models
import torch
import numpy as np
import time
import math
import cv2 # needed for cv2.resize()
from torch.autograd import Variable
import torch.nn.functional as F

class svt_task(threading.Thread):

  def __init__(self, model, task_id, task_definition, outdir=None, outfile=None, use_gpu=False):
    threading.Thread.__init__(self)
    self.original_model = model
    self.task_id = task_id
    self.task_definition = json.loads(task_definition)
    self.task_result = json.loads(task_definition)
    self.use_gpu = use_gpu
    self.time_to_index = {}
    self.task_status = { 'now':[], 'done':[], 'all_done':False }
    self.task_state = 'INIT'
    self.stop_task = False
    self.outdir = outdir
    self.outfile = outfile
    self.FFMPEG_BIN  = 'ffmpeg'
    print('Created task %s' % (self.task_id) )

  def stop(self):
    print('Stopping task [%s]. Please wait ...'%(self.task_id))
    self.stop_task = True
    self.save_results() # save even the temporary results

  def save_results(self):
    if self.outdir is not None or self.outfile is not None:
      if self.outfile is not None:
        task_result_filename = self.outfile
      else:
        video_filename = self.task_definition['files'][0]
        video_name = os.path.splitext(os.path.basename(video_filename))[0]
        task_result_filename = os.path.join(self.outdir, video_name + '_' + self.task_id + '.json')
      with open(task_result_filename, 'w') as f:
        json.dump( self.task_result, f, separators=(',',':') )
      print('Task [%s] tracking data saved to file [%s]' % (self.task_id, task_result_filename))

  # in Windows platform, filenames are file URI (e.g. file:///C:/Dataset/video.mp4)
  def resolve_video_file_path(self):
    video_filename = self.task_result['files'][0]
    if video_filename.startswith('file:///'):
      path1 = video_filename[8:]
      if os.path.isfile(video_filename[8:]):
        return video_filename[8:]
      else:
        if os.path.isfile(video_filename[7:]):
          return video_filename[7:]
        else:
          ## file does not exist
          print('Filename URI [%s] cannot be resolved to a local file!' %(video_filename))
          return None;
    else:
      if os.path.isfile(video_filename):
        return video_filename
      else:
        print('Filename [%s] does not exist!')
        return None;

  def run(self):
    self.task_state = 'START'
    ## assert that the video file exists
    video_filename = self.resolve_video_file_path()
    if video_filename is None:
      self.task_state = 'ERROR'
      self.stop()
      return

    print('Running task %s' % (self.task_id))
    self.model = copy.deepcopy(self.original_model) # create a copy of model

    self.task_state = 'ONGOING'
    start_time = time.time()
    for oid in self.task_result['obj']:
      self.time_to_index[oid] = {}
      for tid in self.task_result['obj'][oid]['trk']:
        self.time_to_index[oid][tid] = {}
        self.init_all_segments(oid, tid) ## initialize segments if it is missing
        for sid in self.task_result['obj'][oid]['trk'][tid]['seg']:
          self.time_to_index[oid][tid][sid] = {}
          tstart = self.task_result['obj'][oid]['trk'][tid]['seg'][sid]['start_region'][0]
          region = self.task_result['obj'][oid]['trk'][tid]['seg'][sid]['start_region'][2:6]
          tend = self.task_result['obj'][oid]['trk'][tid]['seg'][sid]['end_region'][0]

          # pushes results to self.task_definition['obj'][oid]['trk'][tid]['seg'][sid].regions[ ... ]
          self.generate_track('video_file', video_filename, tstart, region, tend, oid, tid, sid)
          if self.stop_task:
            return
    end_time = time.time()
    self.task_status['now'] = []
    self.task_status['all_done'] = True
    self.task_state = 'DONE'
    self.save_results()
    print('Completed task %s in %.2f seconds.' % (self.task_id, (end_time - start_time)))

  # initial track segments consists of manual regions
  def init_all_segments(self, oid, tid):
    existing_sid_list = list(self.task_result['obj'][oid]['trk'][tid]['seg'].keys())
    if len(existing_sid_list) != 0:
      return

    manual_rid_list = list(self.task_result['obj'][oid]['trk'][tid]['manual_regions'].keys())
    if len(manual_rid_list) == 1:
      self.init_segment(oid, tid, manual_rid_list[0], None);
    else:
      for i in range(0, len(manual_rid_list)-1):
        self.init_segment(oid, tid, manual_rid_list[i], manual_rid_list[i+1])

  def init_segment(self, oid, tid, start_manual_region_id, end_manual_region_id=None):
    sid = start_manual_region_id
    self.task_result['obj'][oid]['trk'][tid]['seg'][sid] = {}
    self.task_result['obj'][oid]['trk'][tid]['seg'][sid]['start_region'] = self.task_result['obj'][oid]['trk'][tid]['manual_regions'][start_manual_region_id]
    self.task_result['obj'][oid]['trk'][tid]['seg'][sid]['regions'] = []
    self.task_result['obj'][oid]['trk'][tid]['seg'][sid]['regions'].append( self.task_result['obj'][oid]['trk'][tid]['manual_regions'][start_manual_region_id] )
    if end_manual_region_id is None:
      # track until the end
      self.task_result['obj'][oid]['trk'][tid]['seg'][sid]['end_region'] = [-1,0,0,0,0,0]
    else:
      self.task_result['obj'][oid]['trk'][tid]['seg'][sid]['end_region'] = self.task_result['obj'][oid]['trk'][tid]['manual_regions'][end_manual_region_id]

  def generate_track(self, input_type, input_file, tstart, region, tend, oid, tid, sid):
    print('Processing video %s (%f,%f) : %s:%s:%s' %(input_file, tstart, tend, oid, tid, sid))

    ## Initiaze state with pytorch model
    state = {}
    state['model'] = self.model;

    ## Initialize buffered input reader
    input_data = buffered_input( input_type, input_file, tstart, tend )
    input_data.start() # start loading frames in buffer

    ## Define template and initialize tracker with this template
    # @Li always subtracts (1,1) from supplied (x,y) coordinates
    template_bbox = [ int(region[0]) - 1, int(region[1]) - 1, int(region[2]), int(region[3]) ]
    template, template_frame_offset, template_frame_name = input_data.get_next_frame();
    #print( 'Using template from [%d] %s' % (template_frame_offset, template_frame_name) )
    state = self.init_tracker_with_template(state, template, template_bbox);

    ## Perform tracking on each subsequent frames
    if tend == -1:
      # track until end of video
      tend = input_data.state['input_duration']

    current_time = tstart + input_data.state['frame_time_interval']
    stop_time = tend + input_data.state['frame_time_interval']
    offset_from_start = int(1) # since first frame is used as template
    while input_data.frame_available() \
      and not self.stop_task \
      and current_time < stop_time:
      search, search_frame_offset, search_frame_name = input_data.get_next_frame()
      pos, size, score = self.track(state, search);
      current_time_str = self.get_time_str(current_time)
      bbox = [ float(current_time_str), 0, int(pos[0] - size[0]/2), int(pos[1] - size[1]/2), int(size[0]), int(size[1]) ]

      #print('[%d] Tracking frame [%d] %s, time=%f : score = %.6f, bbox = %s' % ( state['track_count'], search_frame_offset, search_frame_name, current_time, score, ('[%.3f, %.3f, %.3f, %.3f]' % (bbox[2],bbox[3],bbox[4],bbox[5])) ))
      self.task_result['obj'][oid]['trk'][tid]['seg'][sid]['regions'].append(bbox)
      self.task_status['now'] = [oid, tid, sid, offset_from_start, current_time_str]
      current_time = current_time + input_data.state['frame_time_interval']
      offset_from_start = offset_from_start + 1

    offset_from_start = offset_from_start - 1
    true_end = current_time - input_data.state['frame_time_interval']
    true_end_str = self.get_time_str(true_end)
    self.task_status['now'] = []
    self.task_status['done'].append( [ oid, tid, sid, offset_from_start, true_end_str ] )

    ## check if program termination was requested
    if self.stop_task:
      input_data.stop_buffered_input()

  def get_time_str(self, time_value):
    return '%.6f' % (time_value)

  def load_image(self, img_filename):
    img = np.array(Image.open(img_filename))
    return img


  # see: https://github.com/python/cpython/blob/master/Python/pymath.c
  def copysign_python27(self, x, y):
    if ( y > 0. or ( y == 0. and math.atan2(y, -1.) > 0. ) ):
      return math.fabs(x);
    else:
      return -math.fabs(x);

  #
  # `round()` method in python:
  # if python2.7, round(0.5) = 1.0
  #   [if two multiples are equally close, rounding is done away from 0 -- https://docs.python.org/2/library/functions.html#round]
  # if python3.7, round(0.5) = 0.0
  #   [if two multiples are equally close, rounding is done toward the even choice -- https://docs.python.org/3/library/functions.html#round]
  #
  def round_python27(self, x):
    absx = math.fabs(x)
    y = math.floor(absx)
    if ( absx - y >= 0.5 ):
      y += 1.0
    return self.copysign_python27(y, x)

  # To track an object, the user selects a region containing this object in a
  # given image frame. This region is called template. The user selected template
  # can be of any size and aspect ratio. Therefore, this template needs to be
  # transformed into an image size that is accepted as input to the model
  #
  # Input
  #   img                         full frame image containing the template
  #   bbox_center                 center coordinates of the bounding box containing the template
  #   model_square_input_size     size of the model input (square shaped image) for template
  #   square_crop_size            size of the square to which the user selected template is expanded (to get additional context)
  # Returns
  #   a square image of size [model_square_input_size x model_square_input_size]
  #   containing the user selected object and some context around it
  def transform_img_for_model_input(self, img, bbox_center, model_square_input_size, square_crop_size, img_channel_avg):
      # if the template is near image boundary, image channel average of the
      # template image is used to fill the empty regions
      if isinstance(bbox_center, float):
          bbox_center = [bbox_center, bbox_center]

      template_width  = img.shape[0]
      template_height = img.shape[1]
      template_center_to_boundary_length = (square_crop_size + 1) / 2
      context_xmin = self.round_python27(bbox_center[0] - template_center_to_boundary_length)
      context_xmax = context_xmin + square_crop_size - 1
      context_ymin = self.round_python27(bbox_center[1] - template_center_to_boundary_length)
      context_ymax = context_ymin + square_crop_size - 1
      left_pad = int( max(0., -context_xmin) )
      top_pad = int(max(0., -context_ymin))
      right_pad = int(max(0., context_xmax - template_height + 1))
      bottom_pad = int(max(0., context_ymax - template_width + 1))

      #print('bbox_center=(%.2f,%.2f)' % (bbox_center[0], bbox_center[1]) )
      #print('template_center_to_boundary_length=%.1f' % (template_center_to_boundary_length) )
      #print('\ncontext_x(%.2f,%.2f), context_y(%.2f,%.2f), pad(%.2f, %.2f)' % (context_xmin, context_xmax, context_ymin, context_ymax, left_pad, top_pad))
      #print('\npad(%.2f,%.2f,%.2f,%.2f)' % (top_pad, right_pad, bottom_pad, left_pad))
      #print('\nimg_channel_avg = ')
      #print(img_channel_avg)
      #print('\nimg[129,133:143,] = ')
      #print(img.shape)
      #print(img[129,133:143,])
      #print(img[66,308:320,])
      # why is this image's blue channel data different from @li?
      # this discripency is reflected in the image channelwise average
      # [ 93.9404783  116.46350592  80.49950197] @mine
      # [ 80.50766272 116.46360454  93.94219921] @li

      context_xmin = context_xmin + left_pad
      context_xmax = context_xmax + left_pad
      context_ymin = context_ymin + top_pad
      context_ymax = context_ymax + top_pad

      #print('\nany([top_pad, bottom_pad, left_pad, right_pad]) = %d' % (any([top_pad, bottom_pad, left_pad, right_pad])))
      r, c, k = img.shape
      if any([top_pad, bottom_pad, left_pad, right_pad]):
          ## fill average image colour if the template region is near the boundary of image
          te_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad, k), np.uint8)  # 0 is better than 1 initialization
          te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = img
          if top_pad:
              te_im[0:top_pad, left_pad:left_pad + c, :] = img_channel_avg
          if bottom_pad:
              te_im[r + top_pad:, left_pad:left_pad + c, :] = img_channel_avg
          if left_pad:
              te_im[:, 0:left_pad, :] = img_channel_avg
          if right_pad:
              te_im[:, c + left_pad:, :] = img_channel_avg
          square_img_data = te_im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]
      else:
          square_img_data = img[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]

      #print('\nnot np.array_equal(model_square_input_size, square_crop_size) = %d' % (not np.array_equal(model_square_input_size, square_crop_size)))

      #print('square_img_data[%d:%d, %d:%d, :]' % (int(context_ymin),int(context_ymax + 1), int(context_xmin), int(context_xmax + 1)) )
      #print('square_img_data = reshape to %d x %d' % (model_square_input_size, model_square_input_size))
      #print(square_img_data.shape)
      #print(square_img_data.__class__)
      #print('square_img_data[0,7:30,:] = ')
      #print(square_img_data[0,7:30,:])

      if not np.array_equal(model_square_input_size, square_crop_size):
          # resize the cropped image to the template size expected by model
          # since resize operation can only be done on PIL image, we convert
          # numpy.ndarray to PIL image, resize and convert it back to numpy.ndarray
          #square_img = Image.fromarray(square_img_data)
          #model_input_img = ImageOps.fit(square_img, (model_square_input_size, model_square_input_size), Image.ANTIALIAS)
          #model_input = np.array(model_input_img)
          model_input = cv2.resize(square_img_data, (model_square_input_size, model_square_input_size))
      else:
          model_input = square_img_data

      ## NOT NEEDED ANYMORE BECAUSE WE ARE
      ## IMPORTANT: model_input is in RGB format, Height x Width x Channel
      ## Model was trained on images in BGR format, so convert images from RGB to BGR
      ## required operations: change to BGR format and Channel x Height x Width
      #model_input_red = np.copy(model_input[:,:,0])
      #model_input[:,:,0] = model_input[:,:,2]
      #model_input[:,:,2] = model_input_red

      model_input_tensor = torch.from_numpy( np.transpose(model_input, (2, 0, 1)) ).float()  # Channel x Height x Width

      #print('model_input_tensor[:,30:40,43] = ')
      #print(model_input_tensor[:,30:40,43])
      return model_input_tensor

  ## generate anchors
  ## [arguments]
  ##  total_stride                : (search_input_size - template_input_size) / total_stride + 1 = size of square feature map
  ##  anchor_scale_list           : list of scales by which all initial anchors will be scaled
  ##  anchor_aspect_ratio_list    : list of aspect ratio for all initial anchors
  ##  square_feature_map_length   : the dimension of final scores generated by classification and regression branches of region proposal network
  ##
  ## [description]
  ## the area of all generated anchors must be same as the area of initial anchor
  ## therefore, for all generate anchors of dimension Aw x Ah, and aspect ratio = aspect_ratio
  ##  Ah = Aw * aspect_ratio         ---- (1) by definition of aspect ratio
  ##  Aw * Ah = initial_anchor_area  ---- (2) because the area of anchor remains constant
  ##
  ## where, initial_anchor_area = 4 * total_stride
  ##
  ## Therefore, substituting values of aspect_ratio and initial_anchor_area in (1) and (2)
  ## and substituting (1) in (2), we get
  ##
  ##      Aw * Aw * aspect_ratio = initial_anchor_area
  ## or,  Aw = sqrt( initial_anchor_area / aspect_ratio )
  ## and substituting value of Aw in (2), we get the value of Ah of the new anchor
  ##
  ## we scale each anchor using the scale provided in anchor_scale_list
  def generate_anchor(self, total_stride, anchor_scale_list, anchor_aspect_ratio_list, square_feature_map_length):
      anchor_count = len(anchor_aspect_ratio_list) * len(anchor_scale_list)
      anchors = np.zeros((anchor_count, 4), dtype=np.float32)
      initial_anchor_area = total_stride * total_stride;
      anchor_count = 0
      for anchor_aspect_ratio in anchor_aspect_ratio_list:
        anchor_width = int( np.sqrt( initial_anchor_area / anchor_aspect_ratio ) )
        anchor_height  = int( anchor_width * anchor_aspect_ratio )
        ## check: (anchor_height * anchor_width) = initial_anchor_area
        for anchor_scale in anchor_scale_list:
          anchor_scaled_height = anchor_height * anchor_scale
          anchor_scaled_width  = anchor_width  * anchor_scale
          anchors[anchor_count, 0] = 0  # will be updated later
          anchors[anchor_count, 1] = 0  # will be updated later
          anchors[anchor_count, 2] = anchor_scaled_width
          anchors[anchor_count, 3] = anchor_scaled_height
          anchor_count = anchor_count + 1

      feature_map_anchors = np.zeros( (anchor_count, square_feature_map_length, square_feature_map_length, 4), dtype=np.float32 )
      center_of_feature_map = (square_feature_map_length - 1 ) / 2 # Li uses ori = square_feature_map_length / 2
      offset_from_center_of_feature_map = -center_of_feature_map * total_stride;

      for anchor_index in range(anchor_count):
        anchor = anchors[anchor_index]
        for i in range(square_feature_map_length):
          for j in range(square_feature_map_length):
            anchors_ij = np.copy(anchor)
            # update the (x,y) coordinate of each anchor for feature map location (i,j)
            # @todo
            anchors_ij[0] = offset_from_center_of_feature_map + total_stride * j
            anchors_ij[1] = offset_from_center_of_feature_map + total_stride * i
            feature_map_anchors[anchor_index, i, j] = anchors_ij

      feature_map_anchors = np.reshape(feature_map_anchors, (-1, 4)) # collapse the (i,j) dimension of feature map as it is not needed
      return feature_map_anchors

  def init_tracker_with_template(self, state, template, template_bbox):
    state['model'].eval();
    if self.use_gpu:
      state['model'] = state['model'].cuda();

    state['image_width'] = template.shape[1]
    state['image_height'] = template.shape[0]

    target_cx = template_bbox[0] + template_bbox[2]/2
    target_cy = template_bbox[1] + template_bbox[3]/2
    target_w  = template_bbox[2]
    target_h  = template_bbox[3]

    state['template_position'] = np.array( [target_cx, target_cy] )
    state['template_size']     = np.array( [target_w , target_h ] )

    state['target_position'] = state['template_position']
    state['target_size']     = state['template_size']
    state['model_template_size'] = 127         # 127x127
    state['model_search_size'] = 255           # 255x255
    state['total_stride'] = 8
    state['penalty_k'] = 0.31
    state['window_influence'] = 0.448
    state['lr'] = 0.14
    state['search_model'] = 'adaption'
    state['anchor_aspect_ratio_list'] = [0.33, 0.5, 1, 2, 3]
    state['anchor_scale_list'] = [8, ]
    state['anchor_count'] = len(state['anchor_aspect_ratio_list']) * len(state['anchor_scale_list'])

    if state['search_model'] == 'adaption':
      if ( (state['target_size'][0] * state['target_size'][1]) / (float(state['image_width'] * state['image_height'])) ) < 0.004:
        state['model_search_size'] = 287 # small object big search region
      else:
        state['model_search_size'] = 271

    # 17x17 for model_search_size = 255
    # 19x19 for model_search_size = 271 # OTB2017 dataset
    state['square_feature_map_length'] = int( (state['model_search_size'] - state['model_template_size']) / state['total_stride'] + 1)

    state['anchors'] = self.generate_anchor(state['total_stride'], state['anchor_scale_list'], state['anchor_aspect_ratio_list'], state['square_feature_map_length'])
    state['context'] = 0.5
    context_length = 0.5 * ( state['target_size'][0] + state['target_size'][1] )
    square_crop_size = round( np.sqrt( (state['target_size'][0] + context_length) * (state['target_size'][1] + context_length) ) ) # see equation (15) of [Li et al. 2018]

    ## initialize model with template image
    #print('Initializing model with template image')
    state['template_img_channel_avg'] = np.mean(template, axis=(0, 1))
    #print('template_img_channel_avg=')
    #print(template.shape)
    #print(template.__class__)
    #print(template.dtype)
    #print(np.version.version)
    #print(state['template_img_channel_avg'])

    template_subwindow = self.transform_img_for_model_input(template, state['target_position'], state['model_template_size'], square_crop_size, state['template_img_channel_avg'])
    #template_subwindow_img = Image.fromarray(template_subwindow)
    #template_subwindow_img.save('/home/tlm/exp/svt2/tools/0001_crop.jpg')

    template_subwindow_tensor = Variable(template_subwindow.unsqueeze(0))

    #print('template_subwindow_tensor')
    #print(template_subwindow_tensor.shape)
    #print(template_subwindow_tensor.__class__)
    #print(template_subwindow_tensor.dtype)
    #print(template_subwindow_tensor[0,0,43:50,67:74])
    #print(template_subwindow_tensor[0,0:3,0,0])

    if self.use_gpu:
      state['model'].temple( template_subwindow_tensor.cuda() )
    else:
      state['model'].temple( template_subwindow_tensor )

    # cosine window
    state['window'] = np.outer(np.hanning(state['square_feature_map_length']), np.hanning(state['square_feature_map_length']))
    state['window'] = np.tile(state['window'].flatten(), state['anchor_count'])

    state['track_count'] = 0

    #print( 'state[] : ' )
    #print( state['anchor_scale_list'] )
    #print( state['anchor_aspect_ratio_list'] )
    #print( state['anchor_count'] )
    #print( state['anchors'][943:955,] )
    #print( state['anchors'].shape )
    return state

  def track(self, state, search):
    context_length = 0.5 * ( state['target_size'][0] + state['target_size'][1] )
    search_square_crop_size = np.sqrt( (state['target_size'][0] + context_length) * (state['target_size'][1] + context_length) )
    scale_z = state['model_template_size'] / search_square_crop_size
    search_pad = ( state['model_search_size'] - state['model_template_size'] ) / 2
    search_pad_scaled = search_pad / scale_z
    search_length = self.round_python27( search_square_crop_size + 2*search_pad_scaled )
    #print('context_length=%.2f' %(context_length))
    #print('search_square_crop_size=%.2f' %(search_square_crop_size))
    #print('scale_z=%.2f' %(scale_z))
    #print('search_pad=%.2f' %(search_pad))
    #print('search_pad_scaled=%.2f' %(search_pad_scaled))
    #print('search_length=%.2f' %(search_length))

    search_subwindow = self.transform_img_for_model_input(search, state['target_position'], state['model_search_size'], search_length, state['template_img_channel_avg'])
    #search_subwindow_img = Image.fromarray(search_subwindow)
    #search_subwindow_img.save('/home/tlm/exp/svt2/tools/0002_crop.jpg')

    # track object
    #tracker_eval(net, x_crop, target_pos, target_sz, window, scale_z, p):
    #tracker_eval(net, x_crop.cuda(), target_pos, target_sz * scale_z, window, scale_z, p)
    search_subwindow_tensor = Variable( search_subwindow.unsqueeze(0) )

    if self.use_gpu:
      score, delta = state['model'].track( search_subwindow_tensor.cuda() )
    else:
      score, delta = state['model'].track( search_subwindow_tensor )

    #print('score = ')
    #print(score.shape)
    #print(score[0,3:7,7:13,7])
    #print(score[0,0:3,1:7,7])
    #print('delta = ')
    #print(delta.shape)
    #print(delta[0,3:7,7:13,7])
    #print(delta[0,0:3,1:7,7])

    search_pos = state['target_position']
    search_size = state['target_size'] * scale_z

    delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1).data.cpu().numpy()
    score = F.softmax(score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0), dim=1).data[:, 1].cpu().numpy()

    delta[0, :] = delta[0, :] * state['anchors'][:, 2] + state['anchors'][:, 0]
    delta[1, :] = delta[1, :] * state['anchors'][:, 3] + state['anchors'][:, 1]
    delta[2, :] = np.exp(delta[2, :]) * state['anchors'][:, 2]
    delta[3, :] = np.exp(delta[3, :]) * state['anchors'][:, 3]

    def change(r):
        return np.maximum(r, 1./r)

    def sz(w, h):
        pad = (w + h) * 0.5
        sz2 = (w + pad) * (h + pad)
        return np.sqrt(sz2)

    def sz_wh(wh):
        pad = (wh[0] + wh[1]) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return np.sqrt(sz2)

    # size penalty
    s_c = change( sz(delta[2, :], delta[3, :]) / sz_wh(search_size) )  # scale penalty
    r_c = change( (search_size[0] / search_size[1]) / (delta[2, :] / delta[3, :]) )  # ratio penalty

    penalty = np.exp(-(r_c * s_c - 1) * state['penalty_k'])
    pscore = penalty * score

    # window float
    pscore = pscore * (1 - state['window_influence']) + state['window'] * state['window_influence']
    best_pscore_id = np.argmax(pscore)

    target = delta[:, best_pscore_id] / scale_z
    search_size = search_size / scale_z
    lr = penalty[best_pscore_id] * score[best_pscore_id] * state['lr']  # lr for OTB

    res_x = target[0] + search_pos[0]
    res_y = target[1] + search_pos[1]

    res_w = search_size[0] * (1 - state['lr']) + target[2] * state['lr']
    res_h = search_size[1] * (1 - state['lr']) + target[3] * state['lr']

    new_target_position = np.array([res_x, res_y])
    new_target_size     = np.array([res_w, res_h])
    new_target_position[0] = max(0, min(state['image_width'] , new_target_position[0]))
    new_target_position[1] = max(0, min(state['image_height'], new_target_position[1]))
    new_target_size[0] = max(10, min(state['image_width'], new_target_size[0]))
    new_target_size[1] = max(10, min(state['image_height'], new_target_size[1]))

    # update state
    state['target_position'] = new_target_position
    state['target_size']     = new_target_size
    state['score']           = score[best_pscore_id]

    state['track_count'] = state['track_count'] + 1
    return new_target_position, new_target_size, score[best_pscore_id]
