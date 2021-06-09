##
## connect the object detections in multiple video frames
## obtained by a pretrained object detector
##
## Author:  Abhishek Dutta <adutta@robots.ox.ac.uk>
## Date:    10 Dec. 2018
##

import threading
import os
import csv
import json # for debug
import cv2
import numpy as np
import math
import urllib.request     # to download model file
import shutil             # to copy files

import svt.models as models
import torch
from functools import partial
import pickle
from torch.autograd import Variable
import torch.nn.functional as F

class siamrpn_tracker():
  def __init__(self, model_path=None, config=None):
    if model_path is None:
      raise ValueError('model_path must be provided')
    else:
      self.model_path = model_path

    self.state   = {}           # state of the tracker initialized with a template
    self.pretrained_model = {}  # placeholder for pretrained model loaded into CPU or GPU
    self.config = config
    self.use_gpu = True

    if self.config['download_model_if_missing']:
      self._download_model_if_missing(model_url=self.config['model_url'],
                                      model_path=self.model_path,
                                      force_update=self.config['force_model_download'])

    self._setup_gpu()
    self._preload_model();

  def init_tracker(self, template_img, template_bbox):
    self.state = {}
    template_bbox = [ int(template_bbox[1]),
                      int(template_bbox[2]),
                      int(template_bbox[3]),
                      int(template_bbox[4]) ]

    ## Initialize state with pytorch model
    self.state['model'] = self.pretrained_model;
    self._init_tracker_with_template(template_img, template_bbox);

  def track(self, search):
    context_length = 0.5 * ( self.state['target_size'][0] + self.state['target_size'][1] )
    search_square_crop_size = np.sqrt( (self.state['target_size'][0] + context_length) * (self.state['target_size'][1] + context_length) )
    scale_z = self.state['model_template_size'] / search_square_crop_size
    search_pad = ( self.state['model_search_size'] - self.state['model_template_size'] ) / 2
    search_pad_scaled = search_pad / scale_z
    search_length = self._round_python27( search_square_crop_size + 2*search_pad_scaled )

    search_subwindow = self._transform_img_for_model_input(search,
                                                           self.state['target_position'],
                                                           self.state['model_search_size'],
                                                           search_length,
                                                           self.state['template_img_channel_avg'])

    # track object
    search_subwindow_tensor = Variable( search_subwindow.unsqueeze(0) )

    if self.use_gpu:
      score, delta = self.state['model'].track( search_subwindow_tensor.cuda() )
    else:
      score, delta = self.state['model'].track( search_subwindow_tensor )

    search_pos = self.state['target_position']
    search_size = self.state['target_size'] * scale_z

    delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1).data.cpu().numpy()
    score = F.softmax(score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0), dim=1).data[:, 1].cpu().numpy()

    delta[0, :] = delta[0, :] * self.state['anchors'][:, 2] + self.state['anchors'][:, 0]
    delta[1, :] = delta[1, :] * self.state['anchors'][:, 3] + self.state['anchors'][:, 1]
    delta[2, :] = np.exp(delta[2, :]) * self.state['anchors'][:, 2]
    delta[3, :] = np.exp(delta[3, :]) * self.state['anchors'][:, 3]

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

    penalty = np.exp(-(r_c * s_c - 1) * self.state['penalty_k'])
    pscore = penalty * score

    # window float
    pscore = pscore * (1 - self.state['window_influence']) + self.state['window'] * self.state['window_influence']
    best_pscore_id = np.argmax(pscore)

    target = delta[:, best_pscore_id] / scale_z
    search_size = search_size / scale_z
    lr = penalty[best_pscore_id] * score[best_pscore_id] * self.state['lr']  # lr for OTB

    res_x = target[0] + search_pos[0]
    res_y = target[1] + search_pos[1]

    res_w = search_size[0] * (1 - self.state['lr']) + target[2] * self.state['lr']
    res_h = search_size[1] * (1 - self.state['lr']) + target[3] * self.state['lr']

    new_target_position = np.array([res_x, res_y])
    new_target_size     = np.array([res_w, res_h])
    new_target_position[0] = max(0, min(self.state['image_width'] , new_target_position[0]))
    new_target_position[1] = max(0, min(self.state['image_height'], new_target_position[1]))
    new_target_size[0] = max(10, min(self.state['image_width'], new_target_size[0]))
    new_target_size[1] = max(10, min(self.state['image_height'], new_target_size[1]))

    # update state
    self.state['target_position'] = new_target_position
    self.state['target_size']     = new_target_size
    self.state['score']           = score[best_pscore_id]

    self.state['track_count'] = self.state['track_count'] + 1
    return new_target_position, new_target_size, score[best_pscore_id]

  def _download_model_if_missing(self, model_url, model_path, force_update=False):
    try:
      if force_update:
        self._download_latest_model(model_url, model_path)
      else:
        if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
          self._download_latest_model(model_url, model_path)
    except:
      raise ValueError('Failed to download tracker model file')

  def _download_latest_model(self, url, file_path):
    try:
      print('Downloading latest model file from [%s]' % (url))
      # create parent folder in file_path, if it does not exist
      file_path_parent = os.path.dirname(file_path)
      if not os.path.isdir(file_path_parent):
        os.makedirs(file_path_parent)
      with urllib.request.urlopen(url) as response, open(file_path, 'wb') as f:
        print('Saving latest model to [%s]' % (file_path))
        shutil.copyfileobj(response, f)
    except:
      raise ValueError('Failed to download tracker model file from [%s] and save to [%s]' % (url, file_path))

  def _setup_gpu(self):
    try:
      if torch.cuda.is_available() and self.config['gpu_id'] != -1:
        self.use_gpu = True
        self.device = torch.device('cuda:' + str(self.config['gpu_id']))
        if self.config['verbose']:
          print('Using GPU %d' % (self.config['gpu_id']))
      else:
        self.use_gpu = False
        self.gpu_id = -1
        self.device = torch.device('cpu')
        if self.config['verbose']:
          print('Using CPU only')
    except:
      raise ValueError('Failed to setup GPU %d' %(self.config['gpu_id']))

  def _load_image(self, fn):
    im = cv2.imread(fn)
    return im

  ## routines to preload pytorch model
  def _preload_model(self):
    if self.config['verbose']:
      print('Preloading model [ %s ] ... ' % (self.model_path), end='', flush=True)

    ## @todo: get rid of absolute path to model file
    ## load configuration
    cfg_json_str = '''
    {
    "anchors": {
        "stride": 8,
        "ratios": [0.33, 0.5, 1, 2, 3],
        "scales": [8],
        "round_dight": 0
    },
    "hp": {
        "instance_size": 255,
        "search_model": "adapation",
        "penalty_k": 0.31,
        "window_influence": 0.448,
        "lr": 0.14
    }
    }'''
    cfg = json.loads(cfg_json_str)
    self.pretrained_model = models.Custom(anchors=cfg['anchors'])
    self._load_pretrained_model(self.model_path)
    self.pretrained_model.to(self.device) # move the pretrained model to GPU (if available)
    if self.config['verbose']:
      print('done', flush=True)

  def _check_keys(self, model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys

    #print('missing keys:')
    #print(missing_keys)
    #print('unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    #print('used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

  def _remove_prefix(self, state_dict, prefix):
    ''' Old style model is stored with all names of parameters share common prefix 'module.' '''
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


  def _load_pretrained_model(self, pretrained_path):
    ## bug fix
    ## see https://github.com/CSAILVision/places365/issues/25#issuecomment-333871990
    pickle.load = partial(pickle.load, encoding="latin1")
    pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
    #model = torch.load(model_file, map_location=lambda storage, loc: storage, pickle_module=pickle)

    #pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(self.device), pickle_module=pickle)
    if self.use_gpu:
      pretrained_dict = torch.load(pretrained_path,
                                   map_location=lambda storage,
                                   loc: storage.cuda(),
                                   pickle_module=pickle)
    else:
      pretrained_dict = torch.load(pretrained_path,
                                   map_location=lambda storage,
                                   loc: storage.cpu(),
                                   pickle_module=pickle)

    if "state_dict" in pretrained_dict.keys():
      pretrained_dict = self._remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
      pretrained_dict = self._remove_prefix(pretrained_dict, 'module.')
    self._check_keys(self.pretrained_model, pretrained_dict)
    self.pretrained_model.load_state_dict(pretrained_dict, strict=False)

  # see: https://github.com/python/cpython/blob/master/Python/pymath.c
  def _copysign_python27(self, x, y):
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
  def _round_python27(self, x):
    absx = math.fabs(x)
    y = math.floor(absx)
    if ( absx - y >= 0.5 ):
      y += 1.0
    return self._copysign_python27(y, x)

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
  def _transform_img_for_model_input(self, img, bbox_center, model_square_input_size, square_crop_size, img_channel_avg):
      # if the template is near image boundary, image channel average of the
      # template image is used to fill the empty regions
      if isinstance(bbox_center, float):
          bbox_center = [bbox_center, bbox_center]

      template_width  = img.shape[0]
      template_height = img.shape[1]
      template_center_to_boundary_length = (square_crop_size + 1) / 2
      context_xmin = self._round_python27(bbox_center[0] - template_center_to_boundary_length)
      context_xmax = context_xmin + square_crop_size - 1
      context_ymin = self._round_python27(bbox_center[1] - template_center_to_boundary_length)
      context_ymax = context_ymin + square_crop_size - 1
      left_pad = int( max(0., -context_xmin) )
      top_pad = int(max(0., -context_ymin))
      right_pad = int(max(0., context_xmax - template_height + 1))
      bottom_pad = int(max(0., context_ymax - template_width + 1))

      context_xmin = context_xmin + left_pad
      context_xmax = context_xmax + left_pad
      context_ymin = context_ymin + top_pad
      context_ymax = context_ymax + top_pad

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

      if not np.array_equal(model_square_input_size, square_crop_size):
          model_input = cv2.resize(square_img_data, (model_square_input_size, model_square_input_size))
      else:
          model_input = square_img_data

      model_input_tensor = torch.from_numpy( np.transpose(model_input, (2, 0, 1)) ).float()  # Channel x Height x Width

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
  def _generate_anchor(self, total_stride, anchor_scale_list, anchor_aspect_ratio_list, square_feature_map_length):
    anchor_count = len(anchor_aspect_ratio_list) * len(anchor_scale_list)
    anchors = np.zeros((anchor_count, 4), dtype=np.float32)
    initial_anchor_area = total_stride * total_stride;
    anchor_count = 0
    for anchor_aspect_ratio in anchor_aspect_ratio_list:
      anchor_width = int( np.sqrt( initial_anchor_area / anchor_aspect_ratio ) )
      anchor_height  = int( anchor_width * anchor_aspect_ratio )
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
          anchors_ij[0] = offset_from_center_of_feature_map + total_stride * j
          anchors_ij[1] = offset_from_center_of_feature_map + total_stride * i
          feature_map_anchors[anchor_index, i, j] = anchors_ij

    feature_map_anchors = np.reshape(feature_map_anchors, (-1, 4)) # collapse the (i,j) dimension of feature map as it is not needed
    return feature_map_anchors

  def _init_tracker_with_template(self, template, template_bbox):
    self.state['model'].eval();
    if self.use_gpu:
      self.state['model'] = self.state['model'].cuda();

    self.state['image_width'] = template.shape[1]
    self.state['image_height'] = template.shape[0]

    target_cx = template_bbox[0] + template_bbox[2]/2
    target_cy = template_bbox[1] + template_bbox[3]/2
    target_w  = template_bbox[2]
    target_h  = template_bbox[3]

    self.state['template_position'] = np.array( [target_cx, target_cy] )
    self.state['template_size']     = np.array( [target_w , target_h ] )

    self.state['target_position'] = self.state['template_position']
    self.state['target_size']     = self.state['template_size']
    self.state['model_template_size'] = 127         # 127x127
    self.state['model_search_size'] = 255           # 255x255
    self.state['total_stride'] = 8
    self.state['penalty_k'] = 0.31
    self.state['window_influence'] = 0.448
    self.state['lr'] = 0.14
    self.state['search_model'] = 'adaption'
    self.state['anchor_aspect_ratio_list'] = [0.33, 0.5, 1, 2, 3]
    self.state['anchor_scale_list'] = [8, ]
    self.state['anchor_count'] = len(self.state['anchor_aspect_ratio_list']) * len(self.state['anchor_scale_list'])

    if self.state['search_model'] == 'adaption':
      if ( (self.state['target_size'][0] * self.state['target_size'][1]) / (float(self.state['image_width'] * self.state['image_height'])) ) < 0.004:
        self.state['model_search_size'] = 287 # small object big search region
      else:
        self.state['model_search_size'] = 271

    # 17x17 for model_search_size = 255
    # 19x19 for model_search_size = 271 # OTB2017 dataset
    self.state['square_feature_map_length'] = int( (self.state['model_search_size'] - self.state['model_template_size']) / self.state['total_stride'] + 1)

    self.state['anchors'] = self._generate_anchor(self.state['total_stride'],
                                                  self.state['anchor_scale_list'],
                                                  self.state['anchor_aspect_ratio_list'],
                                                  self.state['square_feature_map_length'])
    self.state['context'] = 0.5
    context_length = 0.5 * ( self.state['target_size'][0] + self.state['target_size'][1] )
    square_crop_size = round( np.sqrt( (self.state['target_size'][0] + context_length) * (self.state['target_size'][1] + context_length) ) ) # see equation (15) of [Li et al. 2018]

    ## initialize model with template image
    self.state['template_img_channel_avg'] = np.mean(template, axis=(0, 1))

    template_subwindow = self._transform_img_for_model_input(template,
                                                             self.state['target_position'],
                                                             self.state['model_template_size'],
                                                             square_crop_size,
                                                             self.state['template_img_channel_avg'])

    template_subwindow_tensor = Variable(template_subwindow.unsqueeze(0))

    if self.use_gpu:
      self.state['model'].temple( template_subwindow_tensor.cuda() )
    else:
      self.state['model'].temple( template_subwindow_tensor )

    # cosine window
    self.state['window'] = np.outer(np.hanning(self.state['square_feature_map_length']), np.hanning(self.state['square_feature_map_length']))
    self.state['window'] = np.tile(self.state['window'].flatten(), self.state['anchor_count'])

    self.state['track_count'] = 0
