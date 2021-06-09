##
## task manager for SVT server
##
## Author:  Abhishek Dutta <adutta@robots.ox.ac.uk>
## Date:    15 Nov. 2018
##

import json;                  # to parse json sent via http request
from hashlib import blake2b   # to compute hash for task-id
import queue                  # to maintain a task queue
import threading
import os

from svt.svt_task import svt_task

import svt.models as models
import torch
from functools import partial
import pickle

class svt_task_manager(threading.Thread):

  def __init__(self, model_path=None, track_data_dir=None, task_id_size=10, task_queue_size=4, stop_when_empty=False, gpu_id=0):
    threading.Thread.__init__(self)
    if model_path is None:
      raise ValueError('SVT pytorch model path not defined!')
    else:
      self.model_path = model_path

    self.stop_task_manager = False
    self.task_id_size = task_id_size
    self.task_list = {}
    self.stop_when_empty = stop_when_empty

    self.use_gpu = True
    self.gpu_id = gpu_id
    self.setup_gpu()

    print('Initializing task manager ...')
    self.task_queue = queue.Queue( maxsize=task_queue_size )
    self.task_result = {}

    print('Preloading model [ %s ] ...' % (self.model_path))
    self.preload_model();
    self.track_data_dir = track_data_dir

  def setup_gpu(self):
    try:
      if torch.cuda.is_available() and self.gpu_id != -1:
        self.use_gpu = True
        self.device = torch.device('cuda:' + str(self.gpu_id))
        print('Using GPU %d' % (self.gpu_id))
      else:
        self.use_gpu = False
        self.gpu_id = -1
        self.device = torch.device('cpu')
        print('Using CPU only')
    except:
      raise ValueError('Failed to setup GPU %d' %(self.gpu_id))

  def stop(self):
    print('Stopping task manager ...')
    self.stop_task_manager = True
    for task_id in self.task_list:
      if self.task_list[ task_id ].is_alive():
        self.task_list[ task_id ].stop()

  def stop_task(self, task_id):
    print('Stopping task %s ...' %(task_id))
    if self.task_list[ task_id ].is_alive():
      self.task_list[ task_id ].stop()
      return True
    else:
      return False

  def get_task_id(self, task_definition):
    task_id_generator = blake2b( digest_size=self.task_id_size )
    task_id_generator.update( str(task_definition).encode('utf-8'))
    return task_id_generator.hexdigest()

  def _queue_task(self, task_definition, outfile):
    task_id = self.get_task_id(task_definition)
    if task_id not in self.task_list:
      self.task_list[ task_id ] = svt_task(self.model_pretrained, task_id, task_definition, outdir=self.track_data_dir, outfile=outfile, use_gpu=self.use_gpu )
      self.task_queue.put( task_id, block=True, timeout=None)
    else:
      print( 'Task %s already exists, reusing values' %(task_id))
    return task_id

  def create_task(self, task_definition, outfile=None):
    return self._queue_task(task_definition, outfile)

  def get_all_task_result(self, task_id):
    return self.task_list[task_id].task_result

  # fetch_list = [[oid, tid, rid, tstart, tend], ...]
  def get_task_result(self, task_id, fetch_list):
    n = len(fetch_list)
    d = {}
    d['fetch_list'] = fetch_list
    d['fetch_result'] = []
    for i in range(0,n):
      oid = fetch_list[i][0]
      tid = fetch_list[i][1]
      sid = fetch_list[i][2]
      offset_from = int(fetch_list[i][3]) + 1 # excluding the value at this offset
      offset_end  = len(self.task_list[task_id].task_result['obj'][oid]['trk'][tid]['seg'][sid]['regions'])

      if offset_from < offset_end:
        d['fetch_result'].append( self.task_list[task_id].task_result['obj'][oid]['trk'][tid]['seg'][sid]['regions'][offset_from:offset_end] )

    return d

  def get_task_status(self, task_id):
    return self.task_list[task_id].task_status

  def get_task_state(self, task_id):
    return self.task_list[task_id].task_state

  def run(self):
    while not self.stop_task_manager:
      try:
        task_id = self.task_queue.get(block=True, timeout=3)
        self.task_list[ task_id ].start()
        self.task_list[ task_id ].join() # wait until this task completes
        self.task_queue.task_done()
      except queue.Empty:
        if self.stop_when_empty:
          self.stop_task_manager = True
          break;
        else:
          pass # continue waiting for more tasks to be added


  ## routines to preload pytorch model
  def preload_model(self):
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
    self.model = models.Custom(anchors=cfg['anchors'])

    self.model_pretrained = self.load_pretrain(self.model, self.model_path)
    self.model_pretrained.to(self.device)

  def check_keys(self, model, pretrained_state_dict):
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

  def remove_prefix(self, state_dict, prefix):
      ''' Old style model is stored with all names of parameters share common prefix 'module.' '''
      f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
      return {f(key): value for key, value in state_dict.items()}


  def load_pretrain(self, model, pretrained_path):
      ## bug fix
      ## see https://github.com/CSAILVision/places365/issues/25#issuecomment-333871990
      pickle.load = partial(pickle.load, encoding="latin1")
      pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
      #model = torch.load(model_file, map_location=lambda storage, loc: storage, pickle_module=pickle)

      #pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(self.device), pickle_module=pickle)
      if self.use_gpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(), pickle_module=pickle)
      else:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cpu(), pickle_module=pickle)

      if "state_dict" in pretrained_dict.keys():
        pretrained_dict = self.remove_prefix(pretrained_dict['state_dict'], 'module.')
      else:
        pretrained_dict = self.remove_prefix(pretrained_dict, 'module.')
      self.check_keys(model, pretrained_dict)
      model.load_state_dict(pretrained_dict, strict=False)
      return model


