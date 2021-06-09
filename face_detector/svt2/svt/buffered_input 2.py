## Fill a buffer with frames from input video or set of images inside a folder
## so that access to the input frames is fast
##
## Author:  Abhishek Dutta <adutta@robots.ox.ac.uk>
## Date:    22 Oct. 2018

import threading
import queue
import subprocess as sp # for pipe
import numpy as np
#from PIL import Image   # to load image
import cv2
import os               # for file list

FFPROBE_BIN = 'ffprobe'
FFMPEG_BIN  = 'ffmpeg'

class buffered_input(threading.Thread):

  def __init__(self, input_type, input_location, start_time=0.0, end_time=None, buffer_size=10):
    threading.Thread.__init__(self)

    if input_type not in [ 'video_file', 'folder_images' ]:
      raise ValueError('input_type must be {video_file, folder_images}')

    self.state = {}
    self.state['input_type'] = input_type
    self.state['input_location'] = input_location
    self.state['start_time']  = float(start_time)
    self.state['end_time']  = end_time
    self.state['current_time']  = float(start_time)
    self.state['all_frames_extracted'] = False
    self.frame_buffer = queue.Queue( maxsize=buffer_size )
    self.stop_flag = False

  def stop_buffered_input(self):
    print('Stopping buffered input ...')
    self.stop_flag = True

  def frame_available(self):
    if self.state['all_frames_extracted'] and self.frame_buffer.empty():
      return False
    else:
      return True

  # returns frame, frame_offset
  def get_next_frame(self):
    queue_entry = self.frame_buffer.get(block=True, timeout=None);  # unblocks the queue if it was full
    self.frame_buffer.task_done()
    return queue_entry['frame_data'], queue_entry['frame_offset'], queue_entry['frame_name']

  def _load_image(self, fn):
    ## needed to maintain consistency with BGR data read using cv2.imread()
    #img = Image.open(fn).convert('RGB')
    #im = np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3)
    im = cv2.imread(fn)
    return im

  def _add_frame_to_queue(self, frame_data, frame_offset, frame_name):
    self.frame_buffer.put( { 'frame_data':frame_data, 'frame_offset':self.state['current_frame'], 'frame_name':frame_name }, block=True, timeout=None) # blocks if the queue is full

  def run(self):
    if self.state['input_type'] == 'video_file':
      ## read information (frame size, frame rate, etc) about input video
      ## see https://trac.ffmpeg.org/wiki/FFprobeTips
      ## ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,duration,nb_frames -of csv=s=,:p=0 -i 
      info_cmd = [ FFPROBE_BIN, 
                  '-v','fatal',
                  '-select_streams', 'v:0',
                  '-show_entries', 'stream=width,height,r_frame_rate,duration,nb_frames',
                  '-of', 'csv=s=,:p=0', 
                  '-i', self.state['input_location'] ]
      info_pipe = sp.Popen(info_cmd, stdout=sp.PIPE, stderr=sp.PIPE)
      input_info_str = info_pipe.stdout.read().decode('utf8')
      info_pipe.terminate()

      input_info_list = input_info_str.split('\n')[0].split(',') # avoid trailing \n

      self.state['image_width']     = int( input_info_list[0] )
      self.state['image_height']    = int( input_info_list[1] )

      self.state['input_duration']  = float(input_info_list[3])
      if self.state['end_time'] is None:
        self.state['end_time'] = self.state['input_duration']
      else:
        self.state['end_time'] = float(self.state['end_time'])

      self.state['input_frame_count']   = int(input_info_list[4])
      self.state['input_frame_rate']    = round( self.state['input_frame_count'] / self.state['input_duration'] )
      self.state['frame_time_interval'] = float( self.state['input_duration']/ self.state['input_frame_count'] )
      self.state['start_frame'] = int( self.state['start_time'] / self.state['frame_time_interval'] )

      ## read video frame to extract template
      ## see http://zulko.github.io/blog/2013/09/27/read-and-write-video-frames-in-python-using-ffmpeg/
      ffmpeg_read_cmd = [ FFMPEG_BIN,
                      '-ss', str( self.state['start_time'] ),
                      '-i', self.state['input_location'],
                      '-loglevel', 'panic',
                      '-f', 'image2pipe',
#                      '-pix_fmt', 'rgb24',
                      '-pix_fmt', 'bgr24',
                      '-vcodec', 'rawvideo', '-' ]
      ffmpeg_read_pipe = sp.Popen(ffmpeg_read_cmd, stdout = sp.PIPE, bufsize=10**8)

      self.state['current_frame'] = self.state['start_frame']
      self.state['end_frame'] = -1
      while self.state['current_frame'] != self.state['end_frame'] and not self.stop_flag:
        frame_str = ffmpeg_read_pipe.stdout.read( self.state['image_width'] * self.state['image_height'] * 3 )
        if len(frame_str) != 0:
          frame_data = np.frombuffer(frame_str, dtype='uint8')
          frame_data = frame_data.reshape( self.state['image_height'], self.state['image_width'], 3 )
          ## convert from RGB to BGR ( to match cv2.imread() )
          #print('\nWARNING: reading frame in RGB format. Convert to BGR to match cv2.imread() ********')
          self._add_frame_to_queue( frame_data, self.state['current_frame'], self.state['current_frame'] )
          self.state['current_frame'] = self.state['current_frame'] + 1
        else:
          self.state['current_frame'] = self.state['current_frame'] - 1
          self.state['end_frame'] = self.state['current_frame']
          self.state['all_frames_extracted'] = True
          self.stop_flag = True
          break;

    if self.state['input_type'] == 'folder_images':
      input_file_list = os.listdir( self.state['input_location'] )
      input_file_list.sort()

      if self.state['start_time']:
        self.state['start_frame'] = int( self.state['start_time'] )
      else:
        self.state['start_frame'] = 0
      self.state['current_frame'] = self.state['start_frame']

      if self.state['end_time']:
        self.state['end_frame'] = int( self.state['end_time'] )
      else:
        self.state['end_frame'] = len(input_file_list)

      for self.state['current_frame'] in range( self.state['start_frame'], self.state['end_frame'] ):
        frame_fn   = input_file_list[ self.state['current_frame'] ]
        frame_path = os.path.join( self.state['input_location'], frame_fn )
        frame_data = self._load_image(frame_path)
        self._add_frame_to_queue( frame_data, self.state['current_frame'], frame_fn )
        if self.stop_flag:
          break;

      self.state['current_frame'] = self.state['current_frame'] - 1
      self.state['end_frame'] = self.state['current_frame']
      self.state['all_frames_extracted'] = True

## unit test
if __name__ == '__main__':
  inp = buffered_input( input_type='video_file', input_location='/data/svt/videos/svt_demo_fish_25fps.mp4', start_time=0.788, end_time=-1.0, buffer_size=10 )
  inp.start();

  # get first frame (block until a frame is available)
  frame_data, frame_offset = inp.get_next_frame();

  # get all remaining frames
  while inp.state['current_frame'] != inp.state['end_frame']:
    frame_data, frame_offset = inp.get_next_frame();
    print('\nframe %d' %(frame_offset))

  print('\nDone\n')

