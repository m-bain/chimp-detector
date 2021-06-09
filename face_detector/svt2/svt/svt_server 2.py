##
## Seebibyte Visual Tracker (SVT) Server
## Accepts tracking jobs and responds with tracking results over HTTP
##
## Author:  Abhishek Dutta <adutta@robots.ox.ac.uk>
## Date:    15 Nov. 2018
##


from svt.svt_task_manager import svt_task_manager

from http.server import HTTPServer
from http.server import BaseHTTPRequestHandler
import signal
import sys
import json
import os

class svt_http_request_handler(BaseHTTPRequestHandler):
  def do_GET(self):
    global task_manager
    #print( 'GET %s' % (self.path) )

    if self.path.startswith('/svt/status'):
      args = self.path.split('?')
      if len(args) == 2:
        param = args[1].split('=')
        if len(param) == 2 and param[0] == 'task_id':
          task_id = param[1]
          task_status_obj = task_manager.get_task_status(task_id);
          response = json.dumps(task_status_obj)
          self.send_response(200)
          self.send_header('Content-Length', str( len(response) ))
          self.send_header('Content-Type', 'application/json')
          self.send_header('Access-Control-Allow-Origin', '*')
          self.end_headers()
          self.wfile.write( response.encode('utf-8') )
          return

    if self.path.startswith('/svt/fetch'):
      args = self.path.split('?')
      if len(args) == 2:
        param = args[1].split('=')
        if len(param) == 2 and param[0] == 'task_id':
          task_id = param[1]
          response_obj = task_manager.get_all_task_result(task_id);
          response = json.dumps(response_obj)
          self.send_response(200)
          self.send_header('Content-Length', str( len(response) ))
          self.send_header('Content-Type', 'application/json')
          self.send_header('Access-Control-Allow-Origin', '*')
          self.end_headers()
          self.wfile.write( response.encode('utf-8') )
          return

    ## if nothing matches, send HTTP 400 error response
    self.send_response(400)
    self.send_header('Access-Control-Allow-Origin', '*')
    self.end_headers()
    self.wfile.write(b'Invalid request')

  def do_POST(self):
    global task_manager
    #print( 'POST %s' % (self.path) )

    if self.path == '/svt/track':
      task_definition = self.rfile.read( int(self.headers['Content-Length']) );
      task_id = task_manager.create_task( task_definition )
      response = '{"task_id":"%s"}' % (task_id)

      self.send_response(200)
      self.send_header('Content-Length', str( len(response) ))
      self.send_header('Content-Type', 'application/json')
      self.send_header('Access-Control-Allow-Origin', '*')
      self.end_headers()
      self.wfile.write( response.encode('utf-8') )
      return

    if self.path.startswith('/svt/stop'):
      args = self.path.split('?')
      if len(args) == 2:
        param = args[1].split('=')
        if len(param) == 2 and param[0] == 'task_id':
          task_id = param[1]
          ok = task_manager.stop_task(task_id)
          response = '{"status":"ok"}'
          if not ok:
            response = '{"status":"error"}'
          self.send_response(200)
          self.send_header('Content-Length', str( len(response) ))
          self.send_header('Content-Type', 'application/json')
          self.send_header('Access-Control-Allow-Origin', '*')
          self.end_headers()
          self.wfile.write( response.encode('utf-8') )
          return

    if self.path.startswith('/svt/fetch'):
      args = self.path.split('?')
      if len(args) == 2:
        param = args[1].split('=')
        if len(param) == 2 and param[0] == 'task_id':
          task_id = param[1]
          region_list_str = self.rfile.read( int(self.headers['Content-Length']) );
          region_list = json.loads(region_list_str)
          response_obj = task_manager.get_task_result(task_id, region_list);
          response = json.dumps(response_obj)
          self.send_response(200)
          self.send_header('Content-Length', str( len(response) ))
          self.send_header('Content-Type', 'application/json')
          self.send_header('Access-Control-Allow-Origin', '*')
          self.end_headers()
          self.wfile.write( response.encode('utf-8') )
          return

    ## if nothing matches, send HTTP 400 error response
    self.send_response(400)
    self.send_header('Access-Control-Allow-Origin', '*')
    self.end_headers()
    self.wfile.write(b'Invalid request')

def run_svt_server(task_manager_instance, SVT_SERVER_HOSTNAME, SVT_SERVER_PORT):
  global task_manager
  task_manager = task_manager_instance
  try:
    ## Start the http server which manages requests from SVT 
    ## user interface accessible using a web browser
    with HTTPServer((SVT_SERVER_HOSTNAME, SVT_SERVER_PORT), svt_http_request_handler) as httpd:
      httpd.serve_forever()
  except:
    if task_manager:
      task_manager.stop()
  #raise ValueError('Exception occured!')
  else:
    if task_manager:
      task_manager.stop()
  print('Exiting, please wait ... (press Ctrl + c again if the program does not terminate)')

