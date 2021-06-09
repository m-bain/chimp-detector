# Seebibyte Visual Tracker (SVT)

## Source Folder Structure
  * `ui` : contains HTML/CSS/Javascript based user interface of SVT
  * `models` : contains the pytorch model definition for the tracker used by SVT
  * `pretrained_model_data` : placeholder for pretrained models
  * svt server:
    - `svt_server.py` : initializes the task manager and starts a HTTP server to handle requests made by SVT web browser based interface
    - `svt_http_request_handler.py` : defines the handler for GET and POST requests from SVT web based user interface
    - `buffered_input.py` : creates a FIFO queue to hold and serve the frames of a video used by `svt_task.py`
    - `svt_task_manager.py` : maintains a FIFO queue for tracking tasks submitted by through the SVT web user interface
    - `svt_task.py` : performs the core tracking task and is initialized and invoked by the `svt_task_manager.py`

Abhishek Dutta  
7 Nov. 2018
