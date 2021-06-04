import pandas as pd
import numpy as np
import cv2
import os
import glob
import sys
import subprocess
from matplotlib import cm


def smooth_track(track_data):
    res = []
    for track_id in track_data['track_id'].unique():
        tid = track_data[track_data['track_id'] == track_id]
        if len(tid) > 12:
            tid['x'] = smooth_(tid['x'])
            tid['y'] = smooth_(tid['y'])
            tid['width'] = smooth_(tid['width'])
            tid['height'] = smooth_(tid['height'])
            res.append(tid)

    if len(res) > 0:
        return pd.concat(res)
    else:
        return []


def smooth_(x, window_len=12, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    # x = numpy.pad(x, int(window_len/2), 'reflect')
    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))

    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[(window_len // 2 - 1):-(window_len // 2)]

def get_frame_rate(filename):
    if not os.path.exists(filename):
        sys.stderr.write("ERROR: filename %r was not found!" % (filename,))
        return -1
    out = subprocess.check_output(
        ["ffprobe", filename, "-v", "0", "-select_streams", "v", "-print_format", "flat", "-show_entries",
         "stream=r_frame_rate"])
    rate = str(out).split('=')[1].strip()[1:-4].split('/')
    if len(rate) == 1:
        return float(rate[0])
    if len(rate) == 2:
        return float(rate[0]) / float(rate[1])
    return -1

def frames2vid(frames_dir, video_fp, suffix=''):
    fps = get_frame_rate(video_fp)
    exp_video = os.path.join(frames_dir, 'out.mp4')
    exp_audio_video = os.path.join(frames_dir, video_fp.replace('.mp4', f'{suffix}.mp4'))
    cmd = 'ffmpeg -y -r {} -start_number 1 -i {}/%06d.jpg -c:v libx264 -vf fps={} -pix_fmt yuv420p {}'.format(fps,
                                                                                                              frames_dir,
                                                                                                              fps,
                                                                                                              exp_video)
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    process.communicate()
    audio_cmd = 'ffmpeg -y -i %s -vn -ab 256 tmp.mp3' % video_fp
    process = subprocess.Popen(audio_cmd.split(), stdout=subprocess.PIPE)
    process.communicate()
    replace_audio = 'ffmpeg -y -i %s -i %s -c:v copy -map 0:v:0 -map 1:a:0 %s' % (exp_video, 'tmp.mp3', exp_audio_video)
    process = subprocess.Popen(replace_audio.split(), stdout=subprocess.PIPE)
    process.communicate()
    os.remove('tmp.mp3')

def write_tracks(track_data, frame_dir, width, height, conf=False, action_data=None):
    acc_replace = {'crack_nut': 'NUT CRACK', 'eating': 'EAT', 'Drumming': 'DRUM'}

    prev_id = -1
    colormap = [cm.Pastel1.__dict__['colors'][idx] for idx in range(9)]
    for idx, row in track_data.iterrows():
        curr_id = '%06d.jpg' % (row['frame_id'])
        if prev_id != curr_id:
            img = cv2.imread(os.path.join(frame_dir, curr_id))
            try:
                acc_height, acc_width = img.shape[:2]
            except:
                import pdb; pdb.set_trace()
            # height_s = acc_height / height
            # width_s = acc_width / width
            height_s = height
            width_s = width
            prev_id = curr_id
        track_id = row['track_id'] + 1
        if track_id != -1:
            if conf:
                color = colormap[int(track_id * 8)]
            else:
                color = colormap[track_id % 9]
            color = [c * 255 for c in color]
            im_x, im_y, im_w, im_h = int(row['x']), int(row['y']), int(
                row['width']), int(row['height'])
            cv2.rectangle(img, (im_x, im_y), (im_x + im_w, im_y + im_h), color, 2)

            if action_data is None:
                text_width, text_height = \
                cv2.getTextSize(str(track_id), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=1)[0]
                tbox_coords = ((im_x + im_w - text_width + 4, im_y), (im_x + im_w, im_y - text_height - 4))
                cv2.rectangle(img, tbox_coords[0], tbox_coords[1], color, cv2.FILLED)
                cv2.putText(img, str(track_id), (im_x + im_w - text_width + 4, im_y), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 0), lineType=cv2.LINE_AA)
            if action_data is not None:
                adf = action_data[action_data['f1'] <= row['frame_id']]
                adf = adf[adf['f2'] >= row['frame_id']]
                adf = adf[adf['indiv_id'] == row['indiv_id']]
                if len(adf) > 0:
                    action_text = acc_replace[adf['action'].iloc[0]]
                    text_width, text_height = \
                        cv2.getTextSize(action_text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=1)[0]
                    tbox_coords = ((im_x, im_y - 2), (im_x + text_width + 4, im_y - text_height - 4))
                    cv2.rectangle(img, tbox_coords[0], tbox_coords[1], (255, 255, 255), cv2.FILLED)
                    cv2.putText(img, action_text, (im_x + 2, im_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0),
                                lineType=cv2.LINE_AA)
                    cv2.imwrite(os.path.join(frame_dir, curr_id), img)
            cv2.imwrite(os.path.join(frame_dir, curr_id), img)



def create_det_vid(csv_fp, frame_video_dir, video_fp, smooth=True):
    data = pd.read_csv(csv_fp)
    data.sort_values('frame_id', inplace=True)
    if smooth:
        data = smooth_track(data)
    imgs_dir = os.path.join(frame_video_dir, '*.jpg')
    img_ex = cv2.imread(glob.glob(imgs_dir)[0])
    height, width = img_ex.shape[:2]
    write_tracks(data, frame_video_dir, width, height)
    frames2vid(frame_video_dir, video_fp)
    #for file in glob.glob(os.path.join(frame_video_dir, '*.jpg')):
    #    os.remove(file)

if __name__ == "__main__":
    create(
        "results/tmp/19_mini.mp4.full.csv",
        "tmp/19_mini.mp4",
        "19_mini.mp4"
    )