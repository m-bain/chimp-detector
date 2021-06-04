import os
import argparse

MODEL_PATHS = {
    'body': 'ssd300_BFbootstrapBissau4p5k_prebossou_best.pth',
    'face': 'ssd300_CFbootstrap_85000.pth'
}

def main(args):
    print('Processing video:	', args.input_video)
    log_fp = os.path.join('log', args.input_video.replace('.mp4', '.log'))
    results_zip = os.path.join('results', args.input_video.replace('.mp4', '.zip'))
    if not os.path.exists(os.path.dirname(results_zip)):
        os.makedirs(os.path.dirname(results_zip))
    if not os.path.exists(os.path.dirname(log_fp)):
        os.makedirs(os.path.dirname(log_fp))

    detector_pth = MODEL_PATHS[args.detect]

    cmd = 'bash start_pipeline.body_bissau.sh "{}" "{}" "{}" "{}" "{}"'.format(args.input_video, log_fp, results_zip,
                                                                             f'tmp/{args.detect}', detector_pth)
    print(cmd)
    os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_video', required=True, type=str)
    parser.add_argument('--conf_thresh', default=0.37, type=float)
    parser.add_argument('--detect', default='body', choices=['body', 'face'])

    args = parser.parse_args()
    main(args)