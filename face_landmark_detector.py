from cv2 import VideoCapture
import os
from visualization_utilities import plot_blendshapes_groups, plot_processing_check
from video_processing_utilities import get_video_information, create_annotated_video, extract_landmarks
import argparse
from glob import glob as gg
import sys


def processing(file, saver_fold, logfile=False, annotated_video=None, make_plots=False):
    """
    It processes "camera separata" video and extracts face features, including landmarks and blendshape features.
    It saves video information in the logfile and generates an annotated video.
    :param file: str, path/to/video
    :param saver_fold: str, directory where to save results
    :param logfile: bool, if True a txt file containing video information is created
    :param annotated_video:  str, "False", "online" or "post" corresponding to no annotated video generation,
    online generation and generation after processing, respectively.
    """
    name = os.path.basename(file).split('.')[0]
    saver_fold += '/' + name
    if not os.path.exists(saver_fold):
        os.makedirs(saver_fold)
    if logfile:
        logfile = open(saver_fold + '/Video_information.txt', 'w')

    cap = VideoCapture(file)
    if not cap.isOpened():
        print("Error opening video file")
    else:
        frame_width, frame_height, fps, n_frame = get_video_information(cap, logfile=logfile)

        check_video, blendshape_scores = extract_landmarks(saver_fold, cap, annotated_video=annotated_video)

        # Visualizing results
        if make_plots:
            plot_processing_check(saver_fold, check_video)
            plot_blendshapes_groups(saver_fold, blendshape_scores)
            print('Plots saved.\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Extract video labelled as "camera_separata" from <data_dir>.
    Video information are saved in Video_information.txt if <logfile> is True.
    Landmarks and blendshape scores are extracted and saved in <saver_dir>. A video with face landmark grid is generated
    and saved in <saver_dir> if <annotated_video> is 'online' or 'post'. """)
    parser.add_argument('--data_dir', required=True, type=str,
                        help='The file, including the path, of the video to be processed')
    parser.add_argument('--saver_dir', default=os.getcwd() + '/Results/Video_processing', type=str,
                        help='The directory where to save the processed video')
    parser.add_argument('--logfile', default=True, type=bool,
                        help='If True, a logfile with video information is created')
    parser.add_argument('--annotated_video', default=None, type=str,
                        help='If not None, the annotated video is generated during the processing.')
    parser.add_argument('--make_plots', default=False, type=bool,
                        help='If True, plots with the extracted features are generated and saved.')
    args = parser.parse_args()
    processing(args.data_dir, args.saver_dir, args.logfile, args.annotated_video)
