from cv2 import CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FPS, CAP_PROP_FRAME_COUNT
import numpy as np
import cv2
import mediapipe as mp
from visualization_utilities import blur_image, draw_landmarks_on_image
from alive_progress import alive_bar
from torch.utils.data import Dataset
import torch


def get_video_information(video, logfile=None, visualization=True):
    """
    It extracts video properties and save them in a logfile, if this is not None
    :param video:
    :param logfile: None or object, opened txt file
    :param visualization: bool, if True video information are printed
    :return:
    frame_width: int, width of the frames
    frame_height: int, height of the frames
    fps: float, frame rate
    n_frame: int, number of frames in the video

    """
    frame_width = int(video.get(CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(CAP_PROP_FRAME_HEIGHT))
    fps = video.get(CAP_PROP_FPS)
    n_frame = int(video.get(CAP_PROP_FRAME_COUNT))
    seconds = n_frame / fps
    if visualization:
        print("Video information")
        print("Frame count: ", n_frame)
        print("Image frame width: ", frame_width)
        print("Image frame height: ", frame_height)
        print("Frame rate: ", fps)
        print('Video duration (s):', seconds)
        print('Video duration (min):', seconds / 60)
        print('\n')
    if logfile:
        logfile.write('Frame count: {}\n'.format(n_frame))
        logfile.write('Image frame width: {}\n'.format(frame_width))
        logfile.write('Image frame height: {}\n'.format(frame_height))
        logfile.write('Frame rate: {}\n'.format(fps))
        logfile.write('Video duration (in seconds): {}\n'.format(seconds))
        logfile.write('Video duration (in minutes): {}\n'.format(seconds / 60))
        logfile.flush()
        logfile.close()
    return frame_width, frame_height, fps, n_frame


class loader(Dataset):
    def __init__(self, dataset):
        """
        dataset : path to data matrix
        """
        self.dataset = np.load(dataset, allow_pickle=True)
    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, index):
        x_batch = self.dataset[index]
        return x_batch


def create_annotated_video(data_dir, frame_width, frame_height, fps):
    """
    It load a list of annotated images and it generates a video.
    :param data_dir: directory containing processed.npy file
    :param frame_width: int, width of the frames
    :param frame_height: int, height of the frames
    :param fps: float, frame rate
    """
    print('Starting annotated video generation.')
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    out = cv2.VideoWriter(data_dir + '/camera_separata_with_landmarks.avi', fourcc, np.float64(fps),
                          (frame_width, frame_height), True)

    images = loader(data_dir + '/processed.npy')
    img_loader = torch.utils.data.DataLoader(images, batch_size=1, shuffle=False, num_workers=24)
    with alive_bar(len(img_loader), bar='classic', spinner='arrow') as bar:
        for _, data in enumerate(img_loader):
            out.write(data[0].numpy())
            bar()
    out.release()
    print('Complete.\n')


def extract_landmarks(data_dir, video, annotated_video=None):
    """
    It extracts face landmarker and blendshape scores from videos.
    It saves landmarkers, blendshape scores, timestamps, and checkvideo numpy files in <data_dir>.
    :param data_dir: str, data directory,
    :param video: object, opened video
    :param annotated_video: str, if not None the annotated video is generated and saved as 'annotated_video_with_landmarks.avi'
    """
    # Face Landmarker settings
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Create a face landmarker instance with the video mode:
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='face_landmarker.task'),
        running_mode=VisionRunningMode.VIDEO,
        output_face_blendshapes=True,
        num_faces=1,
        min_face_detection_confidence=0.01,
        min_tracking_confidence=0.01)

    blendshape_scores, processed, landmarks = np.empty([]), np.empty([]), np.empty([])
    check_video, timestamps = [], []

    if annotated_video:
        print('The annotated video will be simultaneously generated')
        frame_width, frame_height, fps, _ = get_video_information(video, visualization=False, logfile=False)
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        out = cv2.VideoWriter(data_dir + '/' + annotated_video +'with_landmarks.avi', fourcc, np.float64(fps),
                              (frame_width, frame_height), True)

    with FaceLandmarker.create_from_options(options) as landmarker:
        print('Starting Video Processing')
        with alive_bar(int(video.get(CAP_PROP_FRAME_COUNT)), bar='classic', spinner='arrow') as bar:
            while video.isOpened():
                success, image = video.read()
                if image is None:
                    break

                # if the image is not corrupted
                if success:
                    ts = int(video.get(cv2.CAP_PROP_POS_MSEC))
                    timestamps.append(ts)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
                    face_landmarker_result = landmarker.detect_for_video(mp_image, ts)

                    # if landmarks are captured
                    if face_landmarker_result.face_landmarks:
                        check_video.append(2)
                        # landmark
                        lndm = np.array(face_landmarker_result.face_landmarks[0])[None,]
                        landmarks = np.concatenate([landmarks, lndm], 0) if landmarks.shape else lndm
                        del lndm

                        # create landmarks grid
                        annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), face_landmarker_result, blur=True)

                        # Extract blendshapes
                        results = face_landmarker_result.face_blendshapes[0]
                        scores = np.array([results[i].score for i in range(len(results))])[None,]
                        blendshape_scores = np.concatenate([blendshape_scores, scores],
                                                           0) if blendshape_scores.shape else scores
                        del results

                    else:
                        check_video.append(1)
                        annotated_image = blur_image(image)

                    if annotated_video:
                        out.write(annotated_image)
                        del annotated_image
                else:
                    check_video.append(0)

                bar()

    # Saving variables
    # np.save(saver_fold + '/processed.npy', processed)
    np.save(data_dir + '/landmarks.npy', landmarks)
    np.save(data_dir + '/blendshape_scores.npy', blendshape_scores)
    np.save(data_dir + '/timestamps.npy', np.array(timestamps))
    np.save(data_dir + '/checkvideo.npy', np.array(check_video))
    del processed, landmarks, timestamps
    print('Processing ended.\n')
    return check_video, blendshape_scores

