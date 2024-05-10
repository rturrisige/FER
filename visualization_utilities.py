from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
from scipy.ndimage.filters import gaussian_filter
import plotly.graph_objects as go
import plotly
import numpy as np
import matplotlib.pyplot as plt


blendshape_names = ['neutral', 'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight',
                    'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight', 'eyeBlinkLeft', 'eyeBlinkRight', 'eyeLookDownLeft',
                    'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight', 'eyeLookUpLeft',
                    'eyeLookUpRight', 'eyeSquintLeft', 'eyeSquintRight', 'eyeWideLeft', 'eyeWideRight', 'jawForward', 'jawLeft',
                    'jawOpen', 'jawRight', 'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft', 'mouthFrownRight',
                    'mouthFunnel', 'mouthLeft', 'mouthLowerDownLeft', 'mouthLowerDownRight', 'mouthPressLeft', 'mouthPressRight',
                    'mouthPucker', 'mouthRight', 'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper',
                    'mouthSmileLeft', 'mouthSmileRight', 'mouthStretchLeft', 'mouthStretchRight', 'mouthUpperUpLeft',
                    'mouthUpperUpRight', 'noseSneerLeft', 'noseSneerRight']


blendshape_group_names = ['neutral'] + ['brown' + str(i) for i in range(1, 6)] +\
                         ['cheek' + str(i) for i in range(1, 4)] + ['eye'+ str(i) for i in range(1, 15)] + \
                         ['jaw' + str(i) for i in range(1, 5)] + ['mouth' + str(i) for i in range(1, 24)] + \
                         ['nose' + str(i) for i in range(1, 3)]


def blur_image(image):
    """
    It blurs an image
    :param image: numpy array, image
    :return: numpy array, image blurred by applying a gaussian filter
    """
    return gaussian_filter(image, sigma=(7, 7, 0))


def draw_landmarks_on_image(image, detection_result, blur=True):
    """
    It takes the image and the face landmarkers and overlaps them in an annotated image.
    :param image: video frame
    :param detection_result: landmarkers
    :param blur: bool, if True gaussian filter is applied to the image
    :return: annotated image: image with face landmarkers
    """
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(image)
    if blur:
        annotated_image = blur_image(annotated_image)
    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
          landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_tesselation_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
              landmark_drawing_spec=None,
              connection_drawing_spec=mp.solutions.drawing_styles
              .get_default_face_mesh_iris_connections_style())

    return annotated_image


def plot_face_blendshapes_bar_graph(saver_fold, face_blendshapes):
    """
    It generates a bar plot of the blendshape scores.
    :param saver_fold: directory where to save the praph
    :param face_blendshapes: blandshape scores
    """
    # Extract the face blendshapes category names and scores.
    face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
    face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
    # The blendshapes are ordered in decreasing score value.
    face_blendshapes_ranks = range(len(face_blendshapes_names))

    fig, ax = plt.subplots(figsize=(12, 12))
    bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
    ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
    ax.invert_yaxis()

    # Label each bar with values
    for score, patch in zip(face_blendshapes_scores, bar.patches):
        plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

    ax.set_xlabel('Score')
    ax.set_title("Face Blendshapes")
    plt.tight_layout()
    plt.savefig(saver_fold + '/blendshape_bar.pdf')
    plt.close()


def plot_marker_sequence(saver_fold, title, feature_names, scores, index, fsize=(10, 15)):
    """
    It generates a plot for the selected features.
    :param saver_fold: str, directory where to save the plot
    :param title: str, plot title
    :param feature_names: list, name of the feautres
    :param scores: list, scores
    :param index: tuple or list, initial and final index of scores
    :param fsize: tuple, figure size
    """
    plt.figure(figsize=fsize)
    plt.suptitle(title)
    j = 1
    for i in range(index[0], index[1]):
        plt.subplot(index[1]-index[0], 1, j)
        plt.plot(scores[:, i])
        plt.ylabel(feature_names[i])
        j += 1
    plt.tight_layout()
    plt.savefig(saver_fold + '/' + title + '.pdf')
    plt.close()


def plot_processing_check(saver_fold, check_video):
    """
    It generates a plot about video processing.
    :param saver_fold: str, directory where to save the plot
    :param check_video: numpy array or list, vector containing information about video processing
    :return:
    """
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.plot(check_video)
    plt.yticks([0, 1, 2], ['corrupted', 'no landmarks', 'landmarks'])
    plt.xlabel('Frames')
    plt.subplot(122)
    plt.hist(check_video, bins=3, edgecolor="black")
    plt.xticks([0.35, 1, 1.65], ['corrupted', 'no landmarks', 'landmarks'])
    plt.savefig(saver_fold + '/check_video_processing.pdf')
    plt.close()


def radar_chart(blendshape_scores):
    """
    It generates a radar chart of the blendhsape scores by using plotly.
    :param blendshape_scores: numpy array, blendhshape scores
    """
    blendshape_average = np.mean(blendshape_scores, 0)
    max = np.max(blendshape_average)
    fig = go.Figure(go.Scatterpolar(r=blendshape_average, theta=blendshape_names))
    fig.update_layout(width=600, height=600, polar=dict(
        radialaxis=dict(visible=True,range=[0, max])
    ),
    showlegend=False)
    plotly.offline.plot(fig, filename='./averaged_blendshape_scores.html')



def create_radar_chart(blendshape, blendshape_labels):
    """
    It generates a radar chart of the blendhsape scores by using matplotlib.
    :param blendshape: numpy array, blendhshape scores
    :param blendshape_labels: list, names of the blendshape features
    :return image, numpy array
    """
    # Number of variables (blendshapes)
    num_vars = len(blendshape_labels)
    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    # The plot is a circle, so we need to "complete the loop"
    blendshape = np.concatenate((blendshape, [blendshape[0]]))
    angles += angles[:1]
    # Initialize the radar chart plot
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    # Draw one axe per variable and add labels
    plt.xticks(angles[:-1], blendshape_labels, color='grey', size=10)
    # Plot data
    ax.plot(angles, blendshape, linewidth=1, linestyle='solid')
    # Fill area
    ax.fill(angles, blendshape, 'b', alpha=0.1)
    # Save plot as an RGB image
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image


def plot_blendshapes_groups(saver_fold, blendshape_scores, blendshape_names=blendshape_names):
    """
    It takes the blendshape scores and generates a plot for each blendshape group (e.g., "nose", "cheek").
    :param saver_fold: str, directory where to save the plots
    :param blendshape_scores: numpy array, blendhsape scores
    :param blendshape_names: list, names of the beldnshape features
    """
    # neutral
    plt.title('neutral')
    plt.plot(blendshape_scores[:, 0])
    plt.savefig(saver_fold + '/neutral.pdf')
    plt.close()
    # brow
    plot_marker_sequence(saver_fold, 'brow', blendshape_names, blendshape_scores, [1, 6], fsize=(10, 15))
    # cheek
    plot_marker_sequence(saver_fold, 'cheek', blendshape_names, blendshape_scores, [6, 10], fsize=(10, 15))
    # eye
    plot_marker_sequence(saver_fold, 'eye', blendshape_names, blendshape_scores, [10, 24], fsize=(10, 20))
    # jaw
    plot_marker_sequence(saver_fold, 'jaw', blendshape_names, blendshape_scores, [24, 28], fsize=(10, 15))
    # mouth
    plot_marker_sequence(saver_fold, 'mouth', blendshape_names, blendshape_scores, [28, 50], fsize=(10, 30))
    # nose
    plot_marker_sequence(saver_fold, 'nose', blendshape_names, blendshape_scores, [50, 52], fsize=(10, 5))





