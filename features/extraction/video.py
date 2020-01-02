
# coding: utf-8

# # Video Features Extraction
# ---

from moviepy.editor import VideoFileClip
import numpy as np
import tensorflow as tf
import argparse
import os
from collections import Counter

## Kinects-i3D ##
import sys
sys.path.insert(1, '../kinetics-i3d/') #insert the kinects-i3d project to the path
import i3d

# Global Variables
_NUM_CLASSES = 400
_IMAGE_SIZE = [224,224]
_FRAMES = 64


#return an array of the frames values from a utterance
def get_frames_array(video_path):
    clip = VideoFileClip(video_path, target_resolution=(_IMAGE_SIZE[0],_IMAGE_SIZE[1]))
    frames = np.array([x for x in clip.iter_frames()])
    return pad_array(frames, _FRAMES)


def pad_array(array, size):
    array_qtt = array.shape[0]

    if(array_qtt == size):
        return array

    if(array_qtt < size): #pad with 0

        pad_left_count = int((size - array_qtt) / 2)
        pad_right_count = size - array_qtt - pad_left_count

        pad_left = np.zeros((pad_left_count,) + array.shape[1:])
        pad_right = np.zeros((pad_right_count,) + array.shape[1:])

        result_array = np.concatenate((pad_left, array, pad_right))

    else: #resize
        result_array = np.resize(array.mean(axis=0).astype(int),
                                 (size,) + array.shape[1:])
    return result_array



#get the video informations in a path with segmented videos (by utterance)
def get_video_info(path, sep):
    files = os.listdir(path)
    files = [f for f in files if f[-4:] == '.mp4']
    files = [f[:f.rfind(sep)] for f in files]

    names = list(set(files))
    max_segment = Counter(files).most_common(1)[0][1]

    return names, max_segment

# get a numpy array of the all the video utterances
def get_video_utterances_array(videos_path, video_name,
                               max_utterance, sep='_', start=1):

    print("Geting utterances from the video", video_name)
    result_array = np.empty((0, _FRAMES, _IMAGE_SIZE[0],_IMAGE_SIZE[1], 3))
    count = start
    while(True):
        try:
            f = get_frames_array(videos_path + video_name + sep + str(count) + ".mp4")
            result_array = np.append(result_array, [f], axis=0)
            count +=1
            if(count % 5 == 0):
                print("Utterance", count)

        except Exception as e:
            print("{0} utterances processed".format(count-start))
    #         print(e)
            break

    # pad the result
    result_array = pad_array(result_array, max_utterance)

    return result_array


# get the numpy array of the video
# with padded frames and padded utterances
def get_videos_array(path, sep='_', start =1, filenames=None):
    video_names, max_utterance = get_video_info(path,sep)

    print("Video names: ", video_names)

    result_array = np.empty((0, max_utterance, _FRAMES, _IMAGE_SIZE[0], _IMAGE_SIZE[1], 3))

    for v in video_names:
        if(not filenames or v in filenames):
            video_array = get_video_utterances_array(path,v,max_utterance,sep, start)

            result_array = np.append(result_array, [video_array], axis=0)

    return result_array


#return the video features using a CNN model
def model_visual_features(rgb_array):
    print("Starting modeling...")

    i3d_model = i3d.InceptionI3d(num_classes=_NUM_CLASSES, final_endpoint='Predictions')

    inp = tf.placeholder(tf.float32, [None, _FRAMES, _IMAGE_SIZE[0], _IMAGE_SIZE[1], 3])

    predictions, end_points = i3d_model(inp, is_training=True, dropout_keep_prob=0.5)

    init_op = tf.global_variables_initializer()

    # sample_input = np.zeros((5, 64, _IMAGE_SIZE[0], _IMAGE_SIZE[1], 3))
    sample_input = rgb_array

    with tf.Session() as sess:
        sess.run(init_op)
        out_predictions, out_logits = sess.run([predictions, end_points['Logits']], {inp: sample_input})

    return out_logits


def get_visual_features_from_array(video_array):
    max_utterance = video_array.shape[1]
    result_array = np.empty((0, max_utterance, _NUM_CLASSES))

    count = 0
    for v in video_array:
        visual_features = model_visual_features(v)
        np.save("v_features_{}".format(count), visual_features)
        result_array = np.append(result_array, [visual_features], axis=0)

    return result_array


def get_video_features(path, sep='_', start=1, filenames=None):
    return get_visual_features_from_array(get_videos_array(path, sep, start, filenames))
