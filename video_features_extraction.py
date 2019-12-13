
# coding: utf-8

# # Video Features Extraction
# ---

# In[1]:


from moviepy.editor import VideoFileClip
import numpy as np
import tensorflow as tf
import argparse
import os

## Kinects-i3D ##
import sys
sys.path.insert(1, '../kinetics-i3d/') #insert the kinects-i3d project to the path
import i3d


# ### Global Variables

# In[3]:


_NUM_CLASSES = 400
_IMAGE_SIZE = [224,224]
_FRAMES = 64


# ### Creating frames

# In[8]:


def get_frames_array(video_path):
    clip = VideoFileClip(video_path, target_resolution=(_IMAGE_SIZE[0],_IMAGE_SIZE[1]))
    frames = np.array([x for x in clip.iter_frames()])
    return pad_frames(frames)
    
def pad_frames(frames):
    frames_qtt = frames.shape[0]
    if(frames_qtt < _FRAMES): #padding the frame

        pad_left_count = int((_FRAMES - frames_qtt) / 2)
        pad_right_count = _FRAMES - frames_qtt - pad_left_count

        pad_left = np.zeros((pad_left_count, frames.shape[1],  frames.shape[2],  frames.shape[3]))
        pad_right = np.zeros((pad_right_count, frames.shape[1],  frames.shape[2],  frames.shape[3]))

        rgb_array = np.concatenate((pad_left, frames, pad_right))

#         print('Array padded')

    else: 
        ##TODO: reduce the array -- CHECK IT!
        rgb_array = np.resize(frames.mean(axis=0).astype(int),
                              (_FRAMES, frames.shape[1],  frames.shape[2],  frames.shape[3]))
#         print('Array resized')

    return rgb_array


def get_video_names(path, sep):
    files = os.listdir(path)
    files = [f for f in files if f[-4:] == '.mp4']
    return [f[:f.rfind(sep)] for f in files]


def get_utterances_array(utterances_path, video_name, sep = '_', start=1):
    print("Starting get the utterances...")
    result_array = np.empty((0, _FRAMES, _IMAGE_SIZE[0],_IMAGE_SIZE[1], 3))
    count = start 
    while(True):
        try:
            f = get_frames_array(utterances_path + video_name + sep + str(count) + ".mp4")
            result_array = np.append(result_array, [f], axis=0)
            count +=1
            if(count % 5 == 0):
                print("Utterance", count)

        except Exception as e:
            print("{0} utterances processed".format(count-start))
    #         print(e)
            break
    return result_array


def get_videos_array(path, sep='_', start =1 ):
    video_names = get_video_names(path,sep)

    result_array = np.empty((0, _NUM_CLASSES))

    for v in video_names:
        utterances = get_utterances_array(path,v,sep, start)
        video_feature = model_visual_features(utterances)
        result_array = np.append(result_array, [video_feature], axis=0)



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


def get_video_features(path):
    return model_visual_features(get_utterances_array(path))


path = "../MOSI_Dataset/Segmented/_dI--eQ6qVU_"
get_video_features(path)

# if __name__ == "__main__":
#     argv = sys.argv[1:]
#     parser = argparse.ArgumentParser()
#     print(parser)





