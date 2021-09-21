import pandas as pd
import numpy as np
import os
import csv
from natsort import natsorted
from bisect import bisect_left
from scipy.spatial.transform import Rotation as R
import cv2
import math
import skimage.io

# TODO:
# - Filter out irrelevant frames (when tool is not touching tissue) -- maybe drop the first 5 seconds, then drop frames where force measurement is 0
# - Adjust for the fact that recording start times of tf, image and force are not the same (for now we're assuming that 0.0 happens at the same time for all 3 data streams)


def take_closest(myList, myNumber, lo, hi):
    '''
    Assumes myList is sorted. Returns the index of the closest value to myNumber.
    If two numbers are equally close, return the smallest index.
    lo, hi are the bounds of the list to search in.
    '''
    pos = bisect_left(myList, myNumber, lo=lo, hi=hi)
    if pos == 0:
        return pos
    if pos == len(myList):
        return pos
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
       return pos # after
    else:
       return pos - 1 # before


def vector_decomp(quaternion, force_value):
    '''
    Converts a force vector in tag frame to camera frame (preserving direction and magnitude).
    '''
    # convert to 3x3 rotation matrix
    rotation = R.from_quat(quaternion).as_matrix()
    # get second column of rotation matrix, which is the y-axis of tag frame aka direction vector of force, in terms of camera
    dir_vec = np.array([x[0] for x in rotation])
    # multiply dir_vec with a scalar for desired magnitude
    force_vec = (force_value / np.linalg.norm(dir_vec)) * dir_vec
    return force_vec


def translate_point(quaternion, translation):
    '''
    Translates a 3D point in tag frame to camera frame.
    '''
    rotation = R.from_quat(quaternion).as_matrix()
    point = np.array([-0.23, 0.0, -0.015, 1]) # 10 cm away from tip, 2cm is thickness of force gauge (TO BE UPDATED)
    tf_matrix = np.column_stack((rotation, translation))
    tf_matrix = np.row_stack((tf_matrix, [0,0,0,1]))
    result = np.matmul(tf_matrix, point) # [x, y, z, 1]
    return result[:3]


def project_point(point):
    '''
    Projects a 3D point to a 2D image.
    Output: [x, y] coordinate of the point on the image.
    '''
    objectPoints = point
    # rvec and tvec converts from world frame to camera frame, but since we're already in camera frame the values are set to identity
    rvec = np.eye(3)
    tvec = np.array([0.0, 0.0, 0.0])
    fx = 640.11767578125
    fy = 639.577697753906
    cx = 640.436645507812
    cy = 366.697723388672
    cameraMatrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    distCoeffs = np.array([-0.0572186261415482, 0.062038104981184, -0.0010403438936919, 0.000172375221154653, -0.0200830921530724])
    result = cv2.projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs)
    return result[0][0][0]


frame_files = natsorted(os.listdir('image_timestamp'))
tf_files = natsorted(os.listdir('position_data'))
force_files = natsorted(os.listdir('force_data'))
image_folders = natsorted(os.listdir('image_data'))

training_images = []
training_labels = []

for i in range(len(frame_files)):

    # read csv to dataframe
    tf_data = pd.read_csv('position_data/' + tf_files[i])
    force_data = pd.read_csv('force_data/' + force_files[i])
    frame_data = pd.read_csv('image_timestamp/' + frame_files[i])

    # sync tf and frame according to force b/c it's the sparsest
    tf_time = tf_data['timestamp'].values
    force_time = force_data['timestamp'].values
    frame_time = frame_data['timestamp'].values

    synced_force_ind = []
    synced_tf_ind = []
    synced_frame_ind = []
    tf_ind = 0
    frame_ind = 0
    for force_ind, t in enumerate(force_time):
        if (t <= frame_time[-1] and t <= tf_time[-1] and t >= 5.0):
            tf_ind = take_closest(tf_time, t, tf_ind, len(tf_time))
            frame_ind = take_closest(frame_time, t, frame_ind, len(frame_time))
            synced_force_ind.append(force_ind)
            synced_tf_ind.append(tf_ind)
            synced_frame_ind.append(frame_ind)


    # get corresponding data based on synced indices
    s1 = frame_data.iloc[synced_frame_ind, 1:]
    s1.reset_index(drop=True, inplace=True)
    s2 = force_data.iloc[synced_force_ind, 1:]
    s2.reset_index(drop=True, inplace=True)
    s3 = tf_data.iloc[synced_tf_ind, 1:]
    s3.reset_index(drop=True, inplace=True)

    # concat all data into dataframe
    synced_data = pd.concat([s1, s2, s3], ignore_index=True, axis=1)
    synced_data.columns = ['frame', 'force', 'translation-x', 'translation-y', 'translation-z', 'rotation-x', 'rotation-y', 'rotation-z', 'rotation-w']

    # compute force vector for each frame
    def f(x):
        quaternion = x[5:9].values
        force = x[1]
        return vector_decomp(x[5:9], x[1])

    synced_data['force_vector'] = synced_data.apply(f, axis=1, raw=True)

    # compute coordinate of contact point for each frame
    def g(x):
        quaternion = x[5:9].values
        translation = x[2:5].values
        contact_point = np.array(translate_point(quaternion, translation), dtype='float')
        return project_point(contact_point)

    synced_data['contact_point'] = synced_data.apply(g, axis=1, raw=True)

    # load corresponding images to synced frames
    image_dir_path = os.path.join('image_data', image_folders[i])
    all_images = natsorted(os.listdir(image_dir_path))
    frame_list = synced_data['frame'].values
    contact_point_list = synced_data['contact_point'].values

    for j, f in enumerate(frame_list):
        path = os.path.join(image_dir_path, all_images[f])
        image = skimage.io.imread(path)
        x = math.floor(contact_point_list[j][0])
        y = math.floor(contact_point_list[j][1])
        image = image[y-112:y+112, x-112:x+112, :]
        training_images.append(image)

    # write data to csv
    data_path = 'synced_data/' + frame_files[i]
    synced_data.to_csv(data_path)

    for label in synced_data['force_vector'].to_numpy():
        training_labels.append(label)

# outside for loop (consolidating data into one place for training)

# manually delete weird data
training_images = np.delete(training_images, 458, 0)
training_labels = np.delete(training_labels, 458, 0)

# write data to dataset folder
np.save('dataset/training_labels/training_labels.npy', training_labels)
print(np.shape(training_labels))

path = 'dataset/training_images/'
training_images = np.array(training_images)
for i, image in enumerate(training_images):
    skimage.io.imsave(os.path.join(path, '%d.png'%i), image)


# camera frame: x is right, y is down, z is straight ahead (from the camera, facing away from the viewer)
# IMPORTANT: note about tag orientation
# when oriented correctly: (looking at the tag) x right, y up, z straight ahead
# our orientation: (looking at the tag) x up, y left, z straight ahead
