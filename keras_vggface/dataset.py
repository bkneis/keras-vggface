import os
import cv2
import numpy as np


class DatasetError(Exception):
    pass


def get_training_data() -> (np.ndarray, np.ndarray, int):
    # Get total number of examples and init labels and data array
    root_dir = '/home/arthur/projects/keras-vggface/data/'
    subject_dirs = get_immediate_subdirectories(root_dir)
    n_classes = len(subject_dirs)

    num_samples = 0

    for idx, subject_dir in enumerate(subject_dirs):
        # Loop over each image of subject
        subject_path = os.path.join(root_dir, subject_dir)
        session_dirs = get_immediate_subdirectories(subject_path)
        # Ignore any subjects with only on session
        num_sessions = len(session_dirs)
        if num_sessions < 2:
            print('less than 2', subject_dir)
            continue
        for _ in session_dirs:
            num_samples += 1

    data = []  # np.zeros((num_samples, 224, 224, 3))
    labels = [] # np.zeros((num_samples, 1))

    # Loop over the folder structure for each subject
    for idx, subject_dir in enumerate(subject_dirs):
        # Loop over each image of subject
        subject_path = os.path.join(root_dir, subject_dir)
        print('Subject %s is ID %s ' % (subject_path, idx))
        session_dirs = get_immediate_subdirectories(subject_path)
        # Ignore any subjects with only on session
        num_sessions = len(session_dirs)
        if num_sessions < 2:
            print('less than 2', subject_dir)
            continue
        for session_id, session_dir in enumerate(session_dirs):
            sub_path = os.path.join(subject_path, session_dir)
            # Transform data and append to array with label
            img = cv2.imread(sub_path + '/average.png')
            if img is None:
                raise DatasetError
            try:
                img = cv2.resize(img, (224, 224))
            except cv2.Error as err:
                raise DatasetError(err)

            data.append(img)
            labels.append(int(idx))
            # data = np.append(data, img)
            # labels = np.append(labels, int(subject_dir))

    # Return the data, labels and number of classes
    return np.array(data), np.array(labels).reshape(-1,1), n_classes


def prune_dataset():
    pass


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]
