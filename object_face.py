import numpy as np
import tensorflow as tf
import sys
from PIL import Image
import cv2
cap = cv2.VideoCapture(0)

from utils import label_map_util
from utils import visualization_utils as vis_util

# ------------------ Face Model Initialization ------------------------------ #
face_label_map = label_map_util.load_labelmap('training_face/labelmap.pbtxt')
face_categories = label_map_util.convert_label_map_to_categories(
    face_label_map, max_num_classes=2, use_display_name=True)
face_category_index = label_map_util.create_category_index(face_categories)

face_detection_graph = tf.Graph()

with face_detection_graph.as_default():
    face_od_graph_def = tf.GraphDef()
    with tf.gfile.GFile('inference_graph_face_ssd/frozen_inference_graph.pb', 'rb') as fid:
        face_serialized_graph = fid.read()
        face_od_graph_def.ParseFromString(face_serialized_graph)
        tf.import_graph_def(face_od_graph_def, name='')

    face_session = tf.Session(graph=face_detection_graph)

face_image_tensor = face_detection_graph.get_tensor_by_name('image_tensor:0')
face_detection_boxes = face_detection_graph.get_tensor_by_name('detection_boxes:0')
face_detection_scores = face_detection_graph.get_tensor_by_name('detection_scores:0')
face_detection_classes = face_detection_graph.get_tensor_by_name('detection_classes:0')
face_num_detections = face_detection_graph.get_tensor_by_name('num_detections:0')
# ---------------------------------------------------------------------------- #

# ------------------ General Model Initialization ---------------------------- #
general_label_map = label_map_util.load_labelmap('data/mscoco_label_map.pbtxt')
general_categories = label_map_util.convert_label_map_to_categories(
    general_label_map, max_num_classes=90, use_display_name=True)
general_category_index = label_map_util.create_category_index(
    general_categories)

general_detection_graph = tf.Graph()

with general_detection_graph.as_default():
    general_od_graph_def = tf.GraphDef()
    with tf.gfile.GFile('ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb', 'rb') as fid:
        general_serialized_graph = fid.read()
        general_od_graph_def.ParseFromString(general_serialized_graph)
        tf.import_graph_def(general_od_graph_def, name='')

    general_session = tf.Session(graph=general_detection_graph)

general_image_tensor = general_detection_graph.get_tensor_by_name('image_tensor:0')
general_detection_boxes = general_detection_graph.get_tensor_by_name('detection_boxes:0')
general_detection_scores = general_detection_graph.get_tensor_by_name('detection_scores:0')
general_detection_classes = general_detection_graph.get_tensor_by_name('detection_classes:0')
general_num_detections = general_detection_graph.get_tensor_by_name('num_detections:0')
# ---------------------------------------------------------------------------- #


def face(image_np):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    (boxes, scores, classes, num) = face_session.run(
    [face_detection_boxes, face_detection_scores,
        face_detection_classes, face_num_detections],
        feed_dict={face_image_tensor: image_np_expanded})
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        face_category_index,
        use_normalized_coordinates=True,
        line_thickness= 8,
        min_score_thresh=0.85)
    # Get coordinates of detected boxes ; ymin, ymax, xmin, xmax
    coordinates_face = vis_util.return_coordinates(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        face_category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.85)

    print("face: ", *coordinates_face)

                                


def general(image_np):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    (boxes, scores, classes, num) = general_session.run(
    [general_detection_boxes, general_detection_scores,
        general_detection_classes, general_num_detections],
        feed_dict={general_image_tensor: image_np_expanded})
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        general_category_index,
        use_normalized_coordinates=True,
        line_thickness= 8)
    # Get coordinates of detected boxes ; ymin, ymax, xmin, xmax
    coordinates_object = vis_util.return_coordinates(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        general_category_index,
        use_normalized_coordinates=True,
        line_thickness=8)

    print("object: ", *coordinates_object)



if __name__ == '__main__':
    print(' in main')
