#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
import sys
import cv2
import warnings
import numpy as np
from PIL import Image
from yolo import YOLO
from timeit import time
from moviepy.editor import VideoFileClip

from deep_sort import nn_matching
from deep_sort import preprocessing
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet

warnings.filterwarnings('ignore')

# Function to check whether a point is inside a defined area
def center_point_inside_polygon(bounding_box, polygon_coord):
    center = (int((bounding_box[0] + bounding_box[2])/2), int(bounding_box[3]))
    polygon_coord = np.array(polygon_coord, np.int32)
    polygon_coord = polygon_coord.reshape((-1, 1, 2))
    result = cv2.pointPolygonTest(polygon_coord, center, False)
    if result == -1:
        return "outside"
    return "inside"

def main(yolo):

    filename = "cyber.mp4"
    clip = VideoFileClip(filename)

    cap2 = cv2.VideoCapture(filename)
    co = 0
    ret2 = True
    while ret2:
        ret2, frame2 = cap2.read()
        co += 1
    cap2.release()

    Input_FPS = co / clip.duration

   # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    frame_count = 0
    
   # deep_sort 
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = True 
    
    video_capture = cv2.VideoCapture(filename)

    if writeVideo_flag:
    # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter('output.mp4', fourcc, Input_FPS, (w, h))
        
    fps = 0.0

    # Dictionary to store the number of frames each frame_id was present in the defined area
    track_dict = {}

    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break

        t1 = time.time()

       # image = Image.fromarray(frame)
        image = Image.fromarray(frame[...,::-1]) #bgr to rgb
        boxs = yolo.detect_image(image)
       # print("box_num",len(boxs))
        features = encoder(frame,boxs)
        
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        
        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # Defining the co-ordinates of the area of interest
        pts = np.array([[855,201],[1015.5,190.5],[1018.5,318],[766.5,316.5]], np.int32)
        pts = pts.reshape((-1,1,2)) # Yellow box
        pts2 = np.array([[766.5,316.5],[1018.5,318.0],[1040,720],[510.0,720]], np.int32)
        pts2 = pts2.reshape((-1,1,2)) # Pink box
        cv2.polylines(frame, [pts], True, (0,255,255), thickness=2)
        cv2.polylines(frame, [pts2], True, (255,0,255), thickness=2)
        
        # Drawing tracker boxes and frame count for people inside the area of interest
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()

            point_test = center_point_inside_polygon(bbox, pts)

            if point_test == 'inside':
                
                if track.track_id not in track_dict.keys():
                    track_dict[track.track_id] = 1
                else:
                    track_dict[track.track_id] += 1

                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
                wait_time = round((track_dict[track.track_id] / Input_FPS), 2)
                cv2.putText(frame, str(track.track_id) + "->Time:" + str(wait_time) + " seconds",(int(bbox[0]), int(bbox[1])),0, 0.8, (0,255,0),2)

            point_test = center_point_inside_polygon(bbox, pts2)

            if point_test == 'inside':
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
                cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1])),0, 0.8, (0,255,0),2)

        # Drawing bounding box detections for people inside the area of interest
        for det in detections:
            bbox = det.to_tlbr()

            point_test = center_point_inside_polygon(bbox, pts)

            if point_test == 'inside':
                cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)

            point_test = center_point_inside_polygon(bbox, pts2)

            if point_test == 'inside':
                cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)
        
        if writeVideo_flag:
            # save a frame
            out.write(frame)
            
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        frame_count += 1

        op = "FPS_" + str(frame_count) + ": " + str(round(fps, 2))
        print("\r" + op , end = "")
        
        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
 
    print("\n-----------------------------------------------------------------------")
    print("WAIT TIME ESTIMATION ( Unique Person ID -> Time spent in counter area )\n")
    for k, v in track_dict.items():
        print(k, "->", str(round((v/Input_FPS), 2)) + " seconds")

    video_capture.release()
    if writeVideo_flag:
        out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(YOLO())
