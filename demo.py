#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

# Importing standard libraries
import os
import sys
import cv2
import csv
import warnings
import numpy as np
from PIL import Image
from yolo import YOLO
from timeit import time
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

# Importing other custom .py files
from deep_sort import nn_matching
from deep_sort import preprocessing
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet

warnings.filterwarnings('ignore')

# Function to check whether a point is inside a defined area
# We are checking if the bottom line of the bounding box enters an area of interest
def center_point_inside_polygon(bounding_box, polygon_coord):
    center = (int((bounding_box[0] + bounding_box[2])/2), int(bounding_box[3]))
    polygon_coord = np.array(polygon_coord, np.int32)
    polygon_coord = polygon_coord.reshape((-1, 1, 2))
    result = cv2.pointPolygonTest(polygon_coord, center, False)
    if result == -1:
        return "outside"
    return "inside"

# Main Function which implements the YOLOv3 Detector and DeepSort Tracking Algorithm
def main(yolo):

    # Determining the FPS of a video having variable frame rate
    # cv2.CAP_PROP_FPS is not used since it returns 'infinity' for variable frame rate videos
    filename = "cyber.mp4"
    # Determining the total duration of the video
    clip = VideoFileClip(filename)

    cap2 = cv2.VideoCapture(filename)
    co = 0
    ret2 = True
    while ret2:
        ret2, frame2 = cap2.read()
        # Determining the total number of frames
        co += 1
    cap2.release()

    # Computing the average FPS of the video
    Input_FPS = co / clip.duration

    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    frame_count = 0
    
    # Implementing Deep Sort algorithm
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    
    # Cosine distance is used as the metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    
    video_capture = cv2.VideoCapture(filename)

    # Define the codec and create a VideoWriter object to save the output video
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'MP4V'), Input_FPS, (int(video_capture.get(3)), int(video_capture.get(4))))

    # To calculate the frames processed by the deep sort algorithm per second
    fps = 0.0

    # Initializing empty variables for counting and tracking purpose
    queue_track_dict = {}         # Count time in queue
    alley_track_dict = {}         # Count time in alley
    store_track_dict = {}         # Count total time in store
    latest_frame = {}             # Track the last frame in which a person was identified
    reidentified = {}             # Yes or No : whether the person has been re-identified at a later point in time
    plot_head_count_store = []    # y-axis for Footfall Analysis
    plot_head_count_queue = []    # y-axis for Footfall Analysis
    plot_time = []                # x-axis for Footfall Analysis

    # Loop to process each frame and track people
    while True:
        ret, frame = video_capture.read()
        if ret != True:
            break

        head_count_store = 0
        head_count_queue = 0
        t1 = time.time()

        image = Image.fromarray(frame[...,::-1])   # BGR to RGB conversion
        boxs = yolo.detect_image(image)
        features = encoder(frame,boxs)
        
        # Getting the detections having score of 0.0 to 1.0
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        
        # Run non-maxima suppression on the bounding boxes
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        
        # Call the tracker to associate tracking boxes to detection boxes
        tracker.predict()
        tracker.update(detections)

        # Defining the co-ordinates of the area of interest
        pts = np.array([[790,230],[1025,230],[1020,400],[630,400]], np.int32)
        pts = pts.reshape((-1,1,2))     # Queue Area
        pts2 = np.array([[630,400],[1020,400],[970,720],[300.0,720]], np.int32)
        pts2 = pts2.reshape((-1,1,2))   # Alley Region
        cv2.polylines(frame, [pts], True, (0,255,255), thickness=2)
        cv2.polylines(frame, [pts2], True, (255,0,255), thickness=2)
        
        # Drawing tracker boxes and frame count for people inside the areas of interest
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()

            # Checking if the person is within an area of interest
            queue_point_test = center_point_inside_polygon(bbox, pts)
            alley_point_test = center_point_inside_polygon(bbox, pts2)

            # Checking if a person has been reidentified in a later frame
            if queue_point_test == 'inside' or alley_point_test == 'inside':
                if track.track_id in latest_frame.keys():
                    if latest_frame[track.track_id] != frame_count - 1:
                        reidentified[track.track_id] = 1

            # Initializing variables incase a new person has been seen by the model
            if queue_point_test == 'inside' or alley_point_test == 'inside':
                head_count_store += 1
                if track.track_id not in store_track_dict.keys():
                    store_track_dict[track.track_id] = 0
                    queue_track_dict[track.track_id] = 0
                    alley_track_dict[track.track_id] = 0
                    reidentified[track.track_id] = 0

            # Processing for people inside the Queue Area
            if queue_point_test == 'inside':
                head_count_queue += 1
                queue_track_dict[track.track_id] += 1
                latest_frame[track.track_id] = frame_count
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
                wait_time = round((queue_track_dict[track.track_id] / Input_FPS), 2)
                cv2.putText(frame, str(track.track_id) + "->Time:" + str(wait_time) + " seconds", (int(bbox[0]), int(bbox[1])), 0, 0.8, (0, 0, 0), 4)
                cv2.putText(frame, str(track.track_id) + "->Time:" + str(wait_time) + " seconds", (int(bbox[0]), int(bbox[1])), 0, 0.8, (0, 255, 77), 2)

            # Processing for people inside the Alley Region
            if alley_point_test == 'inside':
                alley_track_dict[track.track_id] += 1
                latest_frame[track.track_id] = frame_count
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
                cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1])), 0, 0.8, (0, 0, 0), 4)
                cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1])), 0, 0.8, (0, 255, 77), 2)

            # Getting the total Store time for a person
            if track.track_id in store_track_dict.keys():
                store_track_dict[track.track_id] = queue_track_dict[track.track_id] + alley_track_dict[track.track_id]

        # Drawing bounding box detections for people inside the store
        for det in detections:
            bbox = det.to_tlbr()

            # Checking if the person is within an area of interest
            queue_point_test = center_point_inside_polygon(bbox, pts)
            alley_point_test = center_point_inside_polygon(bbox, pts2)

            if queue_point_test == 'inside' or alley_point_test == 'inside':
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255,0,0), 2)

        # Video Overlay - Head Count Data at that instant
        cv2.putText(frame, "Head Count: " + str(head_count_store), ( 30, 610 ), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.7, (0, 0, 0), 3, cv2.LINE_AA, False)
        cv2.putText(frame, "Head Count: " + str(head_count_store), ( 30, 610 ), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.7, (0, 255, 77), 2, cv2.LINE_AA, False)

        # Calculating the average wait time in queue
        total_people = len([v for v in queue_track_dict.values() if v > 0])
        total_queue_frames = sum(v for v in queue_track_dict.values() if v > 0)
        avg_queue_frames = 0
        if total_people != 0:
            avg_queue_frames = total_queue_frames / total_people
        avg_queue_time = round((avg_queue_frames / Input_FPS), 2)

        # Video Overlay - Average Wait Time in Queue
        cv2.putText(frame, "Average Queue Time: " + str(avg_queue_time) + ' seconds', ( 30, 690 ), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.7, (0, 0, 0), 3, cv2.LINE_AA, False)
        cv2.putText(frame, "Average Queue Time: " + str(avg_queue_time) + ' seconds', ( 30, 690 ), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.7, (0, 255, 77), 2, cv2.LINE_AA, False)

        # Calculating the average wait time in the store
        total_people = len(store_track_dict)
        total_store_frames = sum(store_track_dict.values())
        avg_store_frames = 0
        if total_people != 0:
            avg_store_frames = total_store_frames / total_people
        avg_store_time = round((avg_store_frames / Input_FPS), 2)

        # Video Overlay - Average Store time
        cv2.putText(frame, "Average Store Time: " + str(avg_store_time) + ' seconds', ( 30, 650 ), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.7, (0, 0, 0), 3, cv2.LINE_AA, False)
        cv2.putText(frame, "Average Store Time: " + str(avg_store_time) + ' seconds', ( 30, 650 ), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.7, (0, 255, 77), 2, cv2.LINE_AA, False)

        # Write the frame onto the VideoWriter object
        out.write(frame)

        # Calculating the frames processed per second by the model  
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        frame_count += 1

        # Printing processing status to track completion
        op = "FPS_" + str(frame_count) + ": " + str(round(fps, 2))
        print("\r" + op , end = "")

        # Adding plot values for Footfall Analysis every 2 seconds (hard coded for now)
        if frame_count % 50 == 0:
            plot_time.append(round((frame_count / Input_FPS), 2))
            plot_head_count_store.append(head_count_store)
            plot_head_count_queue.append(head_count_queue)
        
        # Press Q to stop the video
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Data Processed as per the video provided
    print("\n-----------------------------------------------------------------------")
    print("QUEUE WAIT TIME ( Unique Person ID -> Time spent )\n")
    for k, v in queue_track_dict.items():
        print(k, "->", str(round((v/Input_FPS), 2)) + " seconds")

    print("\n-----------------------------------------------------------------------")
    print("ALLEY TIME ( Unique Person ID -> Time spent )\n")
    for k, v in alley_track_dict.items():
        print(k, "->", str(round((v/Input_FPS), 2)) + " seconds")

    print("\n-----------------------------------------------------------------------")
    print("STORE TIME ( Unique Person ID -> Time spent  )\n")
    for k, v in store_track_dict.items():
        print(k, "->", str(round((v/Input_FPS), 2)) + " seconds")

    # Defining data to be written into the csv file
    csv_columns = ['Unique Person ID', 'Queue Time in AOI', 'Total Store Time', 'Re-Identified']
    csv_data = []
    csv_row = {}
    csv_file = 'Store_Data.csv'
    for k, v in store_track_dict.items():
         csv_row = {}
         if reidentified[k] == 1:
             reid = 'Yes'
         else:
             reid = 'No'
         csv_row = {csv_columns[0]: k, csv_columns[1]: round((queue_track_dict[k] / Input_FPS), 2), csv_columns[2]: round((v / Input_FPS), 2), csv_columns[3]: reid}
         csv_data.append(csv_row)

    # Writing the data into the csv file
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in csv_data:
            writer.writerow(data)

    # Plotting the stack area graph and saving it as a .png file
    plot_head_count = np.vstack([plot_head_count_queue, plot_head_count_store])
    labels = ["Queue Head Count", "Store Head Count"]
    fig, ax = plt.subplots()
    ax.stackplot(plot_time, plot_head_count, labels = labels)
    ax.legend(loc='upper left')
    plt.xlabel('Time Stamp (in seconds)')
    plt.ylabel('Head Count')
    plt.xlim(0, round(frame_count / Input_FPS) + 1)
    plt.ylim(0, max(plot_head_count_store) + 2)
    plt.title('Footfall Analysis')
    plt.savefig('Footfall_Analysis.png', bbox_inches='tight')

    # Releasing objects created
    video_capture.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(YOLO())
