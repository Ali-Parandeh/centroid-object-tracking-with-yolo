# import the necessary packages
import os

import cv2
import imutils
import numpy as np

from pyimagesearch.centroidtracker import CentroidTracker
from yolo.model import YOLO

# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()
(H, W) = (None, None)

video_capture = cv2.VideoCapture("./data/Kids_playing_in_leaves.mp4")

frame_output_path = "./data/generated_frames/"

try:
    if not os.path.exists(frame_output_path):
        os.makedirs(frame_output_path)
except OSError:
    print('Error: Creating directory of data')

# Capture the very first frame
return_status, frame = video_capture.read()

current_frame = 0
count = 0
frame_predictions_dictionary_with_id = {}

while return_status:

    if (count == 4):
        # tracking
        frame = imutils.resize(frame, width=400)

        # if the frame dimensions are None, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # Saving the current frame's image as a jpg file
        frame_name = "frame" + str(current_frame) + ".jpg"
        frame_location = frame_output_path + frame_name
        print("Creating..." + frame_location)
        cv2.imwrite(frame_location, frame)

        predictions = YOLO().predict(frame)

        rects = []
        centroids = []
        boxes = predictions[0]
        labels = predictions[1]
        frame_predictions_dictionary_with_id[frame_name] = []

        image = frame.copy()
        image_h, image_w, _ = image.shape

        color_mod = 255

        for i in range(len(boxes)):
            xmin = int(boxes[i][0] * image_w)
            ymin = int(boxes[i][1] * image_h)
            xmax = int(boxes[i][2] * image_w)
            ymax = int(boxes[i][3] * image_h)

            temp = np.array([xmin, ymin, xmax, ymax])

            rects.append(temp.astype("int"))
            text = labels[i]

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (color_mod, 255, 0), 2)

            centroids.append([int((xmin + xmax) / 2), int((ymin + ymax) / 2)])

        # update our centroid tracker using the computed set of bounding
        # box rectangles
        objects = ct.update(rects)

        # loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)

            cv2.circle(image, (centroid[0], centroid[1]), 2, (0, 255, 0), -1)
            centroid_co = [centroid[0], centroid[1]]

            if centroid_co in centroids:
                i = centroids.index(centroid_co)
                text = "{} - ID {}".format(labels[i], objectID)
                frame_predictions_dictionary_with_id[frame_name].append(text)

            cv2.putText(image, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

        # show the output frame
        cv2.imshow("Frame", image)
        key = cv2.waitKey(1) & 0xFF

        current_frame += 1
        count = 0

    print(frame_predictions_dictionary_with_id)
    # Capture frame-by-frame
    return_status, frame = video_capture.read()
    count += 1

# do a bit of cleanup
cv2.destroyAllWindows()
video_capture.release()
