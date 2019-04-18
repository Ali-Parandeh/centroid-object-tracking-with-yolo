# import the necessary packages
import os

import cv2
import imutils
import numpy as np
import natsort

from pyimagesearch.centroidtracker import CentroidTracker
from yolo.model import YOLO
from tqdm import tqdm


class Object_Tracker():

    def __init__(self):
        # initialize our centroid tracker and frame dimensions
        self.ct = CentroidTracker()

    def generate_object_list_of_frame_with_id(self, folder_path):
        # checking whether the given path is a directory
        if os.path.isdir(folder_path):
            fnames = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                      if os.path.isfile(os.path.join(folder_path, f))]

        else:
            fnames = [folder_path]

        # fnames.sort(key=lambda f: int(filter(str.isdigit, f)))
        fnames = natsort.natsorted(fnames, reverse=False)

        frame_predictions_dictionary_with_id = {}

        for f in tqdm(fnames, desc='Processing Batch'):

            (H, W) = (None, None)

            frame_output_path = "./data/generated_frames/"

            try:
                if not os.path.exists(frame_output_path):
                    os.makedirs(frame_output_path)
            except OSError:
                print('Error: Creating directory of data')

            frame_object = cv2.imread(f)

            # tracking
            frame = imutils.resize(frame_object, width=400)

            # if the frame dimensions are None, grab them
            if W is None or H is None:
                (H, W) = frame.shape[:2]

            # predict the objects in the frame
            predictions = YOLO().predict(frame)

            rects = []
            centroids = []
            boxes = predictions[0]
            labels = predictions[1]
            frame_predictions_dictionary_with_id[str(f)[len(folder_path):]] = []

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

                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (color_mod, 255, 0), 2)

                centroids.append([int((xmin + xmax) / 2), int((ymin + ymax) / 2)])

            # update our centroid tracker using the computed set of bounding
            # box rectangles
            objects = self.ct.update(rects)

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
                    frame_predictions_dictionary_with_id[str(f)[len(folder_path):]].append(text)

                cv2.putText(image, text, (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

            # show the output frame
            cv2.imshow("Frame", image)
            key = cv2.waitKey(1) & 0xFF

        # do a bit of cleanup
        cv2.destroyAllWindows()

        return frame_predictions_dictionary_with_id


def run():
    ot = Object_Tracker()
    a = ot.generate_object_list_of_frame_with_id("./data/generated_frames/")
    print(a)


if __name__ == "__main__":
    run()
