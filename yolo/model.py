from net.netarch import YoloArchitecture, YoloInferenceModel


class YOLO(object):

    def __init__(self):
        self.debug_timings = True
        self.yolo_arch = YoloArchitecture()
        self.model = self.yolo_arch.get_model()
        self.inf_model = YoloInferenceModel(self.model)

    def predict(self, frame):
        boxes_labels = self.inf_model.predict(frame)
        boxes = boxes_labels[0]
        labels = boxes_labels[1]

        print("Object Identification Process Finished")
        return boxes, labels
