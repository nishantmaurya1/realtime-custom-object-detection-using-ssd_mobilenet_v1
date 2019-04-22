# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 10:59:44 2019

@author: nishant-765461
"""

import numpy as np
import json
import os
import tensorflow as tf
import wx


from utils import label_map_util

from utils import visualization_utils as vis_util

import cv2

# Path to frozen detection graph. This is the actual model that is used
# for the object detection.
PATH_TO_CKPT = 'C:/Users/765461/Desktop/helmet_detection/models/object_detection/helmet_detection_graph/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('C:/Users/765461/Desktop/helmet_detection/models/object_detection/training', 'object-detection.pbtxt')

NUM_CLASSES = 2
def upload_video():
    if __name__ == "__main__":
        app = wx.PySimpleApp()
    wildcard = "Python source (*.py)|*.py|" \
            "Compiled Python (*.pyc)|*.pyc|" \
            "All files (*.*)|*.*"
    dialog = wx.FileDialog(None, "Choose a file", os.getcwd(), 
            "", wildcard, wx.FD_OPEN)
    if dialog.ShowModal() == wx.ID_OK:
       file=dialog.GetPath()
    detect_in_video(file)
    dialog.Destroy()
def detect_in_video(file):

    # VideoWriter is the responsible of creating a copy of the video
    # used for the detections but with the detections overlays. Keep in
    # mind the frame size has to be the same as original video.
    out = cv2.VideoWriter('video2.mp4', cv2.VideoWriter_fourcc(
        'M', 'J', 'P', 'G'), 10, (856,478))

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object
            # was detected.
            detection_boxes = detection_graph.get_tensor_by_name(
                'detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class
            # label.
            detection_scores = detection_graph.get_tensor_by_name(
                'detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name(
                'detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name(
                'num_detections:0')
            cap = cv2.VideoCapture(file)

            while(cap.isOpened()):
                # Read the frame
                ret, frame = cap.read()

                # Recolor the frame. By default, OpenCV uses BGR color space.
                # This short blog post explains this better:
                # https://www.learnopencv.com/why-does-opencv-use-bgr-color-format/
                color_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                image_np_expanded = np.expand_dims(color_frame, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores,
                        detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                 # Here output the category as string and score to terminal
                print([category_index.get(i) for i in classes[0]],scores)
                print("detected: ",classes,"with accuracy : ",scores)
                print("Total score for %s is %s" % (classes, scores))
                with open('data.json', 'w') as outfile:
                    json.dump([category_index.get(i) for i in classes[0]], outfile)
                
                # Visualization of the results of a detection.
                # note: perform the detections using a higher threshold
                vis_util.visualize_boxes_and_labels_on_image_array(
                    color_frame,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8,
                    min_score_thresh=.20)

                cv2.imshow('frame', color_frame)
                output_rgb = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)
                out.write(output_rgb)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            out.release()
            cap.release()
            cv2.destroyAllWindows()
            def OnExit(self, event):
                self.Destroy()
                self.Close()
class MyFrame(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, None, -1,
                          "Monitoring")
        p = wx.Panel(self)
        topLbl = wx.StaticText(p, -1, "Worker Safety Compliance Monitoring and Controlling")
        topLbl.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD))
        
        mainSizer = wx.BoxSizer(wx.VERTICAL)
        mainSizer.Add(topLbl, 0, wx.ALL, 5)
        mainSizer.Add(wx.StaticLine(p), 0,
                wx.EXPAND|wx.TOP|wx.BOTTOM, 5)
        
        btn = wx.Button(p, -1, "Choose Video")
        btn1 = wx.Button(p, -1, "Camera Feed")
        addrSizer = wx.FlexGridSizer(cols=2, hgap=5, vgap=5)
        addrSizer.AddGrowableCol(1)
        mainSizer = wx.BoxSizer(wx.VERTICAL)
        mainSizer.Add(topLbl, 0, wx.ALL, 5)
        mainSizer.Add(wx.StaticLine(p), 0,
                wx.EXPAND|wx.TOP|wx.BOTTOM, 5)
        mainSizer.Add(addrSizer, 0, wx.EXPAND|wx.ALL, 10)
        lbl=wx.StaticText(p,-1,style=wx.ALIGN_CENTER)
        self.Bind(wx.EVT_BUTTON, self.OnAddItem, btn)
        self.Bind(wx.EVT_BUTTON, self.OnAddItem, btn1)
        btnSizer = wx.BoxSizer(wx.HORIZONTAL)
        btnSizer.Add((20,20), 1)
        btnSizer.Add(btn)
        btnSizer.Add((20,20), 1)
        btnSizer.Add(btn1)
        btnSizer.Add((20,20), 1)

        mainSizer.Add(btnSizer, 0, wx.EXPAND|wx.BOTTOM, 10)

        p.SetSizer(mainSizer)

        # Fit the frame to the needs of the sizer.  The frame will
        # automatically resize the panel as needed.  Also prevent the
        # frame from getting smaller than this size.
        mainSizer.Fit(self)
        mainSizer.SetSizeHints(self)
    def OnAddItem(self, event):
        upload_video()
        def OnExit(self, event):
            self.Destroy()
            self.Close()
    def OnExit(self, event):
        self.Destroy()
        self.Close()

if __name__ == '__main__':
    app = wx.PySimpleApp()
    frame = MyFrame()
    frame.Show()
    app.MainLoop()
