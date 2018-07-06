import numpy as np
import tensorflow as tf
import cv2 as cv
import sys
import os
import time # for fps
import argparse

#uncomment to give input and output video in arguments
# video_inputpath = sys.argv[1]
# video_outputpath = sys.argv[2]
# Read the graph.
with tf.gfile.FastGFile('frozen_inference_graph.pb','rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Session() as sess:
    # Restore session
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    #uncomment to give input and output video in arguments
    # stream = cv.VideoCapture(video_inputpath)
    stream = cv.VideoCapture("test.mp4")
    if stream.isOpened():
        videowidth = int(stream.get(cv.CAP_PROP_FRAME_WIDTH))   # float
        videoheight = int(stream.get(cv.CAP_PROP_FRAME_HEIGHT)) # float
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        # fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
    # videoout = cv.VideoWriter(video_outputpath,fourcc=fourcc, fps=30, frameSize=(videowidth,videoheight))
    videoout = cv.VideoWriter("output.mp4",fourcc=fourcc, fps=30, frameSize=(videowidth,videoheight))

    while True:
        (grabbed, frame) = stream.read()  # grab and read the frames from video
        stime = time.time() # for fps 
   
    
        rows = frame.shape[0]
        cols = frame.shape[1]
        #print("%d %d" %(rows,cols))
        inp = cv.resize(frame, (300, 300))
        inp = inp[:, :, [2, 1, 0]]  # BGR2RGB
        rowsfps = int(rows - 0.05*rows)
        colsfps = int(cols - 0.1*cols)

    # Run the model
        out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                    sess.graph.get_tensor_by_name('detection_scores:0'),
                    sess.graph.get_tensor_by_name('detection_boxes:0'),
                    sess.graph.get_tensor_by_name('detection_classes:0')],
                   feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

    # Visualize detected bounding boxes.
        num_detections = int(out[0][0])
        for i in range(num_detections):
            classId = int(out[3][0][i])
            #print(classId)
            score = float(out[1][0][i])
            bbox = [float(v) for v in out[2][0][i]]
            if score > 0.7:
                x = bbox[1] * cols
                y = bbox[0] * rows
                right = bbox[3] * cols
                bottom = bbox[2] * rows
                if classId == 1:
                    cv.rectangle(frame, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)
                    cv.putText(frame, "Auto Rickshaw", (int(x), int(y)), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
                    # print("Autorick shaw score ",score) #Uncomment to measure accuracy of Autorick shaw
                else:
                    cv.rectangle(frame, (int(x), int(y)), (int(right), int(bottom)), (255, 51, 125), thickness=2)
                    cv.putText(frame, "Car", (int(x), int(y)), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
                    # print("Car score ",score) #Uncomment to measure accuracy of Car
        # cv.putText(frame, 'FPS: {:.1f}'.format(1 / (time.time() - stime)), (colsfps, rowsfps), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
        
        cv.imshow('TensorFlow MobileNet-SSD', frame)
        videoout.write(frame)
        if cv.waitKey(25) & 0xFF == ord('q'):
            cv.destroyAllWindows()
            break
    stream.release()
    cv.destroyAllWindows()