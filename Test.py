import cv2
import numpy as np

import dlib

net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")

# face_detector = dlib.get_frontal_face_detector()
face_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

capture = cv2.VideoCapture(0)
ret, frame = capture.read()
b, g, r = cv2.split(frame)
img2 = cv2.merge([r, g, b])
h = len(frame)
w = len(frame[0])
blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104., 117., 123.], False, False)

net.setInput(blob)
detections = net.forward()

detected_faces = 0

for i in range(0, detections.shape[2]):
    # 获取当前检测结果的置信度
    confidence = detections[0, 0, i, 2]
    # 如果置信大于最小置信度，则将其可视化
    if confidence > 0.7:
        detected_faces += 1
        # 获取当前检测结果的坐标
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        print(box)
        (startX, startY, endX, endY) = box.astype('int')
        # 绘制检测结果和置信度
        text = "{:.3f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        # face = frame[startY:endY, startX:endX]
        face = dlib.rectangle(startX, startY, endX, endY)
        shape = face_predictor(img2, face)
        # print("shape", shape)
        for _, pt in enumerate(shape.parts()):
            pt_pos = (pt.x, pt.y)
            cv2.circle(frame, pt_pos, 2, (255, 0, 0), 1)
        # cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 3)
        cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        face_descriptor = face_rec_model.compute_face_descriptor(img2, shape)
        np.save(f"./faces/face_{detected_faces}", np.array(face_descriptor))
print(f'detect {detected_faces} faces')
# 显示图片
cv2.imshow('img', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
