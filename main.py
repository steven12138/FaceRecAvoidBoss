import cv2
import numpy as np
import dlib
import os
import win32api
import win32con
import time
from pynput import keyboard

pressed = set()


def on_press(key):
    global pressed
    pressed.add(key)


def on_release(key):
    global pressed
    pressed.remove(key)


print("wait keycode")


# Collect events until released

def listen():
    with keyboard.Listener(
            on_press=on_press,
            on_release=on_release) as listener:
        listener.join()


import _thread

try:
    _thread.start_new_thread(listen, ())
except BaseException:
    print("ERROR TO START THREAD")

print("start")
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")

# face_detector = dlib.get_frontal_face_detector()
face_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")


class HideState:
    def __init__(self):
        self.is_hide = False
        self.last_operation = 0
        self.hide_cnt = 0
        self.r = .5
        self.v = 10

    def hide(self):
        if len(pressed):
            return
        self.hide_cnt += 1;
        if self.hide_cnt < self.v:
            return
        else:
            self.hide_cnt = 0
        if time.time() - self.last_operation < self.r:
            return
        if not self.is_hide:
            self.is_hide = True
            self.last_operation = time.time()
            win32api.keybd_event(162, 0, 0, 0)
            win32api.keybd_event(91, 0, 0, 0)
            win32api.keybd_event(39, 0, 0, 0)
            win32api.keybd_event(39, 0, win32con.KEYEVENTF_KEYUP, 0)
            win32api.keybd_event(91, 0, win32con.KEYEVENTF_KEYUP, 0)
            win32api.keybd_event(162, 0, win32con.KEYEVENTF_KEYUP, 0)

    def show(self):
        if len(pressed):
            return
        if time.time() - self.last_operation < self.r:
            return
        if self.is_hide:
            self.is_hide = False
            self.last_operation = time.time()
            win32api.keybd_event(162, 0, 0, 0)
            win32api.keybd_event(91, 0, 0, 0)
            win32api.keybd_event(37, 0, 0, 0)
            win32api.keybd_event(37, 0, win32con.KEYEVENTF_KEYUP, 0)
            win32api.keybd_event(91, 0, win32con.KEYEVENTF_KEYUP, 0)
            win32api.keybd_event(162, 0, win32con.KEYEVENTF_KEYUP, 0)


def load_faces():
    dir = os.listdir('faces')
    known_faces = {}
    for file in dir:
        path = "faces/" + file
        name = file.split(".")[0].split("_")[1]
        face = np.load(path)
        known_faces[name] = face
    return known_faces


known_faces = load_faces()

threshold = .2


def compare_face(face1, face2):
    diff = 0
    for i in range(len(face1)):
        diff += (face1[i] - face2[i]) ** 2
    # diff = np.sqrt(diff)
    # print("diff", diff)
    if diff < threshold:
        return True
    else:
        return False


def rec_face(face):
    global known_faces
    for name in known_faces:
        target_face = known_faces[name]
        if compare_face(face, target_face):
            return name
    return False


hide_ctrl = HideState()

capture = cv2.VideoCapture(0)
while True:
    _, frame = capture.read()
    b, g, r = cv2.split(frame)
    img2 = cv2.merge([r, g, b])
    h = len(frame)
    w = len(frame[0])
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104., 117., 123.], False, False)

    net.setInput(blob)
    detections = net.forward()

    detected_faces = 0
    has_unknown = False
    for i in range(0, detections.shape[2]):
        # 获取当前检测结果的置信度
        confidence = detections[0, 0, i, 2]
        # 如果置信大于最小置信度，则将其可视化
        if confidence > 0.7:
            detected_faces += 1
            # 获取当前检测结果的坐标
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            # print(box)
            (startX, startY, endX, endY) = box.astype('int')
            # 绘制检测结果和置信度
            text = "{:.3f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            # face = frame[startY:endY, startX:endX]
            face = dlib.rectangle(startX, startY, endX, endY)
            shape = face_predictor(img2, face)
            # print("shape", shape)
            # for _, pt in enumerate(shape.parts()):
            #     pt_pos = (pt.x, pt.y)
            # cv2.circle(frame, pt_pos, 2, (255, 0, 0), 1)
            # cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 3)
            # cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            face_descriptor = face_rec_model.compute_face_descriptor(img2, shape)
            face_descriptor = np.array(face_descriptor)
            name = rec_face(face_descriptor)
            cv2.putText(frame, text + " " + (name if name else "???"), (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 0, 255), 2)
            if not name:
                hide_ctrl.hide()
                has_unknown = True
    if not has_unknown:
        hide_ctrl.show()
    # dets = face_detector(frame, 1)
    # for index, face in enumerate(dets):
    #     print('face {}; left {}; top {}; right {}; bottom {}'.format(index, face.left(), face.top(), face.right(),
    #                                                                  face.bottom()))
    #     print()
    #     # 在图片中标注人脸，并显示
    #     left = face.left()
    #     top = face.top()
    #     right = face.right()
    #     bottom = face.bottom()
    #     cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
    # 显示图片
    cv2.imshow('img', frame)
    key = cv2.waitKey(1)
    if key == 27:  # 判断是哪一个键按下
        break
cv2.destroyAllWindows()
exit(0)
