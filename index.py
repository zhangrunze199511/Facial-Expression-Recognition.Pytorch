from mtcnn.mtcnn import MTCNN
import cv2


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import _thread

import transforms as transforms
from skimage import io
from skimage.transform import resize
from models import *
from flask import Flask, jsonify, request
from flask_cors import CORS
from cam import Camera



app = Flask(__name__)
CORS(app)
cam = Camera(800, 600)
IMAGE_PATH = 'E:/workplace/electron_wp/Doubing_web_client/public/facial.png'

detector = MTCNN()

cut_size = 44
transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
colors = [(80, 80, 255), (210, 67, 91), (80, 255, 80), (80,255, 255), (255, 255, 80), (203, 148, 223), (80, 80, 80)]
net = VGG('VGG19')
checkpoint = torch.load(os.path.join('FER2013_VGG19', 'PrivateTest_model.t7'))
net.load_state_dict(checkpoint['net'])
net.cuda()
net.eval()
print('server ready')



def write(c1, c2, cls, img):
    label = "{0}".format(class_names[cls])
    color = colors[cls]
    # color = (0, 0, 255)
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img


def get_face_img(detector, img):
    results = detector.detect_faces(img)

    if len(results) == 0:
        cv2.imwrite(IMAGE_PATH, img)
        return None
    faces = []
    # for face_detected in results:
    face_detected = results[0]
    bb = face_detected['box']

    shape = img.shape
    x = 0 if (bb[0] - 10) < 0 else bb[0] - 10
    y = 0 if (bb[1] - 10) < 0 else bb[1] - 10
    xb = shape[1] if (bb[0] + bb[2] + 10 > shape[1]) else bb[0] + bb[2] + 10
    yb = shape[0] if (bb[1] + bb[3] + 10 > shape[0]) else bb[1] + bb[3] + 10

    face_img = img[y:yb, x:xb]
    faces.append(face_img)

    #     cv2.rectangle(img, (x, y), (xb, yb), (0, 0, 255), 2)
    #
    # cv2.imwrite('E:/workplace/electron_wp/Doubing Project/public/facial.png', img)

    return faces, (x,y), (xb, yb)






def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def get_emotion_from_face_img(image):
    raw_img, c1, c2 = get_face_img(detector, image)

    if len(raw_img) == 0:
        return None
    gray = rgb2gray(np.array(raw_img[0]))
    gray = resize(gray, (48, 48), mode='symmetric').astype(np.uint8)

    img = gray[:, :, np.newaxis]

    img = np.concatenate((img, img, img), axis=2)
    img = Image.fromarray(img)
    inputs = transform_test(img)

    ncrops, c, h, w = np.shape(inputs)

    inputs = inputs.view(-1, c, h, w)
    inputs = inputs.cuda()
    inputs = Variable(inputs, volatile=True)
    outputs = net(inputs)

    outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops

    score = F.softmax(outputs_avg)
    _, predicted = torch.max(outputs_avg.data, 0)

    cls =int(predicted.cpu().numpy())

    new_image = write(c1, c2, cls, image)

    # cv2.rectangle(image, c1, c2, (0, 0, 255), 2)

    cv2.imwrite(IMAGE_PATH, new_image)



    return int(predicted.cpu().numpy())

@app.route('/')
def hello_world():
    return jsonify({
        'success': True
    })
#

#
@app.route('/emotion/update', methods=['POST'])
def cam_update_img():
    global cam
    if not cam:
        cam = Camera(800, 600)
    try:
        result = cam.start_once(get_emotion_from_face_img)
        # _thread.start_new_thread(hello, (1234, ))
        if result == None:
            return jsonify({
                'success': True,
                'result': -1
            })

        return jsonify({
            'success': True,
            'result': result,
            'emotion': class_names[result]
        })
    except:
        print("Error: start thread error ")
        return jsonify({
            'success': False,
        })



if __name__ == '__main__':
    app.run()

