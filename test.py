import os
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr

from keras.models import load_model
from keras.optimizers import Adam
import cv2
import numpy as np

def main(digit, option):
    model = load_model('models/model.h5')

    # Adjusting image
    file_name="images/"+digit+"_"+option+".png"
    img = cv2.imread(file_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(28,28))
    img = np.reshape(img,[784])
    img = np.array([img])

    classes = model.predict_classes(img)
    print("Predicted class:",classes[0])

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--digit', type=str, default="0",
                       help='any digit from 0 to 9')

    parser.add_argument('--option', type=str, default="a",
                       help='a-f, whichever exists')

    args = parser.parse_args()
    main(str(args.digit),str(args.option))
