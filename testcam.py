from keras.models import load_model
from keras.optimizers import Adam
import cv2
import numpy as np

x_bound_start = 50
y_bound_start = 150
x_bound_end = 250
y_bound_end = 350
thickness = 2

model = load_model('models/model.h5')

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

cap = cv2.VideoCapture(0)
 
while(cap.isOpened()):
    ret, frame = cap.read()
    cv2.rectangle(frame,(x_bound_start,y_bound_start),(x_bound_end,y_bound_end),(0,255,0),thickness) 
    cv2.imshow('frame',frame)

    keyPress = cv2.waitKey(1) & 0xFF

    if keyPress == ord('q'):
        break
    
    if keyPress == ord('c'):
        img = frame[y_bound_start + thickness:y_bound_end - thickness,x_bound_start + thickness:x_bound_end - thickness]
        cv2.imshow('extracted', img)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray,(28,28))
        final_img = np.reshape(resize,[1,784])
        
        classes = model.predict_classes(final_img)
        print('Predicted digit:',classes[0])

cap.release()
cv2.destroyAllWindows()

