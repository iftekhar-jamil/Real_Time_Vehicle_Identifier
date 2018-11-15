import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import cv2
from datetime import datetime


classifier = load_model("car_bike_new.h5")
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (80,150)
fontScale = 2
fontColor = (255,0,0)
lineType = 5
car = 0
bike = 0
cap = cv2.VideoCapture('F:\Project\Convolutional_Neural_Networks\dataset\Toy_car.mov')
#cap = cv2.VideoCapture('F:\Project\Convolutional_Neural_Networks\dataset\motorcycle racing-H264 75.mov')
fgbg = cv2.createBackgroundSubtractorMOG2()
while(1):
    ret, frame = cap.read()
    if(ret==True):
        frame = cv2.resize(frame, (800, 800))
        crop_img = frame[200:600, 200:600]
        fgmask = fgbg.apply(crop_img)
        
        contours = cv2.findContours(fgmask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
            
        if(len(contours) > 0):
            contour = max(contours, key = cv2.contourArea)
            x,y,w,h = cv2.boundingRect(contour)
            cv2.rectangle(crop_img,(x,y),(x+w,y+h),(0,255,0),2)
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(crop_img,[box],0,(0,0,255),2)
    #        print(cv2.contourArea(contour))
            if(cv2.contourArea(contour) > 45000):
                cv2.imwrite('temp.jpg',cv2.resize(crop_img, (128, 128)))
                img_path = 'temp.jpg'    
    
                test_image = image.load_img(img_path, target_size = (128, 128))
                test_image = image.img_to_array(test_image)
                test_image = np.expand_dims(test_image, axis = 0)
                result = classifier.predict(test_image)
                if result[0][0] >= 0.5:
                    prediction = 'bike'
                    bike+=1
                else:
                    prediction = 'car'
                    car+=1
                if(car>bike):
                    prediction= "car"
                else:
                    prediction="bike"
                print("{},  {}".format(prediction,cv2.contourArea(contour)))
                
        cv2.putText(crop_img, str(prediction), bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
     
        cv2.imshow('fgmask',frame)
        cv2.imshow('frame',fgmask)
        cv2.imshow('crop',crop_img)
    
        
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    else: 
        cv2.destroyAllWindows()
        break    
        
        

cap.release()
cv2.destroyAllWindows()