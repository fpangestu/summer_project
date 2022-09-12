# import cv2

# cam=cv2.VideoCapture(0)
# ccfr2=cv2.CascadeClassifier('Opencv-master/palm.xml')
# while True:
#     retval,image=cam.read()
#     grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#     palm=ccfr2.detectMultiScale(grey,scaleFactor=1.05,minNeighbors=3)
#     for x,y,w,h in palm:
#         image=cv2.rectangle(image,(x,y),(x+w,y+h),(256,256,256),2)
    
#     cv2.imshow("Window",image)
#     if cv2.waitKey(1) & 0xFF==ord('q'):
#         cv2.destroyAllWindows()
#         break
# del(cam)


import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0) #Камера
hands = mp.solutions.hands.Hands(max_num_hands=2) #Объект ИИ для определения ладони
draw = mp.solutions.drawing_utils #Для рисование ладони

while True:
    #Закрытие окна
    if cv2.waitKey(1) & 0xFF == 27:
        break

    success, image = cap.read() #Считываем изображение с камеры
    # image = cv2.flip(image, -1) #Отражаем изображение для корекктной картинки
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Конвертируем в rgb
    results = hands.process(imageRGB) #Работа mediapipe

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

            draw.draw_landmarks(image, handLms, mp.solutions.hands.HAND_CONNECTIONS) #Рисуем ладонь
            
            # for x,y,w,h in handLms:
            #     image=cv2.rectangle(image,(x,y),(x+w,y+h),(256,256,256),2)

    cv2.imshow("Hand", image) #Отображаем картинку