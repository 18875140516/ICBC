import cv2
url = 'rtmp://211.67.20.74:1935/myapp'
cap = cv2.VideoCapture(url)
if cap.isOpened():
    print('open successfully')
else:
    print('error')
while True:
    ret, img = cap.read()
    if ret == False:
        print('read error')
        break
    else:
        cv2.imshow('img', img)
        cv2.waitKey(30)