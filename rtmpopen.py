import cv2
url = 'http://127.0.0.1:8081/hls/test2.m3u8'
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