import cv2
import argparse
import json
points = []

def onMouseCallback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x,y))
    # elif event == cv2.EVENT_LBUTTONDBLCLK:
    #     points = points[:len(points)-1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='image which need to be calibrate')
    parser.add_argument('--output', default='../config/queue.json', type=str, help='output file to save the calibration')
    args = parser.parse_args()
    assert args.input != None
    assert args.output.split('.')[-1] == 'json'
    img = None
    if args.input.split('.')[-1] == 'jpg' or args.input.split('.')[-1] == 'png':
        img = cv2.imread(args.input)
    elif args.input.split('.')[-1] == 'avi' or args.input.split('.')[-1] == 'mp4':
        cap = cv2.VideoCapture(args.input)
        ret, img = cap.read()
        cap.release()
    else:
        exit(-1)
    cv2.namedWindow('calib')
    cv2.setMouseCallback('calib', onMouseCallback)
    while 1:
        cp = img.copy()
        for i in range(len(points)-1):
            cv2.line(cp, points[i], points[i+1], (255,255,255), thickness=1)
        cv2.imshow('calib', cp)
        q = cv2.waitKey(30)
        if q == ord('q'):
            break
        elif q == ord('x'):
            points = points[:len(points) - 1]


    with open(args.output, 'r+') as w:
        root = json.load(w)
        root['area'] = []
        for p in points:
            root['area'].append(p)


    with open(args.output, 'w')  as w:

        json.dump(root, w)
        # root['area'] = points
        # json.dump(root, w)
        # print(root)
