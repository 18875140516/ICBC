import os
import datetime
import pyinotify
import logging
import base64
import cv2
pos = 0
filename = './test.jpg'
#USE_IMSHOW = True
#use pyinotify to send new image to frontend by websocket client
class RegisterEventHandler:
    def __init__(self, filename='/tmp/test.jpg', wsclient=None):
        class ImageEventHandler(pyinotify.ProcessEvent):
            def my_init(self, **kargs):
                print(kargs)
                print(type(kargs['filename']), type(kargs['wsclient']))
                self.filename = kargs['filename']
                self.wsclient = kargs['wsclient']

            def process_IN_ACCESS(self, event):
                pass
            def process_IN_OPEN(self, event):
                pass
            def process_IN_CLOSE_WRITE(self, event):
#                if USE_IMSHOW:
#                    img = cv2.imread(self.filename)
#                    cv2.imshow('img', img)
#                    cv2.waitKey(30)
                if self.wsclient is not None:
                    with open(self.filename, 'rb') as r:
                        img_b64 = base64.b64encode(r.read())

                    self.wsclient.send(img_b64)    
                pass
        self.__create_file(filename)
        wm = pyinotify.WatchManager()
        #wm.add_watch('test.txt', pyinotify.ALL_EVENTS, rec=True)

        wm.add_watch(filename, pyinotify.ALL_EVENTS, rec=True)
        handler = ImageEventHandler(wsclient=wsclient, filename=filename)
 
        # notifier
        notifier = pyinotify.Notifier(wm, handler)
        notifier.loop()
        
        pass
    def __create_file(self, filename='/tmp/test.jpg'):
        path = filename[0:filename.rfind('/')]
        if not os.path.isdir(path):
            os.makedirs(path)
        if not os.path.isfile(filename):
            fd = open(filename, mode='w', encoding='utf-8')
            fd.close()
        else:
            pass

#-----------------------test ----------------------
#def main():
#    #输出前面的log
#    # printlog()
#    # watch manasger
#    handler = RegisterEventHandler(filename='/tmp/test.jpg')
# 
#if __name__ == '__main__':
#    main()
