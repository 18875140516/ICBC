import numpy as np
import cv2
import matplotlib.pyplot as plt
class Gallery:
    def __init__(self, store_image=True, max_age=30):
        self.threshold = 0.2
        self.threshold2 = 0.3
        self.max_age = max_age
        self.store = dict()
        self.name_file = dict()
        self.store_image = store_image
        if store_image:
            self.file_image = dict()
        pass

    def __next__(self):
        pass

    def set_max_age(self, max_age):
        self.max_age = max_age

    '''
    the id of a person -> filename
    '''

    def name2file(self, name=None):
        if name is None:
            assert 'please input the image id'
            return None
        if name not in self.name_file.keys():
            assert 'it is a not exist key'
            return None
        return self.name_file[name]

    '''
    filename -> feature
    '''

    def file2feature(self, file=None):
        if file is None:
            assert 'file is None'
            return None
        if file not in self.store.keys():
            assert 'file not in store'
            return None
        return self.store[file]


    '''
    filename  -> image
    '''

    def file2image(self, file):
        if file  is None:
            assert 'file is None'
            return None
        if file not in self.file_image.keys():
            assert 'file not exist '
            return None
        return self.file_image[file]

    '''
    set feature by name and frame
    '''
    def set_feature(self, name, feature, frame=-1, img=None):
        #当与name对应的特征超过半数都相似，那么将该特征放入gallery中

        features = self.get_features_by_name(name)
        if features is not None:
            features = np.array(features)
            dist = np.dot(features, feature[np.newaxis, :].T)
            dist = 1 - dist
            dist = dist < self.threshold2
            cnt = np.sum(dist)
            if cnt < dist.size//2:
                return

        self.name_file.setdefault(name, [])
        if self.max_age <= len(self.name_file[name]):
            old_file = self.name_file[name][0]
            self.name_file[name] = self.name_file[name][1:]
            self.store.pop(old_file)
            if self.store_image:
                self.file_image.pop(old_file)



        file = 'id{:04d}_{:04d}'.format(name, frame)
        self.name_file[name].append(file)
        self.store[file] = feature
        if img is not None:
            self.file_image[file] = img
        pass

    '''
    get all the filenames
    '''

    def get_files(self):
        return self.store.keys()

    '''
    get all the features
    '''

    def get_features(self):
        return self.store.values()


    def get_features_by_name(self, name):
        files = self.name2file(name)
        if files == None:
            return None
        features = []
        for file in files:
            features.append(self.file2feature(file))
        return features

    '''
    get all the ids of person
    '''

    def get_names(self):
        return self.name_file.keys()

    def show_name(self, name):
        files = self.name2file(name)
        for i in files:
            print('file = ',i)
            img = self.file2image(i)

            cv2.imshow('img', img)
            cv2.waitKey(0)
        pass

    def show_file(self, file):
        img = self.file_image[file]
        cv2.imshow(file,img)
        cv2.waitKey(0)
        pass

    def show_name2(self, name):
        files = self.name2file(name)
        l = len(files)
        l = l + 1 if l %2 == 1 else l
        for i, file in enumerate(files):
            plt.subplot(2, l//2, i+1)
            img = self.file2image(file)
            plt.imshow(img[...,::-1])
            plt.axis('off')
        plt.show()

    '''
    input: feature, shape = [1, 2048], 
            [image, ], image file of detected person
    return: ID, the closest feature ID, if none , return -1
            dist, correspond distance between detection and gallery 
    '''
    def find_one_by_det(self, det, frame_id=None, image=None):
        names = self.get_names()
        temp_fea = np.array(det.feature)
        temp_fea = temp_fea[np.newaxis, :]#shape = [1, 2048]
        name_dist = []
        for name in names:
            features = self.get_features_by_name(name)#list[feature]
            features = np.array(features)

            # print('temp fea shape',temp_fea.shape)
            # print('features', features)
            # print('tempfeature', temp_fea)
            # dist = cdist(features, temp_fea, 'euclidean')#shape = [None, 1]
            dist = np.dot(features, temp_fea.T)
            print('name=', name)
            print('dist',dist)
            sp = dist.shape
            dist.sort(axis=0)
            name_dist.append((name, dist[:(sp[0]+1)//2, ...].mean(axis=0)[0]))#

        name_dist.sort(key=lambda x:   x[1])
        if name_dist[0][1] < self.threshold:#similar
            self.set_feature(name_dist[0][0], temp_fea, frame=frame_id, img=image)
            return name_dist[0]
        else:
            return None, -1


    def find_all_by_det(self, detections, frame_id=None, images=None):
        ret = []
        if images is not None:
            for det, img in zip(detections, images):
                ret.append(self.find_one_by_det(det, frame_id=frame_id, image=img))
        else:
            for det in detections:
                ret.append(self.find_one_by_det(det, frame_id=frame_id))
        return ret

    def find_one_by_feature(self, feature, used_id=None):# shape = [2048]
        # print(feature.shape)
        names = self.get_names()
        if used_id is not None:
            names = [name for name in names if name not in used_id]
        temp_fea = np.array(feature)
        temp_fea = temp_fea[np.newaxis, :]  # shape = [1, 2048]
        # print(temp_fea.shape)
        name_dist = []
        for name in names:
            features = self.get_features_by_name(name)  # list[feature]
            features = np.array(features)

            # print('temp fea shape',temp_fea.shape)
            # print('features', features)
            # print('tempfeature', temp_fea)
            # dist = cdist(features, temp_fea, 'euclidean')  # shape = [None, 1]
            # print('name=', name)
            dist = np.dot(features, temp_fea.T)

            # print('dist', dist)
            sp = dist.shape
            dist = np.array(sorted(dist,  reverse=True))
            name_dist.append((name, dist[:(sp[0] + 1) // 2, ...].mean(axis=0)[0]))  #
        name_dist = [(name, 1 - i) for name, i in name_dist]
        name_dist.sort(key=lambda x: x[1])
        if len(name_dist) > 0 and name_dist[0][1] < self.threshold:  # not similar
            # self.set_feature(name_dist[0][0], temp_fea[0])
            return name_dist[0]
        else:
            return None, -1

g_gallery = Gallery(store_image=True)