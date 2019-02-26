import cv2
import numpy as np
import os
import pandas as pd
import re
import datetime

path = '~/Data/AI_City_Challenge/Tracking/'

def Get_gt():

    data_path = os.path.join(os.path.expanduser(path), 'train')
    save_path = os.path.join(os.path.expanduser('~/Data/AI_City_Challenge/ALL_gt_bbox/'), usage, 'gt_bbox_1_fps')

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for subset in os.listdir(data_path):
        subset_path = os.path.join(data_path, subset)

        for camera in os.listdir(subset_path):
            camera_num = int(camera[1:])

            video_file = os.path.join(subset_path, camera, 'vdo.avi')
            gt_file = os.path.join(subset_path, camera, 'gt', 'gt.txt')

            cap = cv2.VideoCapture(video_file)
            gt_bbox = pd.read_csv(gt_file, header=None)

            frames = []
            success = cap.isOpened()

            while(success):

                success, frame = cap.read()
                frames.append(frame)
                cv2.waitKey(0)

            cap.release()

            for index in range(len(gt_bbox)):

                frame_num = gt_bbox.iloc[index, 0]
                pid = gt_bbox.iloc[index, 1]
                bbox_left = int(gt_bbox.iloc[index, 2])
                bbox_top = int(gt_bbox.iloc[index, 3])
                bbox_width = int(gt_bbox.iloc[index, 4])
                bbox_height = int(gt_bbox.iloc[index, 5])

                bbox_bottom = bbox_top + bbox_height
                bbox_right = bbox_left + bbox_width

                frame = frames[frame_num - 1]
                gt_pic = frame[bbox_top:bbox_bottom, bbox_left:bbox_right]

                save_file = os.path.join(save_path, "{id}_c{cam}_f{fra}.jpg".format(id=pid, cam=camera_num, fra=frame_num))

                cv2.imwrite(save_file, gt_pic)
                cv2.waitKey(0)
                print(save_file)

        print(video_file, 'was completed!')


def Get_det():

    data_path = os.path.join(os.path.expanduser(path), 'test')
    save_path = os.path.join(os.path.expanduser('~/Data/AI_City_Challenge/ALL_det_bbox/'), 'det_bbox_YOLO3_test_all')

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for subset in os.listdir(data_path):
        subset_path = os.path.join(data_path, subset)

        for camera in os.listdir(subset_path):
            camera_num = int(camera[1:])

            video_file = os.path.join(subset_path, camera, 'vdo.avi')
            det_file = os.path.join(subset_path, camera, 'det', 'det_yolo3.txt')

            cap = cv2.VideoCapture(video_file)
            det_bbox = pd.read_csv(det_file, header=None)

            frames = []
            success = cap.isOpened()

            while(success):

                success, frame = cap.read()
                frames.append(frame)
                cv2.waitKey(0)

            cap.release()

            for index in range(len(det_bbox)):

                frame_num = det_bbox.iloc[index, 0]
                pid = det_bbox.iloc[index, 1]
                bbox_left = int(det_bbox.iloc[index, 2])
                bbox_top = int(det_bbox.iloc[index, 3])
                bbox_width = int(det_bbox.iloc[index, 4])
                bbox_height = int(det_bbox.iloc[index, 5])

                bbox_bottom = bbox_top + bbox_height
                bbox_right = bbox_left + bbox_width

                frame = frames[frame_num - 1]
                gt_pic = frame[bbox_top:bbox_bottom, bbox_left:bbox_right]

                save_file = os.path.join(save_path, "{id}_c{cam}_f{fra}.jpg".format(id=pid, cam=camera_num, fra=frame_num))

                cv2.imwrite(save_file, gt_pic)
                cv2.waitKey(0)
                print(save_file)

        print(video_file, 'was completed!')


def Get_det_val():

    data_path = os.path.join(os.path.expanduser(path), 'train')
    save_path = os.path.join(os.path.expanduser('~/Data/AI_City_Challenge/ALL_det_bbox/'), 'det_bbox_YOLO3_val')
    time_path = os.path.join(os.path.expanduser(path), 'cam_timestamp')
    counter = 0

    for subset in os.listdir(data_path):
        subset_path = os.path.join(data_path, subset)
        timestamp_path = os.path.join(time_path, '{}.txt'.format(subset))

        with open(timestamp_path) as t:
            temp = t.readlines()

        temp = [item.strip('\n') for item in temp]
        timestamp = {}

        for item in temp:
            timestamp[item[:4]] = float(item[5:])

        for camera in os.listdir(subset_path):

            fps = 10
            if subset == 'S03' and camera == 'c015':
                fps = 8

            camera_num = int(camera[1:])

            video_file = os.path.join(subset_path, camera, 'vdo.avi')
            det_file = os.path.join(subset_path, camera, 'det', 'det_yolo3.txt')

            cap = cv2.VideoCapture(video_file)
            det_bbox = pd.read_csv(det_file, header=None)

            frames = []
            success = cap.isOpened()

            while(success):

                success, frame = cap.read()
                frames.append(frame)
                cv2.waitKey(0)

            cap.release()

            for index in range(len(det_bbox)):

                frame_num = det_bbox.iloc[index, 0]

                if frame_num < 0.8 * len(frames):
                    continue

                pid = det_bbox.iloc[index, 1]
                bbox_left = int(det_bbox.iloc[index, 2])
                bbox_top = int(det_bbox.iloc[index, 3])
                bbox_width = int(det_bbox.iloc[index, 4])
                bbox_height = int(det_bbox.iloc[index, 5])

                bbox_bottom = bbox_top + bbox_height
                bbox_right = bbox_left + bbox_width

                frame = frames[frame_num - 1]
                gt_pic = frame[bbox_top:bbox_bottom, bbox_left:bbox_right]

                sync_frame = frame_num + timestamp[camera] * fps

                save_file = os.path.join(save_path, "{id}_c{cam}_f{fra}_n{count}.jpg".format(id=pid, cam=camera_num, fra=int(round(sync_frame)), count=counter))

                cv2.imwrite(save_file, gt_pic)
                cv2.waitKey(1)
                counter += 1
                # print(save_file)

            print('Start frame', int(round(timestamp[camera] * fps)))
            print('Final frame', int(round(sync_frame)))
            print('Camera', camera, 'was completed!')

        print('Subset', subset, 'was completed!')
        print(counter)

def Get_detections_val():

    data_path = os.path.join(os.path.expanduser(path), 'train')
    save_path = os.path.join(os.path.expanduser(path), 'val')
    time_path = os.path.join(os.path.expanduser(path), 'cam_timestamp')

    for subset in os.listdir(data_path):
        subset_path = os.path.join(data_path, subset)
        subset_save_path = os.path.join(save_path, subset)
        timestamp_path = os.path.join(time_path, '{}.txt'.format(subset))

        with open(timestamp_path) as t:
            temp = t.readlines()

        temp = [item.strip('\n') for item in temp]
        timestamp = {}

        for item in temp:
            timestamp[item[:4]] = float(item[5:])

        for camera in os.listdir(subset_path):

            det_save_path = os.path.join(subset_save_path, camera, 'det')

            if not os.path.exists(det_save_path):
                os.makedirs(det_save_path)

            det_file_path = os.path.join(det_save_path, 'det_yolo3.txt')

            fps = 10
            if subset == 'S03' and camera == 'c015':
                fps = 8

            camera_num = int(camera[1:])

            video_file = os.path.join(subset_path, camera, 'vdo.avi')
            det_file = os.path.join(subset_path, camera, 'det', 'det_yolo3.txt')

            cap = cv2.VideoCapture(video_file)
            det_bbox = pd.read_csv(det_file, header=None)

            frames = []
            success = cap.isOpened()

            while(success):

                success, frame = cap.read()
                frames.append(frame)
                cv2.waitKey(0)

            cap.release()

            for index in range(len(det_bbox)):

                frame_num = int(det_bbox.iloc[index, 0])

                if frame_num < 0.8 * len(frames):
                    continue

                sync_frame = frame_num + timestamp[camera] * fps

                det_bbox.iloc[index, 0] = int(round(sync_frame))

            det_bbox = det_bbox[det_bbox[0] >= (0.8 * len(frames))]

            det_bbox.to_csv(det_file_path, header=0, index=0)

            print('Start frame', int(round(timestamp[camera] * fps)))
            print('Final frame', int(round(sync_frame)))
            print('Camera', camera, 'was completed!')

        print('Subset', subset, 'was completed!')


if __name__ == '__main__':
    # date_str = '{}'.format(datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S'))
    # print(date_str)
    # Get_gt('train')
    # print('Train Folder Completed!')

    date_str = '{}'.format(datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S'))
    print(date_str)
    Get_det_val()
    print('Test Folder Completed!')
