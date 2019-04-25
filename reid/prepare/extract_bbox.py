import cv2
import numpy as np
import os
import os.path as osp
import pandas as pd
import datetime
import psutil

path = '~/Data/AIC19/'
og_fps = 10


def get_bbox(type='gt', det_time='train', fps=5, det_bbox_enlarge=0.0, det_type='ssd'):
    # type = ['gt','det','labeled']
    data_path = osp.join(osp.expanduser(path), 'test' if det_time == 'test' else 'train')
    save_path = osp.join(osp.expanduser('~/Data/AIC19/ALL_{}_bbox/'.format(type)), det_time)

    if type == 'gt' or type == 'labeled':
        save_path = osp.join(save_path, 'gt_bbox_{}_fps'.format(fps))
        fps_pooling = int(og_fps / fps)  # use minimal number of gt's to train ide model
    else:
        save_path = osp.join(save_path, det_type)
        if det_bbox_enlarge:
            save_path += '_enlarge{}'.format(det_bbox_enlarge)

    if not osp.exists(save_path):  # mkdir
        if not osp.exists(osp.dirname(save_path)):
            if not osp.exists(osp.dirname(osp.dirname(save_path))):
                # if not osp.exists(osp.dirname(osp.dirname(osp.dirname(save_path)))):
                #     os.mkdir(osp.dirname(osp.dirname(osp.dirname(save_path))))
                os.mkdir(osp.dirname(osp.dirname(save_path)))
            os.mkdir(osp.dirname(save_path))
        os.mkdir(save_path)

    # scene selection for train/val
    if det_time == 'train':
        scenes = ['S03', 'S04']
    elif det_time == 'trainval':
        scenes = ['S01', 'S03', 'S04']
    elif det_time == 'val':
        scenes = ['S01']
    else:  # test
        scenes = os.listdir(data_path)

    for scene_dir in scenes:
        scene_path = osp.join(data_path, scene_dir)
        for camera_dir in os.listdir(scene_path):
            iCam = int(camera_dir[1:])
            # get bboxs
            if type == 'gt':
                bbox_filename = osp.join(scene_path, camera_dir, 'gt', 'gt.txt')
            elif type == 'labeled':
                bbox_filename = osp.join(scene_path, camera_dir, 'det',
                                         'det_{}_labeled.txt'.format('ssd512' if det_type == 'ssd' else 'yolo3'))
            else:  # det
                bbox_filename = osp.join(scene_path, camera_dir, 'det',
                                         'det_{}.txt'.format('ssd512' if det_type == 'ssd' else 'yolo3'))
            bboxs = np.array(pd.read_csv(bbox_filename, header=None))
            if type == 'gt' or type == 'labeled':
                bboxs = bboxs[np.where(bboxs[:, 0] % fps_pooling == 0)[0], :]

            # get frame_pics
            video_file = osp.join(scene_path, camera_dir, 'vdo.avi')
            video_reader = cv2.VideoCapture(video_file)
            # get vcap property
            width = video_reader.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
            height = video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float

            # bboxs
            bbox_left = bboxs[:, 2]
            bbox_top = bboxs[:, 3]
            bbox_width = bboxs[:, 4]
            bbox_height = bboxs[:, 5]
            bbox_bottom = bbox_top + bbox_height
            bbox_right = bbox_left + bbox_width

            # enlarge
            bbox_top = np.maximum(bbox_top - det_bbox_enlarge * bbox_height, 0)
            bbox_bottom = np.minimum(bbox_bottom + det_bbox_enlarge * bbox_height, height - 1)
            bbox_left = np.maximum(bbox_left - det_bbox_enlarge * bbox_width, 0)
            bbox_right = np.minimum(bbox_right + det_bbox_enlarge * bbox_width, width - 1)
            bboxs[:, 2:6] = np.stack((bbox_top, bbox_bottom, bbox_left, bbox_right), axis=1)

            # frame_pics = []
            frame_num = 0
            success = video_reader.isOpened()
            printed_img_count = 0
            while (success):
                assert psutil.virtual_memory().percent < 95, "reading video will be killed!!!!!!"

                success, frame_pic = video_reader.read()
                frame_num = frame_num + 1
                bboxs_in_frame = bboxs[bboxs[:, 0] == frame_num, :]

                for index in range(bboxs_in_frame.shape[0]):
                    frame = int(bboxs_in_frame[index, 0])
                    pid = int(bboxs_in_frame[index, 1])
                    bbox_top = int(bboxs_in_frame[index, 2])
                    bbox_bottom = int(bboxs_in_frame[index, 3])
                    bbox_left = int(bboxs_in_frame[index, 4])
                    bbox_right = int(bboxs_in_frame[index, 5])

                    bbox_pic = frame_pic[bbox_top:bbox_bottom, bbox_left:bbox_right]
                    if bbox_pic.size == 0:
                        continue

                    if type == 'gt' or type == 'labeled':
                        save_file = osp.join(save_path, "{:04d}_c{:02d}_f{:05d}.jpg".format(pid, iCam, frame))
                    else:
                        save_file = osp.join(save_path, 'c{:02d}_f{:05d}_{:03d}.jpg'.format(iCam, frame, index))

                    cv2.imwrite(save_file, bbox_pic)
                    cv2.waitKey(0)
                    printed_img_count += 1

                cv2.waitKey(0)
            video_reader.release()
            assert printed_img_count == bboxs.shape[0]

            print(video_file, 'completed!')
        print(scene_dir, 'completed!')
    print(save_path, 'completed!')


if __name__ == '__main__':
    print('{}'.format(datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')))
    get_bbox(type='gt', fps=1, det_time='trainval')
    # get_bbox(fps=1)
    # get_bbox(det_time='val', fps=1)
    # get_bbox(type='det', det_time='val', det_bbox_enlarge=0, det_type='ssd')
    # get_bbox(type='det', det_time='trainval', det_bbox_enlarge=0,det_type='ssd')
    # get_bbox(type='det', det_time='test', det_bbox_enlarge=0, det_type='ssd')
    print('{}'.format(datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')))
    print('Job Completed!')
