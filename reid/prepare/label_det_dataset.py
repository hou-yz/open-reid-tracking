import os
import cv2
import shutil
import datetime
import pandas as pd


def main(det_time='train', fps=10, IoUthreshold=0.3, merge=False):

    output_file = './Failed_Detection_{}.txt'.format(IoUthreshold)

    if os.path.exists(output_file):
        os.remove(output_file)

    if merge == True:
        save_dir = os.path.join(os.path.expanduser('~/Data/AIC19/Temp'), det_time, 'det_bbox_{}_fps'.format(fps))
    else:
        save_dir = os.path.join(os.path.expanduser('~/Data/AIC19/ALL_det_IoU_bbox'), det_time, 'det_bbox_{}_fps'.format(fps))
    data_dir = os.path.expanduser('~/Data/AIC19/train')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)

    if det_time == 'train':
        scenes = ['S03', 'S04']
    elif det_time == 'trainval':
        scenes = ['S01', 'S03', 'S04']
    elif det_time == 'val':
        scenes = ['S01']

    missingcounter = 0

    # loop for subsets
    for scene in scenes:
        scene_dir = os.path.join(data_dir, scene)

        # loop for cameras
        for camera in os.listdir(scene_dir):
            gt_file_path = os.path.join(scene_dir, camera, 'gt', 'gt.txt')
            det_file_path = os.path.join(scene_dir, camera, 'det', 'det_yolo3.txt')
            video_path = os.path.join(scene_dir, camera, 'vdo.avi')

            iCam = int(camera[1:])

            video_reader = cv2.VideoCapture(video_path)
            frame_pics = []
            success = video_reader.isOpened()
            while (success):
                success, frame_pic = video_reader.read()
                frame_pics.append(frame_pic)
                cv2.waitKey(0)
            video_reader.release()

            gt_file = pd.read_csv(gt_file_path, header=None)
            det_file = pd.read_csv(det_file_path, header=None)

            for gt_index in range(len(gt_file)):

                frame = gt_file.iloc[gt_index, 0]
                pid = gt_file.iloc[gt_index, 1]

                det_windows = det_file[det_file[0] == frame]
                if len(det_windows) > 0:
                    gt_loc = gt_file.iloc[gt_index, 2:6].tolist()
                    IoUs = []

                    for det_index in range(len(det_windows)):
                        det_loc = det_windows.iloc[det_index, 2:6].tolist()
                        IoU = compute_iou(gt_loc, det_loc)
                        IoUs.append(IoU)
                    if max(IoUs) < IoUthreshold:
                        with open(output_file, 'a') as o:
                            o.write(str(camera) + ' ' + str(frame) + ' ' + str(pid) + '\n')
                        print(camera, frame, pid)
                        pid = -1
                        missingcounter =  missingcounter + 1
                    index_closest = IoUs.index(max(IoUs))
                    det_closest = det_windows.iloc[index_closest, :].tolist()

                    bbox_left = int(det_closest[2])
                    bbox_top = int(det_closest[3])
                    bbox_width = int(det_closest[4])
                    bbox_height = int(det_closest[5])

                    bbox_bottom = bbox_top + bbox_height
                    bbox_right = bbox_left + bbox_width

                    frame_pic = frame_pics[frame - 1]
                    bbox_pic = frame_pic[bbox_top:bbox_bottom, bbox_left:bbox_right]

                    if merge == True:
                        save_file = os.path.join(save_dir, "{:04d}_{}_c{:02d}_f{:05d}_det.jpg".format(pid, scene.lower(), iCam, frame))
                    else:
                        save_file = os.path.join(save_dir, "{:04d}_{}_c{:02d}_f{:05d}.jpg".format(pid, scene.lower(), iCam, frame))

                    cv2.imwrite(save_file, bbox_pic)
                    cv2.waitKey(0)

                else:
                    print(camera, frame, 'has no detection in the same frame')
                    with open(output_file, 'a') as o:
                        o.write(str(camera) + ' ' + str(frame) + ' ' + 'has no detection in the same frame')

            print(camera, 'is completed')
        print(scene, 'is completed')
    print(missingcounter)



if __name__ == '__main__':
    print('{}'.format(datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')))
    # main(det_time='train', merge=True)
    main(det_time='val', merge=True)
    # main(det_time='trainval', merge=True)
    print('{}'.format(datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')))
    print('Job Completed!')

