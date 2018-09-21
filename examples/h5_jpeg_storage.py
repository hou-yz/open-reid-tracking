import h5py
import numpy as np
import time
import glob
import os.path as osp


# fname = '/home/wangzd/Data/DukeMTMC/detections/openpose/features/features1.h5'
# fname = '/home/wangzd/houyz/open-reid-PCB_n_RPP/det_features/features1.h5'
# f = h5py.File(fname, 'r')
# # List all groups
# print("Keys: %s" % f.keys())
# group_keys = list(f.keys())
# # Get the data
# f_names = f[group_keys[1]]
# emb = f[group_keys[0]]


# fname = 'det_features/h5py_test.h5'
# with h5py.File(fname, 'w') as f:
#     f.create_dataset('augmentation_types', shape=(1,), data='original_resize')
#     rand_n = np.random.rand(10, 20)
#     dset = f.create_dataset('emb', data=rand_n, dtype=float)
#     pass

# test save jpg in h5 file


def save_h5_dataset(cam):
    root_dir = 'examples/data/det_dataset'
    fpaths = sorted(glob.glob(osp.join(root_dir, 'c%d_*.jpg' % cam)))

    dt = h5py.special_dtype(vlen=np.dtype('uint8'))
    f = h5py.File('h5_jpeg/DPM_camera%d.h5' % (cam), 'w')
    dset = f.create_dataset('jpeg_binary_data', (len(fpaths),), dtype=dt)
    for idx, fpath in enumerate(fpaths):
        fin = open(fpath, 'rb')
        binary_data = fin.read()
        # Save data string converted as a np array
        dset[idx] = np.fromstring(binary_data, dtype='uint8')
        pass

    f.close()
    pass


if __name__ == '__main__':
    tic = time.time()
    for cam in range(1, 9):
        save_h5_dataset(cam)
    toc = time.time() - tic
    print('*************** write file takes time: {:^10.2f} *********************\n'.format(toc))

    # restore image
    # from PIL import Image
    # import io
    # img = Image.open(io.BytesIO(dset[0]))  # This took ~30ms for a hi-res 3-channel image (~2000x2000)
    # img.show()
