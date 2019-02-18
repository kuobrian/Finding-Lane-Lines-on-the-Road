import glob
import numpy




test_img_list = glob.glob('./test_images/*.jpg')
src_p = [(117, 1530), (555, 1116), (1926, 1530), (1538, 1116)]
dst_p = [(117, 1530), (117, 777), (1926, 1530), (1926, 777)]
p = {'sat_thresh': 120, 'light_thresh': 40, 'light_thresh_agr': 150,
  'grad_thresh': (0.7, 1.4), 'mag_thresh': 40, 'x_thresh': 20}