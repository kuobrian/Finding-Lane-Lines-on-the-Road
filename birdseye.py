from utils import show_dotted_image
import cv2
import numpy as np


class BirdsEye:

    def __init__(self, source_p, dest_p):
        self.source_p = source_p
        self.dest_p = dest_p

        self.src_points = np.array(source_p, np.float32)
        self.dest_points = np.array(dest_p, np.float32)



        self.warp_matrix = cv2.getPerspectiveTransform(self.src_points, self.dest_points)
        self.inv_warp_matrix = cv2.getPerspectiveTransform(self.dest_points, self.src_points)

    def sky_view(self, image, show_dotted = False):

        shape = (image.shape[1], image.shape[0])
        warp_image = cv2.warpPerspective(image, self.warp_matrix, shape, flags = cv2.INTER_LINEAR)

        if show_dotted:
            show_dotted_image(warp_image, self.dest_p)

        return warp_image

    def project(self, image, sky_lane, left_fit, right_fit, color = (0, 255, 0)):

        z = np.zeros_like(sky_lane)
        sky_lane = np.dstack((z, z, z))

        kl, kr = left_fit, right_fit
        h = sky_lane.shape[0]
        ys = np.linspace(0, h - 1, h)
        lxs = kl[0] * (ys**2) + kl[1]* ys +  kl[2]
        rxs = kr[0] * (ys**2) + kr[1]* ys +  kr[2]

        pts_left = np.array([np.transpose(np.vstack([lxs, ys]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([rxs, ys])))])
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(sky_lane, np.int_(pts), color)

        shape = (sky_lane.shape[1], sky_lane.shape[0])
        ground_lane = cv2.warpPerspective(sky_lane, self.inv_warp_matrix, shape)

        result = cv2.addWeighted(image, 1, ground_lane, 0.3, 0)
        return result
