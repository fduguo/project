from __future__ import annotations

import math

import cv2
import numpy as np


class _MotionBlurFunction:
    argtypes = None

    def __call__(self, wand, radius, sigma, angle):
        image = getattr(wand, "image", None)
        if image is None:
            return True

        radius = max(float(radius), 1.0)
        kernel_size = max(3, int(round(radius)))
        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        kernel[kernel_size // 2, :] = 1.0
        matrix = cv2.getRotationMatrix2D((kernel_size / 2.0 - 0.5, kernel_size / 2.0 - 0.5), float(angle), 1.0)
        kernel = cv2.warpAffine(kernel, matrix, (kernel_size, kernel_size))
        total = float(kernel.sum())
        if math.isfinite(total) and total > 0:
            kernel /= total

        wand.image = cv2.filter2D(image, -1, kernel)
        return True


class _Library:
    MagickMotionBlurImage = _MotionBlurFunction()


library = _Library()
