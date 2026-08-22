from __future__ import annotations

import cv2
import numpy as np


class Image:
    def __init__(self, blob=None, **_kwargs):
        if blob is None:
            self.image = np.zeros((1, 1, 3), dtype=np.uint8)
        else:
            data = np.frombuffer(blob, dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError("Unable to decode image blob.")
            self.image = image
        self.wand = self

    def make_blob(self):
        ok, encoded = cv2.imencode(".png", self.image)
        if not ok:
            raise ValueError("Unable to encode image blob.")
        return encoded.tobytes()

