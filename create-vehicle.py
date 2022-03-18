#!/usr/bin/env python3

import cv2
import numpy as np


v1 = np.asarray((200, 200))
v2 = np.asarray((400, 400))
v3 = np.asarray((150, 10))
v4 = np.asarray((0, 0))
wh = np.asarray((100, 50))
for i in range(1, 10):
    img = np.zeros((720, 1280, 3), np.uint8)
    img = cv2.rectangle(img, v1, v1+wh, (255, 255, 255), -1)
    img = cv2.rectangle(img, v2, v2+wh, (255, 255, 255), -1)
    img = cv2.rectangle(img, v3, v3+wh, (255, 255, 255), -1)
    img = cv2.rectangle(img, v4, v4+wh, (255, 255, 255), -1)
    v1 += (20, 0)
    v2 += (10, 20)
    v3 = np.int0((150 + i * 20, (150 + i * 20) ** 0.5))
    v4 = (i * 2, (i * 2) ** 2)
    cv2.imwrite('test/vehicle/%d.png' % i, img)

# cv2.imshow('img', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
