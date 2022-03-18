#!/usr/bin/env python3

import cv2
import queue


def isInRoad(img, point):
    tot = 0
    count = 0
    for i in range(-1, 2):
        for j in range(-1, 2):
            npoint = (point[0] + i, point[1] + j)
            if (npoint[0] < 0 or npoint[1] < 0 or
                    npoint[0] >= img.shape[0] or npoint[1] >= img.shape[1]):
                continue
            count += 1
            tot += img[npoint[0]][npoint[1]]
    tot /= count
    return tot >= 45 and tot <= 200


def getRoadBfs(img, row, col, flags, lanes):
    lane = []
    q = queue.Queue()
    q.put((row, col))
    while not q.empty():
        point = q.get()
        if (point[0] < 0 or point[1] < 0 or point[0] >= img.shape[0] or
                point[1] >= img.shape[1] or point in flags):
            continue
        flags.add(point)
        if isInRoad(img, point):
            lane.append(point)
            for i in range(-1, 2, 2):
                for j in range(-1, 2, 2):
                    q.put((point[0] + i, point[1] + j))
    if len(lane) > 0:
        lanes.append(lane)


def getRoad(img):
    lanes = []
    flags = set()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # for i in range(0, binary.shape[0]):
    #     getRoadBfs(binary, i, 0, flags, lanes)
    #     getRoadBfs(binary, i, binary.shape[1] - 1, flags, lanes)
    for j in range(binary.shape[1] // 3, binary.shape[1] // 3 * 2):
        # getRoadBfs(binary, 0, j, flags, lanes)
        getRoadBfs(binary, binary.shape[0] - 1, j, flags, lanes)

    return lanes


for i in range(0, 56):
    img = cv2.imread("kitti/static/%06d.png" % i)
    lanes = getRoad(img)

    for lane in lanes:
        for point in lane:
            image = cv2.circle(img, point[::-1], radius=1,
                               color=(0, 255, 0), thickness=-1)

    cv2.imwrite("output/static/%d.jpg" % i, img)
