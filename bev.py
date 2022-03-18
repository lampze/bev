#!/usr/bin/env python3

from os import walk
import json
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False


def realCoordinate(c, c0, theta):
    """获取车辆 _真实_ 地址."""
    c = np.asarray(c)
    c0 = np.asarray(c0)
    # 计算相对于图片中心的位置并把单位转换成米
    c0 = (c0 - (360, 640)) * [27 / 720, 48 / 1280]

    theta = -theta
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    return c + np.dot(rot, c0)


ids = []


def findVid(c):
    """追踪车辆ID."""
    for i in range(len(ids)):
        if np.linalg.norm(c - ids[i]) < 80:
            ids[i] = c
            return i
    ids.append(c)
    return len(ids) - 1


def getW_h(box):
    """计算车辆的长宽比."""
    w_h = []
    for j in range(1, 4):
        w_h.append(np.linalg.norm(box[j]-box[j - 1]))
        w_h.sort(reverse=True)
    return w_h[0] / w_h[2]


def getVehicle(img, location, time):
    """获取一张图内每个车辆的基础信息."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    vehicles = []

    for contour in contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.asarray(box)
        # 计算车辆中心
        center = (sum(box) / 4)

        # 获取中心真实坐标
        realC = realCoordinate(
            (location['x'], location['y']),
            # (0, 0),
            center,
            location['heading']
            # 0
        )

        # 计算长宽比
        w_h = getW_h(box)

        vid = findVid(center)

        vehicles.append({
            'time': time,
            'id': vid,
            'x': realC[0],
            'y': realC[1],
            'z': location['z'],
            'h': 0,
            'vx': 0,
            'vy': 0,
            'ax': 0,
            'ay': 0,
            'v': 0,
            'a': 0,
            'w_h': w_h,
        })
    return vehicles


def calcVehicleDetail(vehicles):
    """计算速度加速度之类的详细数据."""
    vehicles = pd.DataFrame(vehicles)
    for i in range(len(ids)):
        idata = vehicles.loc[vehicles.id == i].copy()
        for j in range(1, len(idata)):
            jdata = idata.iloc[j].copy()
            t = (jdata.time - idata.iloc[j - 1].time)
            # 计算位置的增量，也就是速度
            c = np.asarray((jdata.x, jdata.y))
            bc = np.asarray((idata.iloc[j - 1].x, idata.iloc[j - 1].y))
            d = (c - bc) / t
            jdata.vx = d[0]
            jdata.vy = d[1]
            # 计算头指向
            jdata.h = np.arctan2(d[1], d[0])
            # 使用上一帧的速度减当前速度得到加速度
            jdata.ax = (jdata.vx - idata.iloc[j - 1].vx) / t
            jdata.ay = (jdata.vy - idata.iloc[j - 1].vy) / t
            jdata.v = np.linalg.norm(d)
            jdata.a = np.linalg.norm((jdata.ax, jdata.ay))
            idata.iloc[j] = jdata
        vehicles.loc[vehicles.id == i] = idata
    return vehicles


vehicles = []
locations = dict()

prefix_path = 'test/'

# 获取location数据
for (dirpath, dirnames, filenames) in walk(prefix_path + "location/"):
    for filename in filenames:
        with open(dirpath + filename) as f:
            locations[int(filename.split('.')[0])] = json.load(f)
    break

# 获取时间戳
times = pd.read_csv(prefix_path + "time.txt",
                    header=None, sep='\t', index_col=1)


# 获取每张图片的车辆数据
for (dirpath, dirnames, filenames) in walk(prefix_path + "vehicle/"):
    for filename in filenames:
        i = int(filename.split('.')[0])
        vehicleImg = cv2.imread(dirpath + filename)
        vehicles += getVehicle(vehicleImg, locations[i], times.loc[i, 0])
    vehicles = calcVehicleDetail(vehicles)


# 头指向示意图
ax = vehicles.plot(kind='scatter', x='x', y='y', s=(vehicles.id + 1)
                   * 50, title='总体头指向示意图')
ax.quiver(vehicles.x, vehicles.y, vehicles.vx, vehicles.vy,
          color='gray', width=0.003, alpha=0.5)
plt.savefig(prefix_path + 'output/position.png')

# 单个目标绘图
for i in range(len(ids)):
    # 速度大小变化趋势
    vehicles.loc[vehicles.id == i, ['v', 'a', 'time']].iloc[1:].plot(
        x='time', title='目标%d的速度与加速度折线图' % i)
    plt.savefig(prefix_path + 'output/av-%d.png' % i)

    # 位置
    idata = vehicles.loc[vehicles.id == i]
    ax = idata.plot(
        kind='scatter', x='x', y='y', s=100, title='目标%d的头指向示意图' % i)
    ax.quiver(idata.x, idata.y, idata.vx, idata.vy,
              color='gray', width=0.003, alpha=0.5)
    plt.savefig(prefix_path + 'output/ph-%d.png' % i)
