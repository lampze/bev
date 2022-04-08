#!/usr/bin/env python3

from os import walk, path, mkdir
import pandas as pd
import matplotlib.pyplot as plt
from evaDataset import calcVehicleDetail

plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

datasetPath = "2022-4-8-17-13/"
outputPath = path.join(datasetPath, 'output')
if not path.exists(outputPath):
    mkdir(outputPath)

vehicles = pd.read_csv(path.join(datasetPath, 'v.txt'), names=[
                       'time', 'id', 'x', 'y'], sep='\s+')

vehicles['h'] = 0
vehicles['vx'] = 0
vehicles['vy'] = 0
vehicles['ax'] = 0
vehicles['ay'] = 0
vehicles['v'] = 0
vehicles['a'] = 0


ids = set(vehicles['id'])

vehicles = calcVehicleDetail(vehicles, ids)


# 头指向示意图
ax = vehicles.plot(kind='scatter', x='x', y='y',
                   s=10, title='总体头指向示意图', alpha=0.5)
ax.quiver(vehicles.x, vehicles.y, vehicles.vx, vehicles.vy,
          color='orange', width=0.01, alpha=0.6)
ax.set_aspect('equal', adjustable='datalim')
plt.savefig(path.join(datasetPath, 'output/position.png'), dpi=300)
plt.close('all')

# 单个目标绘图
for i in ids:
    idata = vehicles.loc[vehicles.id == i]
    if len(idata) < 10:
        continue
    # 速度大小变化趋势
    vehicles.loc[vehicles.id == i, ['v', 'a', 'time']].iloc[2:].plot(
        x='time', title='目标%s的速度与加速度折线图' % i,)
    plt.savefig(path.join(datasetPath, 'output/av-%s.png' % i), dpi=300)
    plt.close('all')

    # 位置
    ax = idata.plot(
        kind='scatter', x='x', y='y', s=idata.reset_index().index*2 + 5, title='目标%s的头指向示意图' % i, alpha=0.5)
    ax.quiver(idata.x, idata.y, idata.vx, idata.vy,
              color='orange', width=0.01, alpha=0.6)
    ax.set_aspect('equal', adjustable='datalim')
    plt.savefig(path.join(datasetPath, 'output/ph-%s.png' % i), dpi=300)
    plt.close('all')
