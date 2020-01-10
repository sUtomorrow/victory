# 数据处理格式说明

每个病人原始HU值矩阵保存为int16类型的npy, npy文件名字为病人的seriesuid.

同时使用一个同名字的pkl文件保存其他信息,字典格式如下:

```
{
  'annotations': [ # 如果是测试数据,没有给标签,就是个空list
    [x, y, z, dx, dy, dz, label], # 全都变换为像素单位,
    ...
    [x, y, z, dx, dy, dz, label]
  ],
  'direction': list, # 长度为9,表示3*3的方向余弦矩阵,用于还原世界坐标
  'origin':  [x, y, z], # origin信息,用于还原世界坐标
  'spacing': [x, y, z], # spacing信息,用于还原世界坐标
}
```