# challenge.ipynb 中文讲解

这份 notebook 是 osapiens Makeathon 2026 森林砍伐检测挑战的“导览地图”。它没有训练模型，而是在带你认识战场：数据在哪里、每种遥感数据长什么样、弱标签怎样编码、那些彩色 confidence 图到底画的是什么，以及最后如何把栅格预测变成可提交的 GeoJSON。

一句话概括：它把同一个样例地块 `18NWG_6_6` 拿出来，用 Sentinel-2、Sentinel-1 和 AlphaEarth Foundations 三种数据分别看一遍，再把 GLAD-S2 的砍伐告警 confidence 叠到这些网格上，帮助你理解“输入影像”和“弱标签”之间的空间关系。

## 1. 这个挑战要解决什么

任务是从多源卫星数据中识别 2020 年之后发生的森林砍伐事件。这里的“砍伐”不是随便少了点绿，而是从森林永久转为非森林；并且只有 2020 年时原本是森林的区域，之后发生变化才算目标事件。

现实里的难点是：卫星数据像一段带噪声的侦探录像。云会遮住光学影像，雷达会有斑点噪声，不同地区、不同季节、不同传感器看到的“森林变化”也不一样。notebook 的作用就是先把这些证据摊开，让你知道每张图在说什么、又可能在撒什么小谎。

## 2. 数据目录在讲什么

下载后，数据位于：

```text
data/makeathon-challenge/
```

核心目录如下：

```text
sentinel-1/       雷达时间序列，按 train/test 和 tile 存放
sentinel-2/       光学多光谱时间序列，按 train/test 和 tile 存放
aef-embeddings/   AlphaEarth Foundations 年度 embedding
labels/train/     训练集弱标签：RADD、GLAD-L、GLAD-S2
metadata/         train/test tile 的 GeoJSON 空间范围
```

每个地块用类似 `18NWG_6_6` 的 tile id 标识。`18NWG` 来自 MGRS 网格，后面的数字表示该大网格中的切片位置。

训练集有影像和标签；测试集只有影像，最终要对测试集 tile 输出预测结果。

## 3. notebook 按顺序做了什么

## 3.1 先说明挑战背景

开头几节解释为什么要做森林砍伐检测：EUDR 需要供应链证明商品没有来自砍伐地，卫星可以全球监控，但数据嘈杂、来源不同、标签不完美。

这部分是业务背景，重点是告诉你：模型不只是要在一个地方看起来准，还要能跨地区泛化。

## 3.2 展示 Sentinel-2

notebook 读取：

```text
data/makeathon-challenge/sentinel-2/train/18NWG_6_6__s2_l2a/18NWG_6_6__s2_l2a_2020_1.tif
```

它打印出的信息是：

```text
Number of bands : 12
Dtype           : uint16
CRS             : EPSG:32618
Shape           : (1002, 1002)
```

Sentinel-2 是光学卫星，像一台会看“颜色以外颜色”的相机。除了红绿蓝，它还看近红外、红边、短波红外等波段，这些波段对植被活力、水分、裸土、烧毁或清理后的地表很敏感。

notebook 用 B4、B3、B2 三个波段合成真彩色图：

```python
red   = src.read(4)
green = src.read(3)
blue  = src.read(2)
```

然后用 2% 到 98% 分位数做拉伸，让图像更适合人眼观看。这个步骤只是显示增强，不改变原始数据含义。

## 3.3 在 Sentinel-2 旁边画 GLAD-S2 confidence

Sentinel-2 可视化旁边的 confidence 图来自：

```text
data/makeathon-challenge/labels/train/glads2/glads2_18NWG_6_6_alert.tif
```

注意：这张 confidence 图不是 Sentinel-2 影像本身算出来的，也不是 notebook 里训练模型得到的结果。它是官方提供的 GLAD-S2 弱标签告警栅格。

因为 Sentinel-2 栅格是本地 UTM 坐标系 `EPSG:32618`，而 GLAD-S2 标签通常在 `EPSG:4326`，notebook 用 `rasterio.warp.reproject` 把 GLAD-S2 标签重投影到 Sentinel-2 的网格上：

```python
reproject(
    source=src.read(1),
    destination=alert_reproj,
    dst_transform=s2_transform,
    dst_crs=s2_crs,
    resampling=Resampling.nearest,
)
```

这里用 `nearest` 最近邻重采样很重要，因为 confidence 是类别值，不是连续数值；如果用双线性插值，`2` 和 `4` 之间可能被插出奇怪的 `3.2`。

颜色含义如下：

| 值 | 含义 | 图上直觉 |
|---:|------|----------|
| 0 | No alert，无告警 | 灰色背景 |
| 1 | Recent only，仅最近一次观测检测到 | 很淡的黄色 |
| 2 | Low confidence，低置信度损失 | 黄色 |
| 3 | Medium confidence，中置信度损失 | 橙色 |
| 4 | High confidence，高置信度损失 | 红色 |

所以这张图是在说：“GLAD-S2 这个弱标签系统认为哪些像素可能发生了森林损失，以及它有多确信。”

## 3.4 展示 Sentinel-1

notebook 读取：

```text
data/makeathon-challenge/sentinel-1/train/18NWG_6_6__s1_rtc/18NWG_6_6__s1_rtc_2020_10_ascending.tif
```

打印结果：

```text
Number of bands : 1
Dtype           : float32
CRS             : EPSG:32618
Shape           : (334, 335)
```

Sentinel-1 是雷达卫星。它不靠太阳光，也不怕云，像是在夜里拿手电筒照森林，只不过这个“手电筒”发的是微波。树冠、枝干、地表粗糙度和水分都会影响雷达回波。

本数据提供的是 RTC 产品，即 Radiometrically Terrain Corrected，已经做过地形和辐射校正，更适合直接建模。notebook 使用的是 VV 极化单通道，并把线性 backscatter 转成 dB：

```python
db = np.where(backscatter > 0, 10 * np.log10(backscatter), np.nan)
```

再做分位数归一化，画成灰度图。亮暗代表雷达回波强弱，不是普通照片中的颜色。

## 3.5 在 Sentinel-1 旁边也画 GLAD-S2 confidence

第二张 confidence 图仍然来自同一个文件：

```text
data/makeathon-challenge/labels/train/glads2/glads2_18NWG_6_6_alert.tif
```

区别只是这次把 GLAD-S2 标签重投影到了 Sentinel-1 的网格上。Sentinel-1 的 shape 是 `(334, 335)`，比 Sentinel-2 的 `(1002, 1002)` 粗很多，所以 confidence 图看起来分辨率也会跟着 Sentinel-1 的网格走。

换句话说：confidence 的“消息来源”没变，变的是“投影到哪张底图的坐标纸上”。

## 3.6 展示 AlphaEarth Foundations

notebook 读取：

```text
data/makeathon-challenge/aef-embeddings/train/18NWG_6_6_2020.tiff
```

打印结果：

```text
Number of bands : 64
Dtype           : float32
CRS             : EPSG:4326
Shape           : (1004, 998)
```

AlphaEarth Foundations 不是传统意义上的一颗“卫星图像”。它更像是一个已经读过大量地球观测数据的 foundation model，把每个像素压缩成 64 维特征向量。你可以把它理解为：原始卫星影像是照片和雷达回波，AlphaEarth 是模型提前写好的“地表语义笔记”。

notebook 随机选了 3 个 embedding 维度当作 RGB 来画：

```text
Randomly selected bands (1-indexed): [6, 42, 49]
```

这张 RGB 图不是自然颜色，也不是物理波段颜色，只是把 64 维特征中的 3 维映射到红绿蓝，帮助人眼看出空间纹理。

## 3.7 在 AlphaEarth 旁边再画 GLAD-S2 confidence

第三张 confidence 图依旧来自：

```text
data/makeathon-challenge/labels/train/glads2/glads2_18NWG_6_6_alert.tif
```

这次 GLAD-S2 和 AlphaEarth 都是 `EPSG:4326`，但 shape、transform 不一定完全一致，所以 notebook 仍然执行重投影/重采样，把标签对齐到 AlphaEarth 的像素网格。

这三组图的规律可以总结为：

```text
左图：某种输入数据的样子
右图：同一个 GLAD-S2 alert confidence 标签，对齐到左图的网格后显示
```

## 4. confidence 图到底依靠哪些数据

notebook 里的 confidence 图全部依靠 GLAD-S2 alert 标签文件：

```text
data/makeathon-challenge/labels/train/glads2/glads2_18NWG_6_6_alert.tif
```

它们不依赖 Sentinel-1 的回波值、不依赖 Sentinel-2 的 RGB 值，也不依赖 AlphaEarth 的 embedding 值来计算 confidence。影像数据只提供“对齐目标”和视觉背景，confidence 数值来自标签栅格本身。

三个 confidence 图之间的区别如下：

| 出现位置 | 标签来源 | 对齐到哪个网格 | 目的 |
|----------|----------|----------------|------|
| Sentinel-2 小节 | GLAD-S2 alert | Sentinel-2 `EPSG:32618`, `(1002, 1002)` | 看光学影像和告警位置关系 |
| Sentinel-1 小节 | GLAD-S2 alert | Sentinel-1 `EPSG:32618`, `(334, 335)` | 看雷达图和告警位置关系 |
| AlphaEarth 小节 | GLAD-S2 alert | AlphaEarth `EPSG:4326`, `(1004, 998)` | 看 embedding 空间纹理和告警位置关系 |

所以不要把这些 confidence 图误解为模型输出。它们是训练用弱标签的可视化，是“别人已经给你的粗糙线索”。

## 5. 不同卫星和数据源的区别

| 数据源 | 类型 | 主要信息 | 优点 | 局限 |
|--------|------|----------|------|------|
| Sentinel-2 | 光学多光谱 | 12 个反射率波段，含 RGB、近红外、红边、短波红外 | 直观，植被和裸土变化明显，适合看砍伐后的颜色/水分变化 | 怕云、烟、阴影；热带地区常被云遮 |
| Sentinel-1 | 雷达 | VV backscatter，RTC 校正后产品 | 全天候、昼夜可用，能穿云，适合云多地区 | 图像不直观，有雷达 speckle，解释门槛更高 |
| AlphaEarth Foundations | 预训练 embedding | 每像素 64 维语义特征 | 已融合多源时空信息，可能更适合下游模型 | 不是物理波段，难直接解释每一维含义 |

更形象地说：

- Sentinel-2 像一双彩色眼睛，能看见树叶的颜色、湿度和裸土变化，但遇到云就被蒙住。
- Sentinel-1 像一副雷达耳朵，不看颜色，听地表回波；云来了也能工作，但声音嘈杂，需要会听。
- AlphaEarth 像一份压缩后的地表档案，已经把很多观测揉成 64 维特征，适合模型读，但人类不容易逐维解释。

## 6. 三种弱标签怎么理解

notebook 还介绍了三个标签来源。它们都是 weak labels，也就是弱标签：不是人工精标真值，而是已有告警系统给出的 noisy indication。可以用来训练，但要带着怀疑精神使用。

## 6.1 RADD

RADD 基于 Sentinel-1 雷达检测森林砍伐。文件在：

```text
data/makeathon-challenge/labels/train/radd/
```

编码方式把 confidence 和日期塞进同一个整数：

```text
0      = no alert
2xxxx  = low confidence alert
3xxxx  = high confidence alert
xxxx   = days since 2014-12-31
```

例子：

```text
20001 = 低置信度，2015-01-01
30055 = 高置信度，2015-02-24
21847 = 低置信度，2020-01-21
```

## 6.2 GLAD-L

GLAD-L 基于 Landsat 光学数据。每年两个文件：

```text
alertYY.tif
alertDateYY.tif
```

`alertYY` 编码：

```text
0 = no loss
2 = probable loss
3 = confirmed loss
```

`alertDateYY` 编码：

```text
0      = no alert
非 0 值 = 20YY 年内的 day-of-year
```

它按年份分文件，所以文件名里带 `YY`。

## 6.3 GLAD-S2

GLAD-S2 基于 Sentinel-2 光学数据，也是 notebook 里 confidence 图使用的标签源。文件在：

```text
data/makeathon-challenge/labels/train/glads2/
```

主要有两个文件：

```text
glads2_{tile_id}_alert.tif
glads2_{tile_id}_alertDate.tif
```

`alert` 编码：

```text
0 = no loss
1 = loss detected only in the most recent observation
2 = low confidence loss
3 = medium confidence loss
4 = high confidence loss
```

`alertDate` 编码：

```text
0      = no alert
非 0 值 = days since 2019-01-01
```

相比 GLAD-L，GLAD-S2 不按年份拆 alert 文件，而是把多年信息放在一个 raster 中。

## 7. 提交示例在干什么

notebook 最后演示如何把一个二值预测栅格转成 GeoJSON。示例偷懒用了 RADD 标签当作假预测：

```text
data/makeathon-challenge/labels/train/radd/radd_18NWG_6_6_labels.tif
```

流程是：

1. 读取单波段 raster。
2. 把所有非零值转成 `1`，表示预测为砍伐。
3. 写成临时二值 GeoTIFF。
4. 调用 `submission_utils.raster_to_geojson` 转成矢量多边形。
5. 输出到：

```text
submission/pred_18NWG_6_6.geojson
```

`raster_to_geojson` 会做几件关键事情：

- 把值为 `1` 的连通区域 vectorize 成 polygon。
- 转到 `EPSG:4326`，也就是经纬度坐标。
- 用估计的 UTM 坐标系计算面积。
- 默认过滤掉小于 `0.5 ha` 的小斑块。
- 给输出加上 `time_step = None` 字段。

示例运行后统计结果是：

```text
Number of polygons : 106
Total area         : 3431.28 ha
Min polygon size   : 0.5165 ha
Max polygon size   : 811.46 ha
Avg polygon size   : 32.3706 ha
```

这只是格式演示，不代表一个真正模型的结果。

## 8. submission 时需要提交什么格式

最终提交的不是 `.tif`，也不是 notebook 里的 matplotlib 图片，而是 GeoJSON 格式的矢量多边形文件。直观地说：你的模型先在栅格上判断哪些像素是砍伐区域，然后要把这些连成片的像素“描边”，变成一个个经纬度 polygon。

单个 tile 的输出示例路径可以像 notebook 里这样：

```text
submission/pred_18NWG_6_6.geojson
```

GeoJSON 的整体结构是一个 `FeatureCollection`：

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "time_step": null
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [-73.123, 4.567],
            [-73.122, 4.567],
            [-73.122, 4.568],
            [-73.123, 4.568],
            [-73.123, 4.567]
          ]
        ]
      }
    }
  ]
}
```

关键要求可以记成四句话：

1. 坐标系要是 `EPSG:4326`，也就是经纬度，顺序是 `[longitude, latitude]`。
2. 每个 `Feature` 表示一块连续的预测砍伐区域。
3. `geometry` 通常是 `Polygon`，也可能在某些情况下由多个 polygon 组成。
4. `properties` 里 notebook 的工具会写入 `time_step: null`；如果你没有做发生时间预测，保持 `null` 就可以。

提交前，你的模型输出通常会先长这样：

```text
单波段 GeoTIFF
0 = no deforestation
1 = deforestation
```

然后调用：

```python
from submission_utils import raster_to_geojson

geojson = raster_to_geojson(
    raster_path="your_binary_prediction.tif",
    output_path="submission/pred_18NWG_6_6.geojson",
    min_area_ha=0.5,
)
```

`raster_to_geojson` 会负责把二值 raster 变成提交需要的 GeoJSON：提取值为 `1` 的连通区域、转成 polygon、重投影到 `EPSG:4326`、过滤掉太小的斑块，并写出文件。

需要特别注意：传进去的 raster 必须已经二值化。也就是说，模型如果输出的是概率图，例如 `0.0` 到 `1.0` 的 confidence map，你要先自己选阈值：

```python
binary = (probability > 0.5).astype("uint8")
```

再写成 GeoTIFF，然后转换为 GeoJSON。leaderboard 要评估的是你预测出来的砍伐区域多边形，不会直接吃模型概率图。

## 9. 建模时最该记住的几件事

第一，输入影像和标签可能不在同一个 CRS、shape、resolution 下。训练前必须认真对齐，不然模型会学到错位的监督信号。

第二，confidence 图是弱标签，不是黄金真值。红色区域更可信，但仍可能误报；灰色区域也不保证绝对没有砍伐。

第三，多模态数据互补很重要。Sentinel-2 看颜色和植被光谱，Sentinel-1 穿云看结构变化，AlphaEarth 提供预训练语义特征。好的方案通常不是迷信单一数据源，而是把它们当成不同证人：有的眼尖，有的不怕黑，有的记性好。

第四，提交格式最终是 GeoJSON polygon，不是模型 logits、不是 confidence raster。无论模型内部多复杂，最后都要落到测试 tile 上的砍伐区域多边形。

## 10. 最短版总结

这个 notebook 做的是数据入门和可视化：

- 说明挑战目标：检测 2020 年后的森林砍伐。
- 展示数据结构：Sentinel-1、Sentinel-2、AlphaEarth、三类弱标签。
- 用 `18NWG_6_6` 样例 tile 展示三种输入数据。
- 三张 confidence 图都来自 GLAD-S2 alert 标签，只是分别重投影到 Sentinel-2、Sentinel-1、AlphaEarth 的网格上。
- 解释 RADD、GLAD-L、GLAD-S2 的标签编码。
- 演示如何把二值预测 raster 转成最终提交用 GeoJSON。
- submission 最终交 GeoJSON polygon，坐标系是 `EPSG:4326`，不是提交原始 `.tif` 或概率图。

如果把整个挑战比作破案：Sentinel-2 是现场照片，Sentinel-1 是穿透云雾的雷达证词，AlphaEarth 是整理过的案情摘要，GLAD/RADD 标签是线人提供的线索。notebook 的任务不是破案，而是教你先把这些证据摆正、看懂、对齐。
