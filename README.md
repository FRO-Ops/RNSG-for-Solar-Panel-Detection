# RNSG: Improved ResNet50 with SE and GAM Attention for Solar Panel Defect Detection

论文状态：处于投刊阶段 
论文标题：**RNSG: Improved ResNet50 with SE and GAM Attention for Solar Panel Defect Detection** 
核心思想：在 ResNet50 基础上，通过引入 SE 通道注意力模块 与 GAM 全局注意力模块，显著提升太阳能球面构件表面缺陷识别的精度与鲁棒性，为工业质检场景提供高效的太阳能球面构件缺陷诊断解决方案。（官方 PyTorch 实现）

# 1. 研究背景与模型定位

太阳能球面构件是工业装备中的关键基础零部件，在生产质检环节中，构件表面的完好、破损、污损缺陷识别依赖人工目视检测，存在检测效率低、主观误差大等问题。受曲面反光、光照不均、缺陷尺度差异大等因素影响，传统深度学习模型难以精准提取太阳能球面构件表面微小缺陷特征。

本文提出一种基于 ResNet50 改进的 RNSG 模型，在 ResNet50 骨干网络中嵌入 SE 通道注意力模块与 GAM 全局注意力模块，通过双注意力协同机制完成特征重标定，强化缺陷区域特征表达，抑制曲面反光带来的背景噪声干扰。在公共开源 WPLDD 太阳能球面构件缺陷数据集上完成验证，该模型能够在可控参数量与推理开销下，提升太阳能球面构件三分类缺陷识别的精度与鲁棒性，为工业场景下太阳能球面构件自动化质检提供轻量化、高效的识别方案。

## 2 RNSG 核心创新点

### 2.1 SE 与 GAM 双注意力协同融合

 SE 注意力模块为通道维度注意力机制，通过全局池化与多层感知机建模通道间依赖关系，自适应增强缺陷相关特征通道权重，抑制无关背景通道干扰；GAM 注意力模块同时建模通道与空间二维特征关联，通过多层卷积放大空间维度上微小缺陷区域特征响应。 本文将 SE 通道注意力与 GAM 混合注意力嵌入 ResNet50 骨干网络深层，构建双注意力协同机制。SE 模块优先筛选有效特征通道，GAM 模块进一步定位太阳能球面构件表面细微破损、污渍等局部缺陷区域，有效缓解球面曲面反光、光照不均带来的特征混淆问题，提升模型对 Clean 完好、Broken 破损、Dirty 污损三类样本的区分能力。

### 2.2 多尺度特征增强嵌入策略

针对太阳能球面构件缺陷尺度差异大的特点（微小划痕、大面积破损、大面积污渍），在 ResNet50 不同残差阶段分层嵌入注意力模块。网络浅层保留原始残差块提取基础纹理、边缘轮廓特征；网络中层提取中等尺度缺陷特征；在网络深层嵌入 SE+GAM 联合注意力模块，对高层语义特征进行特征重标定。 该分层嵌入策略能够实现从底层边缘细节到高层缺陷语义的渐进式特征融合，兼顾小尺度细微破损与大面积污损的特征提取能力，降低曲面反光造成的噪声干扰。

### 2.3 精度与推理效率的平衡优化
基于 ResNet50 骨干网络引入轻量化 SE 与 GAM 注意力模块，仅少量增加参数量与计算开销。模型总参数量 25.09 M，计算量 8.34 GMac，在 RTX 5060 Laptop GPU 上单图推理耗时仅 4.61 ms。在 WPLDD 数据集上取得 93.53% 的测试集准确率。 相较于原生 ResNet50 基线模型，RNSG 通过双注意力机制显著提升太阳能球面构件缺陷识别精度，同时保持模型较高推理速度，具备部署在工业边缘嵌入式设备完成太阳能球面构件表面缺陷实时检测的潜力。

## 3. 实验数据集：WPLDD

### 3.1 数据集概况
WPLDD 为公共开源太阳能球面构件表面缺陷数据集，数据集可从公共开源平台获取，无需自行采集。

数据集包含三类样本：完好构件（Clean）、破损构件（Broken）、污损构件（Dirty），图像总数共 6079 张，统一 resize 至 320×320；数据集划分比例为训练：验证：测试 = 7:2:1（通过代码自动划分）。

### 3.2 数据集结构
数据集存放路径：`D:\KY\WPLDD\CNN_Data_Arranged\320p_split` 
文件夹组织如下：

```
320p_split/ 
├── train/ 
│ ├── Clean/ 
│ ├── Broken/ 
│ └── Dirty/ 
├── val/ 
│ ├── Clean/ 
│ ├── Broken/ 
│ └── Dirty/ 
└── test/ 
  ├── Clean/ 
  ├── Broken/ 
  └── Dirty/
```
文件夹内图像来源于公共开源数据集，涵盖不同拍摄角度、光照变化下的太阳能球面构件样本，可直接用于模型训练、验证与测试。
## 4. 实验环境配置

### 4.1 软件环境安装
推荐使用 Anaconda 创建虚拟环境
```bash
conda create -n py31 python=3.12
conda activate py31
pip install torch torchvision numpy pillow matplotlib scikit-learn pandas
```

### 4.2 硬件与训练参数

-   硬件：Intel i7-14650HX + RTX 5060 Laptop GPU
-   框架：PyTorch
-   训练轮次：120 epoch
-   输入图像尺寸：320×320
-   Batch size：8
-   初始学习率：1e-4
-   优化器：AdamW
-   学习率调度器：CosineAnnealingWarmRestarts
-   训练总耗时：约 1.5 小时


## 5. 实验结果与分析

### 5.1 基线模型对比

任务：太阳能球面构件缺陷三分类（Clean/Broken/Dirty），输入分辨率 320×320，基于 WPLDD 测试集结果

|Model|Acc(%)|Params(M)|FLOPs(GMac)|
|-|-|-|-|
|VGG16|88.25|138.36|15.43|
|ResNet50|90.26|25.60|8.21|
|Swin-Tiny|91.85|28.29|4.47|
|**RNSG(Ours)**|**93.53**|**25.09**|**8.34**|

注： 准确率为多次重复实验的平均值，结果稳定性高，最优结果可达 93.53%； 相较于原生 ResNet50 基线，RNSG 参数量小幅下降（从 25.60M 降至 25.09M），准确率提升 3.27 个百分点； 相较于 Swin Transformer（Tiny），RNSG 参数量降低 11.3%，准确率提升 1.68 个百分点； 在缺陷类别区分上，基线模型（如原生 ResNet50、VGG16）对 Broken 破损与 Dirty 污损样本混淆率较高，RNSG 通过 SE+GAM 双注意力协同机制，将两类样本的混淆率显著降低，增强太阳能球面构件微小缺陷的识别能力； 模型推理耗时仅 4.61 ms / 帧，在保持高精度的同时实现高效推理，具备部署于工业边缘质检设备的潜力。
### 5.2 混淆矩阵消融实验

本文设置 4 组对照实验，分别为原生 ResNet50、仅 SE 注意力、仅 GAM 注意力、RNSG（SE+GAM 联合注意力），对应图 A、B、C、D 四组混淆矩阵。 由混淆矩阵结果可见，RNSG 模型对 Clean 完好、Broken 破损、Dirty 污损三类样本的识别效果最优，有效减少破损与污损样本之间的误判，验证 SE 与 GAM 联合注意力模块的有效性。

## 6. 项目文件说明

项目根目录：`D:\KY\resnet\resnet50+SE+GAM`
```
resnet50+SE+GAM/
├── correct_output/          # 模型输出结果、最优权重存放文件夹
├── juzhentu.ipynb           # 混淆矩阵绘制代码
├── predict_ResNet50_SE_GAM.ipynb  # 模型预测脚本
├── train_ResNet50_SE_GAM.ipynb    # 模型训练主脚本
└── val_predict_result.csv   # 验证集预测结果csv文件
```
-   `train_ResNet50_SE_GAM.ipynb`：模型训练代码，包含数据加载、数据增强、模型搭建、训练循环、保存最优权重
-   `predict_ResNet50_SE_GAM.ipynb`：模型预测脚本，加载训练完成的 best_model.pth，支持单张 / 批量图像预测
-   `juzhentu.ipynb`：读取预测结果，绘制混淆矩阵
-   `correct_output/`：保存训练日志、最优模型权重 best_model.pth
-   `val_predict_result.csv`：验证集预测输出结果，用于后续指标可视化

## 7. 快速运行
### 7.1 训练模型

1.  将数据集路径修改为本地 `D:\KY\WPLDD\CNN_Data_Arranged\320p_split`
2.  在 VSCode 或 Jupyter Notebook 打开 `train_ResNet50_SE_GAM.ipynb`
3.  顺序运行全部 cell，开始训练，训练完成后最优模型权重自动保存至`correct_output`文件夹

### 7.2 模型预测

打开 predict_ResNet50_SE_GAM.ipynb，加载训练得到的 best_model.pth，执行单张图像或批量图像预测，输出预测结果并保存 val_predict_result.csv。

### 7.3 绘制混淆矩阵

打开 juzhentu.ipynb，读取上一步生成的 val_predict_result.csv，运行单元格生成混淆矩阵可视化图。

## 8. 已知问题与后续工作

1.  本模型在光线极度昏暗、严重遮挡的样本上识别性能会下降，后续可以扩充极端光照样本，增强模型鲁棒性；
2.  当前模型仅完成图像分类任务，未来可引入目标检测网络，实现缺陷位置定位 + 分类一体化；
3.  可进一步轻量化模型结构，减少 FLOPs，适配更低算力的嵌入式设备。


## 9. 引用与联系方式

### 9.1 引用

```bibtex
@article{rnsg2026,
  title={RNSG: Improved ResNet50 with SE and GAM Attention for Solar Panel Defect Detection},
  author={[Author List]},
  journal={},
  year={2026},
  note={Submitted for publication}
}
```
If you find this work useful for your research, please cite our paper.
### 9.2 联系方式

Corresponding author: liuwennian@huuc.edu.cn
