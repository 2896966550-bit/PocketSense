# PocketSense

基于手机 IMU 的人体活动识别（HAR）端到端项目：公开数据集基准 + 自建数据 → 信号处理 → 经典 ML 与 1D-CNN 对比 → ONNX INT8 量化 → 浏览器端侧实时推理 → 跨数据集域泛化实验。

## 快速开始

```bash
pip install -r requirements.txt   # 网络慢可用: -i https://pypi.tuna.tsinghua.edu.cn/simple
python run.py smoke               # 端到端冒烟（合成数据，<2 分钟）
python run.py test                # 单元测试
python run.py data-fetch          # 下载三个公开数据集（约 850MB）
python -m http.server 8000 -d web # 打开 http://localhost:8000/infer.html 体验端侧推理
```

## 目录结构

```
psense/            Python 包
  data/            合成数据、schema、数据集下载与解析
  feats/           滤波、滑窗、重采样、时频特征
  models/          按被试划分、RF 基线、1D-CNN
  export/          ONNX 导出与 INT8 量化
  pipeline/        冒烟与流水线入口
web/               端侧推理页与采集 PWA（onnxruntime-web，本地 vendor）
configs/           实验配置（JSON）
tests/             单元与端到端测试
docs/RUNBOOK.md    寒假逐任务执行手册
```

## 数据来源与引用

- UCI-HAR: Anguita et al., ESANN 2013（CC BY 4.0）
- MotionSense: Malekzadeh et al., 2019（研究用途）
- PAMAP2: Reiss & Stricker, 2012

本仓库不二次分发原始数据集，仅提供下载脚本与校验和。

## 工程纪律

- 一律按被试（subject）划分训练/测试，测试窗口不重叠
- 重采样统一到 50Hz，单位统一到 g
- UCI-HAR 只使用原始 Inertial Signals，不自用官方 561 维特征
- 量化前后必须报告精度差
