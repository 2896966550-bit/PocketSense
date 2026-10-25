# PocketSense 寒假执行手册（RUNBOOK）

> 前置状态（2026-10-25 S 阶段完成后）：仓库 `pocketsense/` 已含全部管线；三公开数据集已下载解析入库；
> UCI 官方划分 RF 基线 0.808；网页端侧推理已验证（INT8 · WASM 0.3ms）；采集页与加载器闭环已验证。
> 每完成一项，运行其验收命令并 git commit 一次。任何人（携任一 AI 助手）均可照此执行。

## 环境准备（一次性）
```bash
cd pocketsense
pip install -r requirements.txt        # 网络慢加 -i https://pypi.tuna.tsinghua.edu.cn/simple
python run.py test                     # 预期：17 passed
python run.py smoke                    # 预期：SMOKE PASS <10s
python run.py data-build               # 预期：S2 PASS ~44s（数据已存在则幂等快跑）
```

## W1：MotionSense / PAMAP2 基线 + 特征精修
- 任务：为两个长表数据集写「按人划分 + GroupKFold×3 种子」训练入口（复用 `psense/models/train_rf.py`、`train_cnn.py`，输入改为从 parquet 读入）；补 UCI 的 GroupKFold 版本（对照官方划分）
- 验收：三数据集 RF/CNN 各一张含均值±std 的结果表，落 `reports/`
- 已知点：UCI 0.808 是 27 维特征的朴素值，别慌；W2 特征消融会抬

## W2：信号处理与特征消融（E4）
- 任务：窗口 1/2/3s、有无陀螺仪、有无增强、S/M/L 模型四组消融；统一在按人 CV 下跑
- 验收：消融表 + 每组混淆矩阵；上下楼混淆现象如实记录

## W3：1D-CNN 正式训练 + 早停 + 随机性报告（E1-E3）
- 任务：CNN 在三数据集的正式版本（epochs≥30 + 早停），3 种子；产出最终候选模型
- 验收：最终模型 checkpoint + 测试集指标（均值±std）

## W4：端侧部署完善（E5/E10）
- 任务：把最终模型导出 ONNX INT8（复用 `psense/export/export_onnx.py`，校准数据用真实训练窗口）；
  替换 `web/model_int8.onnx`；`web/infer.html` 加实时时间轴与延迟面板；手机实测（安卓 Chrome + 若已部署 Pages 则 iOS Safari）
- 验收：量化前后精度差 <2%；手机演示可用；延迟表
- 部署：建 GitHub 账号 → push → GitHub Pages（infer.html 与 collect.html 同目录可直接托管）

## W5：跨数据集泛化 + few-shot（E6/E7，报告核心章）
- 任务：3 数据集两两互测（5 类交集）；目标域每人每类 1/5/20 窗微调曲线；t-SNE 域偏移可视化
- 验收：6 个有向对的泛化矩阵图 + 适配曲线图
- 如有自采数据（T7）：放置鲁棒性实验加在本周

## W6：报告与发布（E 全汇总）
- 任务：中文技术报告 8-12 页（章节见蓝图）；README 完善；60-90s 演示视频；
  数据引用与许可核对（UCI CC BY 4.0 / MotionSense / PAMAP2 各自引用）；GitHub release v1.0
- 冻结：**2027-03-01 git tag v1.0，除 typo 不再改动**

## 常用命令速查
```bash
python run.py test          # 单元测试
python run.py smoke         # 合成数据端到端
python run.py data-fetch    # （重）下载数据集
python run.py data-build    # 解析入库 + UCI 基线
python -m http.server 8321 -d web   # 本地打开 http://localhost:8321/infer.html
```

## 已知技术债（寒假顺手处理，不阻塞）
1. 网页端推理未做 0.3-20Hz 带通（合成信号影响小，真实信号接入前补 JS 滤波或接受原始输入并记录）
2. MotionSense 的时间戳是名义 50Hz（CSV 无真实时间戳），跨域实验时在报告如实说明
3. PAMAP2 subject109 无 5 类交集活动（8 人入库，报告说明）
4. collector 真实传感器的单位启发式（>5 判 m/s²）在安卓多机型上需复核
