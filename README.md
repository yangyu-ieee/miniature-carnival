# NeuroChip Copilot：神经微生理系统数据分析与实验规划

## 项目沿革与竞赛边界

NeuroChip Copilot 原型由团队此前围绕神经微生理系统数据分析开发，本目录对应将既有原型面向当前 AI + 器官芯片挑战进行整理和适配；不把它描述为从零开发，也不与团队其他独立项目混合。正式提交前须按实际记录说明此前提交/公开状态、历史结果复用范围及本届新增工作和成员贡献；来源或贡献未经核实的部分不得写成既成事实。

本原型用于神经微生理系统相关研究中的处理状态审阅、证据可视化、MEA 记录质控和预算约束下的候选规划回放。后续实验规划仅基于历史候选回放，尚未验证前瞻实验收益。

## 当前状态

- 已冻结比赛 V1 主线和证据边界。
- 已下载并审计两套公开神经微生理系统/类器官 MEA 数据。DishBrain 是高密度 MEA 上的体外神经网络系统，不将其称为传统微流控器官芯片。
- 两套数据分工使用、不合并训练：DishBrain 用于处理状态指纹与混杂审计；人前脑类器官记录用于原始 MEA 质控。类器官数据没有药物标签，不支持药物分类或跨体系外推。
- 新增人前脑类器官原始记录质控：检查 45 份 8 通道 MAT 记录的 spike-time 完整性、源表 firing rate 一致性和记录内前后半段漂移；发现的 amplitude 计数异常会单独警示。
- 类器官源表使用 Zenodo v3（DOI `10.5281/zenodo.22755004`）；数据版本与实际使用文件应以该 DOI 记录和本项目的数据访问说明为准。
- 数据获取文件清单、OSF/Zenodo 来源、许可核查边界和完整复现命令见 [`DATA_ACCESS_AND_REPRODUCTION.md`](DATA_ACCESS_AND_REPRODUCTION.md)。DishBrain 原始文件许可尚未从当前本地元数据确认，公开仓库不应擅自镜像原始文件或衍生表。
- DishBrain 三药处理条件的跨芯片指纹识别（探索性、芯片分组 OOF）：balanced accuracy
  0.7375、macro-F1 0.7155、macro OVR AUC 0.8794。
- 最早静息会话行选择已修正：只保留真实最早会话行及其缺失值，不再逐列用后续会话的非空值回填。共享芯片子集（11 芯片、64 条记录）的电生理 BA/macro-F1 为 0.6894/0.6928；此前 0.7197/0.7244 属于修正前结果，不再作为当前结果引用。
- 可信度红队页面直接展示共享子集中 11/11 芯片的共同处理顺序及连续区块结构，并把它解释为药物类别与处理阶段难拆的回顾性设计限制；不据此作因果归因。
- 当前不包含湿实验验证，不提供医疗诊断。
- 量子/Ising 仅是后续可插拔实验选择后端，不主张量子优势。

本源码候选沿用已审阅的分析流程与结果；V44.18 调整可信度红队页面两张诊断图的轴标题、边距和标签角度，并修复演示画面中的图表可读性，不改变数据、模型或分析结果。技术报告中保留团队基于数据制作的两张分析图，作为本项目结果展示；它们不是原论文插图的截图或复制。本目录是供本届竞赛评审使用的 source-only 版本，不重复打包报告中的图片，也不包含原始数据、逐行处理数据或派生结果表。代码包不含图片是打包范围说明，不表示报告图表不能用于比赛；报告中已注明数据来源和处理过程。本公开候选不授予项目代码的开源再许可；原始数据与派生数据仍按 [`DATA_ACCESS_AND_REPRODUCTION.md`](DATA_ACCESS_AND_REPRODUCTION.md) 中说明的来源和许可边界处理。

数据行选择更正及受影响字段说明见 [`DATASET_BASELINE_ROW_SELECTION_CORRECTION_20261003.md`](DATASET_BASELINE_ROW_SELECTION_CORRECTION_20261003.md)。

直接依赖的固定版本、许可证标识和上游来源见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。本仓库没有为 NeuroChip Copilot 自有代码附加 MIT、Apache-2.0 等开放许可证；公开供竞赛评审查阅不等于授予复制、修改或再分发权。

## 启动交互 Demo

先按 [`DATA_ACCESS_AND_REPRODUCTION.md`](DATA_ACCESS_AND_REPRODUCTION.md) 获取所需公开数据并运行复现脚本。源码候选不含数据；可运行以下脚本从 OSF/Zenodo 原发布方本地下载并按冻结校验值核验。DishBrain 文件许可尚未从 OSF 文件元数据确认，所以该脚本不会把数据打包或重新发布。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src\download_public_data.py --dataset organoid
# Only if you have authorized access and use permission for the OSF files:
.\.venv\Scripts\python.exe src\download_public_data.py --dataset dishbrain
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## 数据审计

```powershell
.\.venv\Scripts\python.exe src\audit_dataset.py
```

生成：

- `data/processed/recording_audit.csv`
- `data/processed/organoid_feature_table.csv`
- `data/processed/dataset_audit.json`

公开仓库发布前不会提交未经核验的大型中间文件；数据按原始许可证和 DOI 获取。

## DishBrain 复现

```powershell
.\.venv\Scripts\python.exe src\audit_dishbrain.py
.\.venv\Scripts\python.exe src\audit_dataset.py
.\.venv\Scripts\python.exe src\evaluate_drug_fingerprint.py
.\.venv\Scripts\python.exe src\audit_fingerprint_robustness.py
.\.venv\Scripts\python.exe src\audit_dose_negative_control.py
.\.venv\Scripts\python.exe src\protocol_confound_scanner.py
.\.venv\Scripts\python.exe src\evaluate_experiment_portfolio.py
.\.venv\Scripts\python.exe src\audit_portfolio_random_baseline.py
.\.venv\Scripts\python.exe src\audit_portfolio_one_swap.py
.\.venv\Scripts\python.exe src\evaluate_drug_response.py
.\.venv\Scripts\python.exe src\analyze_organoid_recording_stability.py
```

完成数据获取与上列分析后，运行完整证据测试。源代码包不包含受许可约束的数据或派生结果；在未复现这些文件的干净 checkout 中，4 个下载器单元测试会运行，证据契约测试会明确标记为 `skipped`，而不会误报为失败。复现产物到位后，证据契约测试会读取并校验对应结果：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

