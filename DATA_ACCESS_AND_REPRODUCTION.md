# 数据获取与复现说明

本说明列出 NeuroChip Copilot 需要的公开数据文件、应放置的本地路径以及主要复现命令。原始数据不随代码仓库自动再分发；下载和再使用前应遵守数据源当前显示的许可与使用条件。`src/download_public_data.py` 可从 OSF/Zenodo 官方 API 取得指定文件、校验已审计的字节数和 SHA-256，并安全解压 45 个类器官 MAT 文件；它只写入本机 `data/raw/`，不会将数据重新打包或上传。

## 数据来源和文件

### DishBrain 神经微生理系统数据

来源：Watmuff 等，*Communications Biology* (2025)，DOI [10.1038/s42003-025-08194-6](https://doi.org/10.1038/s42003-025-08194-6)。原论文的数据可用性声明链接至 [OSF `xwf7v`](https://osf.io/xwf7v/)，其后的上标 44 是参考文献编号，不属于 OSF 标识；代码可用性声明也链接至同一项目。本项目本地归档的 `xwf7v` 根目录清单列出下表四个实际使用文件；从文件名和用途看，它们是分析所用的整理后/派生输入，不应称作原始电生理记录。当前仍未核实这些整理后输入与原始实验记录之间的逐文件对应关系。为保持可审计性，以下文件仍按本地清单记录的名称、大小和 SHA-256 标识；取得文件前请核对源站当前条款，并保留原文件名后放入 `data/raw/dishbrain/`。匿名 API 可用时可运行下载器；遇到 401/403 时请使用下述本地校验回退流程，不要向脚本传入或硬编码个人凭据。

| 项目使用的整理后输入文件（OSF `xwf7v` 清单） | 大小（字节） | 本地审计 SHA-256 |
|---|---:|---|
| `drug_game_stats` | 69,317 | `bb9aa84c010337bcab0f7e66ceb2356f39b26c9f2bf024e5480fd5b757025f3a` |
| `drug_testing_firing_stats.feather` | 38,306 | `3a6e90a810afa3a9f7b123940af575fde9f016b98925016ffea20caa76e9f67e` |
| `criticality_metrics_drug` | 91,874 | `6cac57d32ffe4373a7d9f4153df4a1e0d3b062a23adb8994145a9f13856266b6` |
| `burst_pattern_class_drug` | 67,737 | `a1e56d86ac80567cee2fd9aaedcb3a24bba1a313dbef6cdfe9433b7af8619425` |

上述 SHA-256 来自本项目已下载文件及 OSF 根目录清单的本地审计记录，用于核对文件是否一致，不代表 OSF 对这些文件授予了特定再分发许可。当前保存的根目录清单没有提供可确认的文件级许可证字段。OSF 的公开内容仍受项目管理员指定的许可和适用版权条件约束；公开代码仓库不得因此默认镜像这些原始文件或其衍生表。下载脚本只便于评审者从原发布方本地取数，并不能替代许可核查。请检查 OSF 项目/文件当前的许可元数据；如未显示许可或许可范围不清，联系数据作者或 OSF 管理员确认后再决定是否公开。OSF 关于许可的说明见[官方许可指南](https://help.osf.io/article/148-licensing)和[服务条款第 5 节](https://osf.io/terms-of-use/)。

**许可边界复核（2026-09-30）：** 关联论文网页标示论文文章采用 CC BY-NC-ND 4.0；该信息只确认论文文章的许可，不能据此推定 OSF 项目中四个独立数据文件采用相同许可。OSF《服务条款》第 5–6 节对公开内容的一般使用授权，同时注明使用受版权、适用限制及内容特定条件约束；OSF 许可指南也提醒，未选许可时不能据此推定可自由复用或改编。参见 [OSF 服务条款](https://osf.io/terms-of-use/) 与 [OSF 许可指南](https://help.osf.io/article/148-licensing)。当前执行环境无法访问 `xwf7v` 项目页及公开 API，尚不能核对四个文件对应的项目/文件许可和额外条件；因此既不判定“禁止使用”，也不判定“许可已无条件解决”。公开包继续只提供来源、哈希和从原发布方取数的说明，不含原始数据、处理表或派生结果。提交前应由参赛队确认比赛分析、公开展示聚合结果和再分发各自适用的权限；仅可见/可下载不等于文件级许可审计已经完成。论文许可信息见[原始论文](https://doi.org/10.1038/s42003-025-08194-6)。

**公开资料补核（2026-10-09，更正）：** 先前从 PDF 文本层读取 URL 时，把 `xwf7v` 后的上标参考文献编号连入了链接。原论文网页明确将数据链接显示为 `https://osf.io/xwf7v`，并将相邻的 44 标为上标引文；代码可用性声明也指向 `xwf7v`。本地清单里的四个文件位于同一 `xwf7v` 项目根目录快照中。仍未核实的是这些整理后输入与原始实验记录的逐文件对应关系，以及具体 OSF 文件级许可。OSF《服务条款》关于 Public 内容的使用许可受管理员指定许可及内容特定限制约束；本地清单未带可确认的文件级许可字段，不据通用条款推定文件许可结论。来源：[原论文 Data availability / Code availability](https://www.nature.com/articles/s42003-025-08194-6)、[OSF Terms §5–6](https://osf.io/terms-of-use/)、[OSF Licensing](https://help.osf.io/article/148-licensing)。

**本地 OSF 根目录快照字段核对（2026-10-01）：** `data/raw/dishbrain/osf_root_listing.json` 的本地文件时间为 2026-09-21 12:26:28，大小 30,306 bytes，SHA-256 `AD695295C04C8F9EC0BD2DE018A95006B42370B79E724BBC28E991638DFF6DE1`；快照记录 20 个文件/文件夹条目。对上列四个实际使用文件逐一核对 JSON 属性：均有文件名、路径、大小、修改时间和哈希等字段，但没有 `license` 或 `rights` 字段。这只证明该份旧快照未携带这些字段；它不是项目级元数据，也不能证明当前 OSF 项目、组件或文件没有单独许可/使用条件。当前许可状态仍为**未核实**，不得由文件可下载、论文许可或本地快照推断为已获授权。

**网页/API 访问状态（2026-10-02）：** 在未登录状态打开 `https://osf.io/xwf7v/` 会转至 OSF 登录页；只读 GET `https://api.osf.io/v2/nodes/xwf7v/` 返回 HTTP 401，错误为缺少认证凭据。未输入账号、密码或令牌。该结果说明本次匿名访问无法核对当前节点元数据，不说明许可不存在或数据不能用于本研究。请代码/数据权利负责人从有权访问的账号核验节点和四个实际使用文件的 rights/license 字段，或向数据发布者确认；在此之前维持“许可/公开展示范围未核实”的结论。

**DishBrain 匿名下载回退流程（2026-10-02）：** 下载器会尝试匿名列出 OSF 文件；若 OSF 返回 HTTP 401/403，会报告无法匿名取得数据且未发送凭据。单独运行 `--dataset dishbrain` 时以失败状态退出；运行 `--dataset all` 时仍会继续尝试独立来源 Zenodo 的类器官数据，最后汇总逐数据集结果并以非零状态退出。不要把 API 拒绝解释为文件许可结论，也不要绕过访问限制。若队员通过其获准的来源访问方式取得上表四个整理后分析输入文件，可保留原文件名并放在 `data/raw/dishbrain/`，然后运行：

```powershell
.\.venv\Scripts\python.exe src\download_public_data.py --dataset dishbrain --verify-existing
```

该模式只核验本地文件大小和 SHA-256，不联网、不上传、不覆盖文件；字节校验成功仍不等于获得公开再分发或展示权限。之后再按下文顺序运行审计与分析脚本。若无获准的文件访问方式，停止数据重建并向数据发布者确认，不应尝试使用他人凭据。

### 人前脑类器官自发活动数据

来源：Itatani 与 Zavaglia，Zenodo [v3 记录 22755004](https://zenodo.org/records/22755004)，DOI [10.5281/zenodo.22755004](https://doi.org/10.5281/zenodo.22755004)，记录许可为 CC BY 4.0。获取并放置：

| 文件 | 本地目标位置 | 说明 |
|---|---|---|
| `all_fig_table_data_revision2.xlsx` | `data/raw/` | v3 源表 |
| `Indiv_organoids.zip` | `data/raw/` | 解压后将 45 个 MAT 文件置于 `data/raw/Indiv_organoids/` |

版本对照使用 Zenodo [v2 记录 20286251](https://zenodo.org/records/20286251) 中的 `fig_table_source_data.xlsx`，放在 `data/raw/`。该旧版文件只用于 v2/v3 字段差异审计；核心 v3 记录质控可以从 v3 数据重建。版本变更和本地哈希见 `DATASET_VERSION_AUDIT_20260927.md`。

## 目录结构

完成数据获取后，项目应至少包含：

```text
data/raw/
├── all_fig_table_data_revision2.xlsx
├── fig_table_source_data.xlsx             # v2 对照文件
├── Indiv_organoids/
│   └── *.mat
└── dishbrain/
    ├── drug_game_stats
    ├── drug_testing_firing_stats.feather
    ├── criticality_metrics_drug
    └── burst_pattern_class_drug
```

不要把未核实许可的原始数据复制到公开仓库或提交附件中。公开代码仓库可保留本说明、来源链接、下载后的哈希核验逻辑和分析代码；评审如需复现，应从原始发布方按其当时许可获取数据。

## 复现顺序

在项目根目录使用 Python 3.12。先创建虚拟环境并安装依赖，再获取数据、运行审计与分析。公开代码包不含原始数据；建议先获取许可明确、可匿名访问的 Zenodo 类器官数据，再按授权情况尝试 DishBrain。遇到 OSF 匿名访问失败时，Zenodo 下载仍会继续：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src\download_public_data.py --dataset organoid
# Only if you have authorized access and use permission for the OSF files:
.\.venv\Scripts\python.exe src\download_public_data.py --dataset dishbrain
```

`--dataset` 可选 `dishbrain`、`organoid` 或 `all`。`--dataset all` 会分别尝试两个数据源；若其中一个失败，会列出该数据集失败原因、保留另一项已校验文件并返回退出码 1，评审者应按提示处理，不能把部分成功误认为完整复现。下载前请核验源站当前使用条款；校验通过只证明文件字节与归档版本一致，不代表获得再分发许可。若已有目标文件但哈希不匹配，脚本会停止并保留原文件，不会静默覆盖。对于 OSF 数据，只有在获准访问和使用的前提下才运行 `--dataset dishbrain`。

然后按以下顺序运行数据审计、模型和规划分析：

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

运行回归测试并启动交互演示：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\streamlit run app.py
```

DishBrain 验证必须以 `chip_id` 为分组单位；不得把同一芯片的记录随机拆到训练集和验证集。类器官数据没有药物标签，只用于原始记录质控与记录内变化分析。当前结果是公开数据的回顾性分析，不表示已完成前瞻湿实验。

## 结果与边界

项目冻结的关键数值、源数据版本、SHA-256、生成脚本和已知偏差记录在 `results/`、`DATASET_VERSION_AUDIT_20260927.md` 与 `DEMO_QA_20260927.md`。复现时应以脚本新生成结果为准；若与冻结结果不一致，应先报告差异，不要直接改报告中的数字。

即使跨芯片处理指纹高于多数类基线，当前数据仍存在药物身份与实验日期/顺序及剂量方案关联；顺序单变量负对照 BA=0.9682，事后剂量单变量负对照 BA=0.7842（共享芯片子集 BA=0.8167）。剂量对照为探索性风险诊断，不证明电生理模型直接使用剂量，也不识别因果来源。现阶段只支持处理条件相关信号的探索性审阅，不支持药物特异机制、临床预测或湿实验成本下降。量子/Ising 后端仅为可选扩展，未证明量子优势。

共享芯片处理区块审计补充：`src/audit_fingerprint_robustness.py` 同时生成 `results/dishbrain_fingerprint_robustness/shared_chip_protocol_block_order.csv` 和 `robustness_audit.json` 中的 `shared_chip_protocol_block_order` 项。当前 11/11 块共享芯片都按 Phenytoin → Perampanel → Carbamazepine 的同一首次处理顺序，并各自形成三个连续处理区块。该结果用于描述可识别性风险，不代表随机化试验、因果效应或独立外部验证；细节见 `output/OOC_PROTOCOL_BLOCK_ORDER_AUDIT_20261003.md`。
