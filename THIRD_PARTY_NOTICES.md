# 第三方软件来源与许可证

本表按当前固定依赖文件 [`requirements.txt`](requirements.txt) 列出直接依赖。版本与许可标识于 2026-09-30 从项目锁定虚拟环境中对应 distribution 的 `METADATA`、随包许可文件核对；上游链接指向相应项目。运行者应保留上游许可声明，并在更新依赖版本时重新核对。

| 包 | 固定版本 | 许可标识/许可文件 | 上游来源 |
|---|---:|---|---|
| h5py | 3.16.0 | BSD-3-Clause | [h5py](https://github.com/h5py/h5py) |
| matplotlib | 3.11.2 | Matplotlib License Agreement（使用随包 `LICENSE` 专用文本；不将其简化映射为未经核实的 SPDX ID） | [Matplotlib](https://github.com/matplotlib/matplotlib) |
| NumPy | 2.5.3 | `BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0`（按 distribution `License-Expression`；覆盖发行包声明的组件许可） | [NumPy](https://github.com/numpy/numpy) |
| openpyxl | 3.1.5 | MIT | [openpyxl](https://foss.heptapod.net/openpyxl/openpyxl) |
| pandas | 3.0.6 | BSD 3-Clause | [pandas](https://github.com/pandas-dev/pandas) |
| PyArrow | 25.0.1 | Apache-2.0 | [Apache Arrow](https://github.com/apache/arrow) |
| scikit-learn | 1.9.1 | BSD-3-Clause | [scikit-learn](https://github.com/scikit-learn/scikit-learn) |
| SciPy | 1.18.1 | BSD 3-Clause；distribution 中另列二进制构建可能捆绑组件的许可声明 | [SciPy](https://github.com/scipy/scipy) |
| seaborn | 0.13.2 | BSD 3-Clause（随包 `LICENSE.md`） | [seaborn](https://github.com/mwaskom/seaborn) |
| Streamlit | 1.64.0 | Apache-2.0 | [Streamlit](https://github.com/streamlit/streamlit) |
| Plotly | 7.1.0 | MIT | [Plotly.py](https://github.com/plotly/plotly.py) |

## 范围说明

- 上述包由运行者依 `requirements.txt` 从各自上游渠道安装；本源码候选不携带或重新分发这些包的源代码、wheel 或二进制文件。SciPy、NumPy 等发行包可能包含额外第三方组件，其随发行包提供的许可声明仍适用。
- 这是依赖来源与许可索引，不替代各上游完整许可文本，也不构成法律意见。
- OpenAI ChatGPT/Codex 的使用方式另见 [`AI_AND_CODE_PROVENANCE.md`](AI_AND_CODE_PROVENANCE.md)；本项目未分发这些服务的模型权重或客户端凭证。
- 本文件不为 NeuroChip Copilot 自有源代码授予任何许可证。是否对项目代码选择并附加 MIT、Apache-2.0 或其他许可证，须由代码权利人及团队确认；公开可访问本身不等于授予复制、修改或再分发权。
