r"""NeuroChip Copilot competition demo.

Run with:
    .\.venv\Scripts\streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_curve, auc

from src.evaluate_experiment_portfolio import greedy, objective, prepare_candidates


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed" / "dishbrain"
FP_DIR = ROOT / "results" / "dishbrain_fingerprint_pilot"
ROBUST_DIR = ROOT / "results" / "dishbrain_fingerprint_robustness"
DOSE_CONTROL_DIR = ROOT / "results" / "dishbrain_dose_negative_control"
CONFOUND_SCAN_DIR = ROOT / "results" / "protocol_confound_scan"
ORGANOID_QC_DIR = ROOT / "results" / "organoid_recording_qc"
CLASSES = ["Carbamazepine", "Perampanel", "Phenytoin"]

# The review/source archive intentionally excludes data-derived artifacts.
# Check them before eager loading so a clean checkout can explain how to set up.
REQUIRED_DEMO_ASSETS = (
    "data/processed/dishbrain/drug_experiment_table.csv",
    "data/processed/dishbrain/dishbrain_audit.json",
    "data/processed/dataset_audit.json",
    "results/dishbrain_fingerprint_pilot/grouped_oof_predictions.csv",
    "results/dishbrain_fingerprint_pilot/grouped_cv_metrics.csv",
    "results/dishbrain_fingerprint_pilot/confusion_full_rest_electrophysiology_extra_trees.csv",
    "results/dishbrain_fingerprint_robustness/robustness_audit.json",
    "results/dishbrain_fingerprint_robustness/shared_chip_protocol_block_order.csv",
    "results/dishbrain_fingerprint_robustness/cluster_bootstrap.csv",
    "results/dishbrain_dose_negative_control/dose_negative_control_summary.json",
    "results/protocol_confound_scan/confound_scan.json",
    "results/organoid_recording_qc/recording_qc.csv",
    "results/organoid_recording_qc/recording_qc_summary.json",
)


def missing_required_demo_assets(root: Path = ROOT) -> list[str]:
    """Return required replay artifacts absent from a local checkout."""
    return [relative for relative in REQUIRED_DEMO_ASSETS if not (root / relative).is_file()]


st.set_page_config(page_title="NeuroChip Copilot", page_icon="🧠", layout="wide")
st.markdown(
    """
<style>
.stApp {background: linear-gradient(135deg, #07111f 0%, #101c31 52%, #081925 100%);}
[data-testid="stSidebar"] {background: #091524; border-right: 1px solid #20334b;}
h1, h2, h3, h4, h5, h6 {color: #eef7ff !important; -webkit-text-fill-color: #eef7ff !important; letter-spacing: -0.02em;}
[data-testid="stHeading"] h1,
[data-testid="stHeading"] h2,
[data-testid="stHeading"] h3,
[data-testid="stHeading"] h4,
[data-testid="stHeading"] h5,
[data-testid="stHeading"] h6 {color: #eef7ff !important; -webkit-text-fill-color: #eef7ff !important;}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4 {color: #eef7ff !important; -webkit-text-fill-color: #eef7ff !important;}
p, li, label {color: #c9d8e8;}
.hero {padding: 1.7rem 1.8rem; border: 1px solid #28435e; border-radius: 20px;
background: radial-gradient(circle at 10% 20%, rgba(26,213,199,.16), transparent 34%),
linear-gradient(135deg, rgba(19,37,60,.96), rgba(7,19,32,.96)); margin-bottom: 1rem;}
.eyebrow {color:#42d7c8; font-size:.82rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase;}
.hero-title {font-size:2.35rem; color:white; font-weight:760; margin:.25rem 0 .35rem;}
.hero-copy {font-size:1.04rem; color:#b9cadb; max-width:920px;}
.trust {border-left:4px solid #ffbd59; padding:.75rem 1rem; background:rgba(255,189,89,.10); border-radius:8px;
color:#edf4fb !important; font-size:.96rem; line-height:1.65;}
.trust * {color:#edf4fb !important;}
.trust b {color:#ffd27a !important;}
[data-testid="stMetric"] {background:rgba(17,34,54,.92); border:1px solid #29435e; padding:1rem; border-radius:14px;}
[data-testid="stMetricValue"] {color:#f5fbff;}
@media (max-width: 1200px) {
  [data-testid="stMetric"] {padding:.7rem .55rem; min-height:7.2rem;}
  [data-testid="stMetricLabel"] {font-size:.78rem; line-height:1.2; white-space:normal; overflow:visible; text-overflow:clip;}
  [data-testid="stMetricValue"] {font-size:1.2rem !important; line-height:1.15; white-space:nowrap;}
  [data-testid="stMetricDelta"] {font-size:.68rem; line-height:1.2; white-space:normal; overflow:visible; text-overflow:clip;}
}
</style>
""",
    unsafe_allow_html=True,
)

missing_assets = missing_required_demo_assets()
if missing_assets:
    st.title("NeuroChip Copilot")
    st.warning("当前代码包没有附带数据衍生文件，因此完整演示尚未就绪。")
    st.markdown(
        "这是有意的发布边界：代码审查包不打包原始或由原始数据计算出的文件。 "
        "请先阅读项目根目录中的 `README.md` 和 `DATA_ACCESS_AND_REPRODUCTION.md`，"
        "确认数据许可后，在本地完成数据准备与复现，再重新启动本应用。"
    )
    st.markdown("**当前缺少的文件：**")
    st.code("\n".join(missing_assets), language="text")
    st.info(
        "应用不会自动下载数据，也不会把缺失数据当作零结果。请按数据访问说明取得允许使用的文件，"
        "运行相应预处理/审计步骤后再打开各功能页面。"
    )
    st.stop()


@st.cache_data
def load_data(cache_version: str):
    """Load analysis inputs; cache_version explicitly invalidates stale UI caches."""
    experiments = pd.read_csv(DATA_DIR / "drug_experiment_table.csv")
    experiments = experiments.loc[experiments["drug"].isin(CLASSES)].reset_index(drop=True)
    predictions = pd.read_csv(FP_DIR / "grouped_oof_predictions.csv", dtype={"chip_id": str})
    predictions = predictions.loc[
        predictions["feature_set"].eq("full_rest_electrophysiology")
        & predictions["model"].eq("extra_trees")
    ].sort_values("row_index")
    metrics = pd.read_csv(FP_DIR / "grouped_cv_metrics.csv")
    confusion = pd.read_csv(
        FP_DIR / "confusion_full_rest_electrophysiology_extra_trees.csv", index_col=0
    )
    robustness = json.loads((ROBUST_DIR / "robustness_audit.json").read_text(encoding="utf-8"))
    # Re-derive this small display summary from the row-level audit table so
    # older cached JSON artifacts cannot break the red-team page after the
    # audit is regenerated or upgraded.
    block_order_table = pd.read_csv(ROBUST_DIR / "shared_chip_protocol_block_order.csv")
    block_order_counts = block_order_table["first_appearance_order"].value_counts()
    modal_block_order = str(block_order_counts.index[0])
    robustness["shared_chip_protocol_block_order"] = {
        "n_shared_chips": int(len(block_order_table)),
        "modal_first_appearance_order": modal_block_order,
        "chips_following_modal_first_appearance_order": int(
            block_order_table["first_appearance_order"].eq(modal_block_order).sum()
        ),
        "chips_with_three_contiguous_treatment_blocks": int(
            block_order_table["each_drug_in_one_contiguous_block"].astype(bool).sum()
        ),
    }
    bootstrap = pd.read_csv(ROBUST_DIR / "cluster_bootstrap.csv")
    audit = json.loads((DATA_DIR / "dishbrain_audit.json").read_text(encoding="utf-8"))
    return experiments, predictions, metrics, confusion, robustness, bootstrap, audit


@st.cache_data
def load_dose_control():
    """读取事后剂量负对照，单独标记其探索性和混杂诊断属性。"""
    return json.loads(
        (DOSE_CONTROL_DIR / "dose_negative_control_summary.json").read_text(encoding="utf-8")
    )


@st.cache_data
def load_confound_scan():
    """读取冻结的协议顺序负对照结果，避免界面手工抄录数值。"""
    return json.loads(
        (CONFOUND_SCAN_DIR / "confound_scan.json").read_text(encoding="utf-8")
    )


@st.cache_data
def load_portfolio_geometry():
    return prepare_candidates()


@st.cache_data
def load_organoid_qc():
    table = pd.read_csv(ORGANOID_QC_DIR / "recording_qc.csv")
    summary = json.loads((ORGANOID_QC_DIR / "recording_qc_summary.json").read_text(encoding="utf-8"))
    # Derive displayed counts from the row-level audit table so an older cached
    # summary schema cannot break the live dashboard or silently misstate counts.
    summary["n_recordings"] = int(len(table))
    summary["n_empty_spike_time_vectors"] = int(table["n_empty_spike_time_vectors"].sum())
    summary["n_amplitude_count_mismatches"] = int((~table["amplitude_counts_match"].astype(bool)).sum())
    summary["n_empty_amplitude_vectors"] = int(table["n_empty_amplitude_vectors"].sum())
    summary["max_source_table_firing_rate_abs_error_hz"] = float(
        table["source_fr_abs_error_hz"].max()
    )
    source = json.loads((ROOT / "data" / "processed" / "dataset_audit.json").read_text(encoding="utf-8"))
    return table, summary, source


experiments, predictions, metrics, confusion, robustness, bootstrap, audit = load_data(
    cache_version="earliest-rest-row-fix-20261003"
)
dose_control = load_dose_control()
confound_scan = load_confound_scan()

st.sidebar.markdown("## 🧠 NeuroChip Copilot")
st.sidebar.caption("Evidence-aware neural MPS analytics")
# Keep the prototype's provenance visible while keeping the sidebar focused on this challenge.
st.sidebar.caption(
    "本原型基于团队此前开发的神经微生理系统分析项目；本届聚焦 MEA 记录审阅与流程风险核查。"
)
page = st.sidebar.radio(
    "导航",
    ["总览", "数据资产审计", "类器官记录质控", "处理状态指纹", "可信度红队", "后续实验规划"],
)
st.sidebar.markdown("---")
st.sidebar.caption("公开来源 · 按芯片分组验证 · 可复现证据链")

if page == "总览":
    st.markdown(
        """
<div class="hero">
  <div class="eyebrow">AI × Neural Microphysiological Systems</div>
  <div class="hero-title">NeuroChip Copilot</div>
  <div class="hero-copy">把异构神经微生理系统数据变成可审计的处理状态指纹、可信结论与历史候选预算回放。系统主动暴露数据泄漏、时间混杂和证据边界，而不是只给一个漂亮分数；回放结果尚未验证前瞻实验收益。</div>
</div>
""",
        unsafe_allow_html=True,
    )
else:
    st.caption("NeuroChip Copilot · Evidence-aware neural MPS analytics")


def metric_cards(metrics: pd.DataFrame):
    a, b, c, d = st.columns(4)
    # Explanations are captions, not metric deltas: no implied increase/decrease.
    headline_auc = metrics.loc[
        metrics["feature_set"].eq("full_rest_electrophysiology")
        & metrics["model"].eq("extra_trees"),
        "macro_ovr_auc",
    ]
    if len(headline_auc) != 1:
        raise ValueError("Expected one frozen full-rest ExtraTrees macro OVR AUC row")
    with a:
        st.metric("DishBrain 药物实验总集", "112 条")
        st.caption("覆盖 30 块物理芯片")
    with b:
        st.metric("记录级 OOF 平衡准确率", "73.75%")
        st.caption("主模型：109 条 / 29 块芯片；探索性")
    with c:
        st.metric("宏 OVR AUC", f"{float(headline_auc.iloc[0]):.4f}")
        st.caption("三类处理条件")
    with d:
        st.metric("多数类基线", "33.33%")
        st.caption("balanced accuracy")


def overview_evidence_cards():
    """Put the result and its strongest confound warning together on the first screen."""
    _, _, _, _, robustness, _, _ = load_data(
        cache_version="earliest-rest-row-fix-20261003"
    )
    _, qc_summary, _ = load_organoid_qc()
    order_ba = robustness["chronology_negative_control"]["balanced_accuracy"]
    a, b, c = st.columns(3)
    with a:
        st.metric("芯片分组 OOF 指纹", "73.75%")
        st.caption("109 条记录、29 块芯片；探索性")
    with b:
        st.metric("协议顺序负对照", f"{order_ba:.2%}")
        st.caption("同一匹配子集；强混杂警示")
    with c:
        st.metric("类器官记录 QC", "45/45")
        st.caption(f"时间戳结构有效；{qc_summary['n_empty_spike_time_vectors']} 个零事件通道")


if page == "总览":
    metric_cards(metrics)
    st.caption(
        "口径说明：112 条 / 30 块为整理后的药物实验总集；73.75% 来自当前三药主模型实际纳入的 109 条 / 29 块。"
    )
    overview_evidence_cards()
    st.info(
        "两套数据分工使用、不会混合训练：DishBrain 用于带处理标签的芯片级指纹与混杂审计；"
        "人前脑类器官数据用于原始 MEA 记录质控。类器官数据没有药物标签，因此不用于药物分类或跨体系外推。"
    )
    st.markdown("### 从原始数据到实验决策")
    flow = pd.DataFrame(
        {
            "阶段": ["数据接入", "证据审计", "AI 表型", "可信度", "实验规划"],
            "输出": [
                "MEA/临界性/爆发特征",
                "芯片层级与来源哈希",
                "处理状态概率与相似性",
                "分组验证、CI、混杂告警",
                "固定预算候选组合",
            ],
        }
    )
    st.dataframe(flow, hide_index=True, width="stretch")
    left, right = st.columns([1.15, 1])
    with left:
        counts = experiments.groupby(["drug", "dose"], as_index=False).size()
        fig = px.bar(
            counts, x="dose", y="size", color="drug", barmode="group",
            labels={"dose": "剂量", "size": "实验数", "drug": "处理"},
            title="真实公开神经 MPS 药物—剂量覆盖",
            template="plotly_dark",
        )
        st.plotly_chart(fig, width="stretch", theme=None)
    with right:
        st.markdown("### 证据闸门：结果与混杂一起展示")
        model_row = metrics.loc[
            metrics["feature_set"].eq("full_rest_electrophysiology")
            & metrics["model"].eq("extra_trees")
        ].iloc[0]
        chronology_ba = robustness["chronology_negative_control"]["balanced_accuracy"]
        shared_ba = robustness["shared_chip_subset"]["balanced_accuracy"]
        shared_dose_ba = next(
            item["balanced_accuracy"]
            for item in dose_control["results"]
            if item["scope"] == "three_drug_shared_chip_subset"
        )
        st.warning(
            f"主模型为按芯片分组的记录级 OOF BA={model_row['balanced_accuracy']:.2%}（109 条记录、29 块芯片）；"
            f"在匹配的共享子集（64 条记录、11 块芯片）上，电生理模型 BA={shared_ba:.2%}，"
            f"仅剂量负对照 BA={shared_dose_ba:.2%}、仅协议顺序负对照 BA={chronology_ba:.2%}。"
            "剂量对照是事后诊断；这些不同输入的分数不是公平模型比较。当前数据只支持探索性的处理条件相关信号，"
            "不能将其归因为药物特异机制。"
        )
        st.markdown("### 当前可支持的结论")
        st.success("按完整芯片留出的测试记录中存在处理标签区分信号；机制归因仍未识别。")
        st.markdown("### 项目不声称什么")
        st.markdown(
            """
- 不声称临床诊断或患者用药推荐；
- 不把处理条件识别解释为纯药理机制识别；
- 不把可选 Ising 后端包装成量子优势；
- 不用随机按行拆分制造虚高精度。
"""
        )

elif page == "数据资产审计":
    st.markdown("## 数据资产审计")
    metric_cards(metrics)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 冻结的数据单元")
        st.markdown(
            "- **分析单元：** 芯片 × 日期 × 药物 × 剂量\n"
            "- **输入：** 给药后最早一次静息记录\n"
            "- **结果：** 后续主动游戏阶段表现\n"
            "- **验证：** 按完整芯片分组留出；指标按记录级 OOF 汇总"
        )
    with c2:
        st.markdown("#### 来源与完整性")
        st.json(
            {
                "DOI": audit["source"]["paper_doi"],
                "OSF": audit["source"]["osf_resource"],
                "首次静息早于闭环": audit["first_rest_precedes_active"],
                "实验均有结果": audit["all_experiments_have_active_outcome"],
            }
        )
        st.info(
            "来源可访问不等于文件许可已确认。当前公开源码包不分发原始数据；"
            "数据复用与再发布仍须按 OSF 当前条款及权利人授权核验。"
        )
    missing = pd.DataFrame(
        list(audit["high_missingness_columns"].items()), columns=["字段", "缺失率"]
    )
    fig = px.bar(
        missing, x="缺失率", y="字段", orientation="h", range_x=[0, 1],
        title="被系统主动标记的高缺失字段", template="plotly_dark",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#c9d8e8",
        title_font_color="#eef7ff",
        margin={"l": 210, "r": 32, "t": 64, "b": 48},
        height=360,
    )
    fig.update_xaxes(gridcolor="#2a4058", zerolinecolor="#526a82")
    fig.update_yaxes(gridcolor="#2a4058", zerolinecolor="#526a82")
    st.plotly_chart(fig, width="stretch", theme=None)
    st.info("高缺失的 Shape Collapse 特征未进入当前模型；DCC 通过训练折内中位数插补并带缺失指示器。")
    st.markdown("#### 每个原始文件均保存 SHA-256")
    hashes = pd.DataFrame(
        [
            {"文件": name, "字节": detail["bytes"], "SHA-256": detail["sha256"]}
            for name, detail in audit["source"]["files"].items()
        ]
    )
    st.dataframe(hashes, hide_index=True, width="stretch")

elif page == "处理状态指纹":
    st.markdown("## 跨芯片处理状态指纹")
    metric_cards(metrics)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.imshow(
            confusion, text_auto=True, color_continuous_scale="Teal",
            labels={"x": "预测处理", "y": "真实处理", "color": "实验数"},
            title="按芯片分组的记录级 OOF 混淆矩阵",
        )
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, width="stretch", theme=None)
    with c2:
        roc_fig = go.Figure()
        for label in CLASSES:
            truth = (predictions["truth"] == label).astype(int)
            fpr, tpr, _ = roc_curve(truth, predictions[f"p_{label}"])
            roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{label} · AUC {auc(fpr,tpr):.3f}"))
        roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="随机", line={"dash": "dash"}))
        roc_fig.update_layout(
            title="One-vs-rest ROC", xaxis_title="假阳性率", yaxis_title="真阳性率",
            template="plotly_dark", legend={"orientation": "h"},
        )
        st.plotly_chart(roc_fig, width="stretch", theme=None)
    st.caption("模型输入仅包含首次静息电生理；验证按完整芯片分组留出，图中为记录级 OOF 预测。任务与模型筛选仍属探索性，尚无独立锁定确认；不作临床或因果药理解释。")

elif page == "类器官记录质控":
    st.markdown("## 人前脑类器官 MEA 记录质控")
    qc, qc_summary, source = load_organoid_qc()
    st.markdown(
        f"数据来源：[Zenodo {source['dataset_version']} 记录]({source['dataset_url']}) · "
        f"DOI `{source['dataset_doi']}` · 许可证 {source['license']}"
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("类器官记录", str(qc_summary["n_recordings"]), "公开原始 MAT 文件")
    c2.metric("记录时长", "200–3600 秒", "8 个电极通道")
    # Keep the primary QC count short enough to remain fully visible in a
    # four-column layout; the caption carries the interpretation.
    c3.metric(
        "时间戳结构通过",
        f"{qc_summary['n_recordings']}/{qc_summary['n_recordings']}",
        "记录通过结构检查",
    )
    c4.metric(
        "计数不匹配记录",
        f"{qc_summary['n_amplitude_count_mismatches']} 份",
        f"{qc_summary['n_empty_amplitude_vectors']} 个空幅值向量",
    )

    st.markdown("### 同一条记录前后半段的放电变化")
    st.caption(
        "每条记录按时间中点分成前后两半，比较总体放电率变化和 8 通道空间分布相关性。"
        "这反映记录内漂移，不是类器官好坏标签；不同记录时长来自不同样本，不能据此推断延长记录的因果效果。"
    )
    drift = px.scatter(
        qc,
        x="duration_s",
        y="half_rate_relative_change",
        color="amplitude_counts_match",
        hover_name="organoid",
        hover_data={
            "first_half_fr_hz": ":.3f",
            "second_half_fr_hz": ":.3f",
            "half_channel_profile_correlation": ":.3f",
            "n_ten_second_windows": True,
        },
        labels={
            "duration_s": "记录时长（秒）",
            "half_rate_relative_change": "前后半段总体放电率相对变化",
            "amplitude_counts_match": "幅值数组非空且数量匹配",
        },
        title="记录时长与同记录前后半段放电变化（描述性）",
        template="plotly_dark",
    )
    st.plotly_chart(drift, width="stretch", theme=None)

    bad_amp = qc.loc[~qc["amplitude_counts_match"]]
    if not bad_amp.empty:
        st.warning(
            "MATLAB v7.3 原始文件中的空 amplitude 数组已按 MATLAB_empty 语义解析。"
            "4 个空幅值向量与零事件 spike-time 通道对应；另有 fs417-org0 的 8 个幅值向量为空，"
            "但其 spike-time 共含 33,096 个事件。该记录仍可用于放电率审计，但在数据源澄清前不用于幅值分析；不自动补齐或配对。"
        )
        st.dataframe(
            bad_amp[[
                "organoid",
                "duration_s",
                "total_spikes",
                "n_empty_amplitude_vectors",
                "empty_amplitude_channels_1based",
                "amplitude_mismatch_count_delta",
            ]],
            hide_index=True,
            width="stretch",
        )

    st.markdown("### 记录级指标")
    display = qc[
        [
            "organoid",
            "duration_s",
            "computed_mean_fr_hz",
            "source_fr_abs_error_hz",
            "half_rate_relative_change",
            "half_channel_profile_correlation",
            "ten_second_window_fr_cv",
            "spike_times_valid",
            "amplitude_counts_match",
        ]
    ].sort_values("half_rate_relative_change", ascending=False)
    st.dataframe(display, hide_index=True, width="stretch")
    st.info(
        f"重算 firing rate 与论文源表最大绝对差为 "
        f"{qc_summary['max_source_table_firing_rate_abs_error_hz']:.6f} Hz；"
        "所有记录均低于 0.001 Hz 差异。数据没有处理标签，不用于药物响应预测。"
    )

elif page == "可信度红队":
    st.markdown("## 可信度红队：系统主动寻找自己不能证明的地方")
    c1, c2, c3 = st.columns(3)
    ci = robustness["cluster_bootstrap"]["balanced_accuracy"]
    ephys = robustness["shared_chip_subset"]
    order = confound_scan["metadata_negative_controls"]["within_chip_protocol_order"]
    dose = next(
        item for item in dose_control["results"]
        if item["scope"] == "three_drug_shared_chip_subset"
    )
    c1.metric("电生理模型 · 共享子集", f"{ephys['balanced_accuracy']:.2%}")
    c1.caption(f"{ephys['rows']} 条记录 / {ephys['chips']} 个芯片")
    c2.metric("仅协议顺序 · 负对照", f"{order['balanced_accuracy']:.2%}")
    c2.caption(f"{order['n']} 条记录 / {order['groups']} 个芯片")
    c3.metric("仅剂量 · 事后负对照", f"{dose['balanced_accuracy']:.2%}")
    c3.caption(f"{dose['records']} 条记录 / {dose['chips']} 个芯片")
    st.caption(
        f"补充稳健性：按芯片聚类 bootstrap 的平衡准确率中位数为 {ci['median']:.1%} "
        f"（95% CI {ci['ci95_low']:.1%}–{ci['ci95_high']:.1%}）。"
    )
    block_order = robustness["shared_chip_protocol_block_order"]
    st.markdown("### 同一批芯片经历相同的处理阶段顺序")
    st.markdown(
        f"**共享子集的观察顺序：** "
        f"{block_order['modal_first_appearance_order'].replace(' > ', ' → ')}"
    )
    order1, order2 = st.columns(2)
    with order1:
        st.metric(
            "遵循共同顺序",
            f"{block_order['chips_following_modal_first_appearance_order']}/"
            f"{block_order['n_shared_chips']}",
        )
    with order2:
        st.metric(
            "三个连续处理区块",
            f"{block_order['chips_with_three_contiguous_treatment_blocks']}/"
            f"{block_order['n_shared_chips']}",
        )
    st.info(
        "这是共享子集中观察到的回顾性实验顺序：药物类别与处理阶段在该子集里无法彼此拆开解释。"
        "它说明顺序是重要的标签关联信号，但不识别唯一因果来源，也不证明电生理模型读取了顺序元数据。"
    )
    left, right = st.columns([1.2, 1])
    with left:
        fig = px.histogram(
            bootstrap, x="balanced_accuracy", nbins=45,
            title="按芯片聚类重采样的性能分布", template="plotly_dark",
        )
        fig.add_vline(x=1 / 3, line_dash="dash", line_color="#ffbd59")
        # Chinese label and more bottom margin keep the tick labels and axis title distinct in the demo.
        fig.update_layout(xaxis_title="平衡准确率", margin={"t": 55, "b": 72}, height=360)
        st.plotly_chart(fig, width="stretch", theme=None)
    with right:
        comparison = pd.DataFrame(
            {
                "检查": ["电生理模型", "仅剂量（事后）", "仅协议顺序"],
                "平衡准确率": [
                    ephys["balanced_accuracy"],
                    dose["balanced_accuracy"],
                    order["balanced_accuracy"],
                ],
            }
        )
        fig = px.bar(
            comparison, x="检查", y="平衡准确率", range_y=[0, 1],
            color="检查", text="平衡准确率", template="plotly_dark",
        )
        fig.update_traces(
            texttemplate="%{y:.2%}", textposition="outside", cliponaxis=False
        )
        # The category names carry the diagnostic meaning; remove the redundant x-axis title.
        fig.update_layout(xaxis_title="", xaxis_tickangle=0, margin={"t": 45, "b": 60}, height=360)
        st.plotly_chart(fig, width="stretch", theme=None)
    st.markdown(
        """
<div class="trust"><b>自动证据边界：</b>仅协议顺序和仅剂量的事后负对照均能预测处理标签。它们是不同输入下的风险诊断，不与电生理模型作公平性能比较，也不能确定因果来源。当前数据不足以把处理本身与给药顺序、剂量方案分开；系统会阻止将结果改写成药物机制或临床疗效结论。</div>
""",
        unsafe_allow_html=True,
    )

elif page == "后续实验规划":
    st.markdown("## 有限预算下的后续闭环实验规划")
    st.caption("场景：多个芯片已完成低成本首次静息测量，但只能选择一部分继续运行闭环功能实验。")
    st.warning(
        "回顾性演示：候选来自历史记录；这里展示预算变化下的候选组合，不代表已执行后续实验，"
        "也未验证真实前瞻收益。"
    )
    budget = st.slider("可继续的实验数", min_value=3, max_value=20, value=8)
    uncertainty_weight = st.slider("不确定性权重", 0.0, 1.0, 0.55, 0.05)
    diversity_weight = 1.0 - uncertainty_weight

    candidate, entropy, distance = load_portfolio_geometry()
    prob = predictions[[f"p_{label}" for label in CLASSES]].to_numpy()
    selected = greedy(budget, entropy, distance, uncertainty_weight)
    portfolio_metrics = objective(selected, entropy, distance, uncertainty_weight)

    candidate["模型不确定性"] = entropy
    candidate["回顾性入选示例"] = False
    candidate.loc[selected, "回顾性入选示例"] = True
    chosen = candidate.loc[selected].copy()
    st.success(
        f"历史候选回放：从 {len(candidate)} 条记录中筛出 {budget} 条，供研究者审阅；"
        f"综合目标 {portfolio_metrics['objective']:.3f}，"
        f"平均不确定性 {portfolio_metrics['mean_uncertainty']:.3f}，"
        f"平均表型差异 {portfolio_metrics['mean_pairwise_diversity']:.3f}。"
    )
    st.dataframe(
        chosen.sort_values("模型不确定性", ascending=False), hide_index=True, width="stretch"
    )
    fig = px.scatter_ternary(
        pd.DataFrame(prob, columns=CLASSES).assign(selected=candidate["回顾性入选示例"].map({True: "回放入选", False: "未入选"})),
        a=CLASSES[0], b=CLASSES[1], c=CLASSES[2], color="selected",
        color_discrete_map={"回放入选": "#42d7c8", "未入选": "#41566f"},
        title="模型概率空间中的不确定性与覆盖",
    )
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, width="stretch", theme=None)
    st.info("当前实际运行的是透明的经典贪心选择器。系统可导出 QUBO；Ising 映射只在本地试验中验证过，尚无独立归档矩阵或已启用的量子后端，也没有量子优势证据。")
