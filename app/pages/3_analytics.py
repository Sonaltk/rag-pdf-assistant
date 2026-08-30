import streamlit as st
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ─────────────────────────────────────────
# Page config
# ─────────────────────────────────────────
st.set_page_config(
    page_title = "Analytics — RAG Assistant",
    page_icon  = "📊",
    layout     = "wide"
)

# ─────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────
if "query_logs"   not in st.session_state:
    st.session_state.query_logs   = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chunk_stats"  not in st.session_state:
    st.session_state.chunk_stats  = None

# ─────────────────────────────────────────
# Header
# ─────────────────────────────────────────
st.title("📊 Analytics Dashboard")
st.markdown(
    "Track pipeline performance, confidence scores, "
    "and RAGAS evaluation metrics."
)
st.divider()


# ─────────────────────────────────────────
# Section 1 — Session Summary
# ─────────────────────────────────────────
st.markdown("### 🔢 Session Summary")

logs = st.session_state.query_logs

if logs:
    total_queries = len(logs)
    avg_time      = round(sum(l["response_ms"] for l in logs) / total_queries)
    high_conf     = sum(1 for l in logs if l.get("confidence") == "HIGH")
    med_conf      = sum(1 for l in logs if l.get("confidence") == "MEDIUM")
    low_conf      = sum(1 for l in logs if l.get("confidence") == "LOW")
    phase2_count  = sum(1 for l in logs if l.get("pipeline") == "phase2")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Queries",    total_queries)
    c2.metric("Avg Response",     f"{avg_time}ms")
    c3.metric("🟢 High Conf",     high_conf)
    c4.metric("🟡 Medium Conf",   med_conf)
    c5.metric("🔴 Low Conf",      low_conf)
else:
    st.info("No queries yet — go to the Chat page and ask some questions!")


# ─────────────────────────────────────────
# Section 2 — Response Time Chart
# ─────────────────────────────────────────
st.divider()
st.markdown("### ⏱️ Response Time per Query")

if logs:
    try:
        import plotly.graph_objects as go

        query_labels = [f"Q{i+1}" for i in range(len(logs))]
        times        = [l["response_ms"] for l in logs]
        colors       = []
        for l in logs:
            conf = l.get("confidence", "")
            if conf == "HIGH":
                colors.append("#6ee7b7")
            elif conf == "MEDIUM":
                colors.append("#fcd34d")
            else:
                colors.append("#fca5a5")

        fig = go.Figure()

        # Bar chart for response times
        fig.add_trace(go.Bar(
            x           = query_labels,
            y           = times,
            marker_color = colors,
            text         = [f"{t}ms" for t in times],
            textposition = "outside",
            name         = "Response Time"
        ))

        # Average line
        avg = sum(times) / len(times)
        fig.add_hline(
            y           = avg,
            line_dash   = "dash",
            line_color  = "#7c83fd",
            annotation_text = f"Avg: {round(avg)}ms"
        )

        fig.update_layout(
            plot_bgcolor  = "#0e1117",
            paper_bgcolor = "#0e1117",
            font_color    = "#e5e7eb",
            xaxis_title   = "Query",
            yaxis_title   = "Response Time (ms)",
            showlegend    = False,
            height        = 350,
            margin        = dict(l=40, r=40, t=20, b=40)
        )
        fig.update_xaxes(gridcolor="#1e2130")
        fig.update_yaxes(gridcolor="#1e2130")

        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "🟢 Green = HIGH confidence | "
            "🟡 Yellow = MEDIUM | "
            "🔴 Red = LOW | "
            "─── Average response time"
        )

    except ImportError:
        # Fallback if plotly not installed
        st.bar_chart(
            {f"Q{i+1}": l["response_ms"] for i, l in enumerate(logs)}
        )
else:
    st.info("Ask questions in the Chat page to see response times here.")


# ─────────────────────────────────────────
# Section 3 — Confidence Distribution
# ─────────────────────────────────────────
st.divider()
col_conf, col_pipeline = st.columns(2, gap="large")

with col_conf:
    st.markdown("### 🎯 Confidence Distribution")

    if logs:
        try:
            import plotly.graph_objects as go

            high = sum(1 for l in logs if l.get("confidence") == "HIGH")
            med  = sum(1 for l in logs if l.get("confidence") == "MEDIUM")
            low  = sum(1 for l in logs if l.get("confidence") == "LOW")

            fig = go.Figure(go.Pie(
                labels = ["HIGH", "MEDIUM", "LOW"],
                values = [high, med, low],
                marker_colors = ["#6ee7b7", "#fcd34d", "#fca5a5"],
                hole   = 0.4,
                textinfo = "label+percent"
            ))
            fig.update_layout(
                plot_bgcolor  = "#0e1117",
                paper_bgcolor = "#0e1117",
                font_color    = "#e5e7eb",
                height        = 300,
                showlegend    = False,
                margin        = dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

        except ImportError:
            high = sum(1 for l in logs if l.get("confidence") == "HIGH")
            med  = sum(1 for l in logs if l.get("confidence") == "MEDIUM")
            low  = sum(1 for l in logs if l.get("confidence") == "LOW")
            st.write(f"🟢 HIGH: {high} | 🟡 MEDIUM: {med} | 🔴 LOW: {low}")
    else:
        st.info("No data yet.")

with col_pipeline:
    st.markdown("### 🔀 Pipeline Usage")

    if logs:
        try:
            import plotly.graph_objects as go

            p1 = sum(1 for l in logs if l.get("pipeline") == "phase1")
            p2 = sum(1 for l in logs if l.get("pipeline") == "phase2")

            fig = go.Figure(go.Pie(
                labels       = ["Phase 1", "Phase 2"],
                values       = [p1, p2],
                marker_colors = ["#7c83fd", "#f59e0b"],
                hole         = 0.4,
                textinfo     = "label+percent"
            ))
            fig.update_layout(
                plot_bgcolor  = "#0e1117",
                paper_bgcolor = "#0e1117",
                font_color    = "#e5e7eb",
                height        = 300,
                showlegend    = False,
                margin        = dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

        except ImportError:
            p1 = sum(1 for l in logs if l.get("pipeline") == "phase1")
            p2 = sum(1 for l in logs if l.get("pipeline") == "phase2")
            st.write(f"Phase 1: {p1} | Phase 2: {p2}")
    else:
        st.info("No data yet.")


# ─────────────────────────────────────────
# Section 4 — RAGAS Scores Comparison
# ─────────────────────────────────────────
st.divider()
st.markdown("### 📈 RAGAS Evaluation — Phase 1 vs Phase 2")

scores_path = Path("phase3_evaluation/eval/scores.json")

if scores_path.exists():
    with open(scores_path) as f:
        scores = json.load(f)

    p1 = scores.get("phase1", {})
    p2 = scores.get("phase2", {})

    metrics = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall"
    ]
    labels = [
        "Faithfulness",
        "Answer Relevancy",
        "Context Precision",
        "Context Recall"
    ]

    p1_scores = [p1.get(m, 0) for m in metrics]
    p2_scores = [p2.get(m, 0) for m in metrics]

    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name         = "Phase 1 (Vector only)",
            x            = labels,
            y            = p1_scores,
            marker_color = "#7c83fd",
            text         = [f"{s:.3f}" for s in p1_scores],
            textposition = "outside"
        ))
        fig.add_trace(go.Bar(
            name         = "Phase 2 (Hybrid + Rerank)",
            x            = labels,
            y            = p2_scores,
            marker_color = "#f59e0b",
            text         = [f"{s:.3f}" for s in p2_scores],
            textposition = "outside"
        ))

        fig.update_layout(
            barmode       = "group",
            plot_bgcolor  = "#0e1117",
            paper_bgcolor = "#0e1117",
            font_color    = "#e5e7eb",
            yaxis_range   = [0, 1.1],
            yaxis_title   = "Score (0-1)",
            height        = 400,
            legend        = dict(
                bgcolor     = "#1e2130",
                bordercolor = "#2d3250",
                borderwidth = 1
            ),
            margin = dict(l=40, r=40, t=20, b=40)
        )
        fig.update_xaxes(gridcolor="#1e2130")
        fig.update_yaxes(gridcolor="#1e2130")

        st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        for label, p1s, p2s in zip(labels, p1_scores, p2_scores):
            delta = round((p2s - p1s) * 100, 1)
            st.metric(label, f"{p2s:.4f}", f"{delta:+.1f}% vs Phase 1")

    # Show improvement table
    st.markdown("#### 📋 Improvement Summary")
    col1, col2, col3, col4 = st.columns(4)
    for col, label, p1s, p2s in zip(
        [col1, col2, col3, col4], labels, p1_scores, p2_scores
    ):
        delta = round((p2s - p1s) * 100, 1)
        col.metric(
            label    = label,
            value    = f"{p2s:.4f}",
            delta    = f"{delta:+.1f}% vs Phase 1"
        )
else:
    st.info(
        "RAGAS scores not found. "
        "Run `python phase3_evaluation/main.py report` to generate."
    )


# ─────────────────────────────────────────
# Section 5 — Recent Queries Table
# ─────────────────────────────────────────
st.divider()
st.markdown("### 📋 Recent Queries")

if st.session_state.chat_history:
    for i, entry in enumerate(
        reversed(st.session_state.chat_history[-10:]), 1
    ):
        conf  = entry.get("confidence", "")
        icons = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}
        icon  = icons.get(conf, "⚪")

        with st.expander(
            f"Q{len(st.session_state.chat_history) - i + 1}: "
            f"{entry['question'][:60]}... "
            f"{icon} {conf} | "
            f"{entry.get('response_ms', 0)}ms",
            expanded=False
        ):
            st.markdown(f"**Question:** {entry['question']}")
            st.markdown(f"**Pipeline:** {entry.get('pipeline', '').upper()}")
            st.markdown(f"**Confidence:** {icon} {conf}")
            st.markdown(f"**Response time:** {entry.get('response_ms', 0)}ms")
            st.divider()
            st.markdown(entry["answer"][:500] + "...")
else:
    st.info("No queries yet — ask questions in the Chat page.")


# ─────────────────────────────────────────
# Section 6 — Document Info
# ─────────────────────────────────────────
st.divider()
st.markdown("### 📄 Current Document Stats")

if st.session_state.chunk_stats:
    s = st.session_state.chunk_stats
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("PDF",           s.get("pdf_name", "unknown"))
    c2.metric("Pages",         s.get("pages", 0))
    c3.metric("Text Chunks",   s.get("text_chunks", 0))
    c4.metric("Figure Chunks", s.get("figure_chunks", 0))
    c5.metric("Total Vectors", s.get("total_chunks", 0))

    # Chunk breakdown visual
    text_pct   = round(
        s.get("text_chunks", 0) / max(s.get("total_chunks", 1), 1) * 100
    )
    figure_pct = 100 - text_pct

    st.markdown(f"""
    <div style='background:#1e2130;border-radius:8px;
                padding:1rem;margin-top:0.5rem'>
        <p style='color:#9ca3af;font-size:0.85rem;margin:0 0 0.5rem 0'>
            Chunk composition
        </p>
        <div style='display:flex;height:8px;border-radius:4px;overflow:hidden'>
            <div style='width:{text_pct}%;background:#7c83fd'></div>
            <div style='width:{figure_pct}%;background:#f59e0b'></div>
        </div>
        <div style='display:flex;gap:1rem;margin-top:0.4rem'>
            <span style='color:#7c83fd;font-size:0.75rem'>
                ■ Text {text_pct}%
            </span>
            <span style='color:#f59e0b;font-size:0.75rem'>
                ■ Figures {figure_pct}%
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("No document loaded — go to Upload page first.")