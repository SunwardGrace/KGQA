import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import streamlit.components.v1 as components
import yaml
import requests
import json
from ui.viz import get_html_content

st.set_page_config(page_title="KGQA - 中文医疗知识图谱问答", page_icon="🏥", layout="wide")

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


@st.cache_resource
def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_api_url():
    config = load_config()
    api_cfg = config.get("api", {})
    host = api_cfg.get("host", "localhost")
    if host == "0.0.0.0":
        host = "localhost"
    port = api_cfg.get("port", 8000)
    return f"http://{host}:{port}"


def call_ask_api(query: str, top_k: int = 10, mode: str = "smart") -> dict:
    url = f"{get_api_url()}/api/ask"
    try:
        response = requests.post(url, json={"query": query, "top_k": top_k, "mode": mode}, timeout=60)
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def call_eval_generate(samples_per_type: int = 50) -> dict:
    url = f"{get_api_url()}/api/eval/generate"
    try:
        response = requests.post(url, json={"samples_per_type": samples_per_type}, timeout=120)
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def call_eval_run() -> dict:
    url = f"{get_api_url()}/api/eval/run"
    try:
        response = requests.post(url, json={}, timeout=300)
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def call_eval_report() -> dict:
    url = f"{get_api_url()}/api/eval/report"
    try:
        response = requests.get(url, timeout=10)
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def call_stats() -> dict:
    url = f"{get_api_url()}/api/stats"
    try:
        response = requests.get(url, timeout=10)
        return response.json()
    except Exception as e:
        return {"error": str(e)}


st.title("KGQA - 中文医疗知识图谱问答系统")

st.warning("**免责声明**: 本系统仅用于知识检索与学习参考，不构成医疗建议。如有健康问题，请咨询专业医生。")

tab1, tab2, tab3, tab4 = st.tabs(["问答", "评测", "统计", "管理"])

with tab1:
    st.header("智能问答")
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        query = st.text_input("请输入您的问题:", placeholder="例如：糖尿病有哪些症状？")
    with col2:
        top_k = st.number_input("返回数量", min_value=1, max_value=50, value=10)
    with col3:
        mode = st.selectbox("回答模式", ["smart", "facts_only"],
                            format_func=lambda x: "智能回答" if x == "smart" else "仅图谱结果")
    if st.button("提问", type="primary"):
        if query:
            with st.spinner("正在查询..."):
                result = call_ask_api(query, top_k, mode)
            if "error" in result:
                st.error(f"查询失败: {result['error']}")
            else:
                st.subheader("查询结果")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"**识别意图**: `{result.get('intent', 'unknown')}`")
                with col_b:
                    st.markdown(f"**响应时间**: `{result.get('latency_ms', 0)}ms`")
                if result.get("parsed_entities"):
                    entities = result["parsed_entities"]
                    st.markdown("**识别实体**:")
                    for e in entities:
                        st.markdown(f"- {e['text']} → {e['canonical']} ({e['type']})")

                # 显示 LLM 智能回答
                llm_answer = result.get("llm_answer", "")
                if llm_answer:
                    st.subheader("🤖 智能回答")
                    st.markdown(llm_answer)
                    st.divider()

                st.subheader("答案列表")
                answers = result.get("answers", [])
                if answers:
                    for i, ans in enumerate(answers, 1):
                        conflict_mark = " ⚠️" if ans.get("conflict") else ""
                        score = ans.get("score", 0)
                        sources = ", ".join(ans.get("source", [])) or "unknown"
                        st.markdown(f"{i}. **{ans['text']}**{conflict_mark} (score: {score:.2f}, source: {sources})")
                else:
                    st.info("未找到相关答案")
                if result.get("warnings"):
                    for w in result["warnings"]:
                        if "冲突" in w:
                            st.warning(w)
                subgraph = result.get("subgraph", {})
                if subgraph.get("nodes"):
                    st.subheader("知识图谱可视化")
                    html_content = get_html_content(subgraph, height="500px")
                    if html_content:
                        components.html(html_content, height=520, scrolling=True)
                    with st.expander("查看子图数据"):
                        st.json(subgraph)
                with st.expander("查看Cypher查询"):
                    st.code(result.get("cypher", ""), language="cypher")

with tab2:
    st.header("自动评测 (Auto Eval)")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("生成评测集")
        samples = st.number_input("每类样本数", min_value=10, max_value=200, value=50)
        if st.button("生成评测集"):
            with st.spinner("正在生成评测集..."):
                result = call_eval_generate(samples)
            if "error" in result:
                st.error(f"生成失败: {result['error']}")
            else:
                st.success(f"生成成功! 共 {result.get('samples_generated', 0)} 条样本")
    with col2:
        st.subheader("运行评测")
        if st.button("运行评测"):
            with st.spinner("正在运行评测 (可能需要几分钟)..."):
                result = call_eval_run()
            if "error" in result:
                st.error(f"评测失败: {result['error']}")
            else:
                st.success(f"评测完成! 报告路径: {result.get('report_path', '')}")
    st.subheader("评测报告")
    if st.button("加载最新报告"):
        result = call_eval_report()
        if "error" in result:
            st.error(f"加载失败: {result['error']}")
        else:
            st.markdown(result.get("content", ""))

with tab3:
    st.header("图谱统计")
    if st.button("刷新统计"):
        result = call_stats()
        if "error" in result:
            st.error(f"获取失败: {result['error']}")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("节点总数", result.get("node_count", 0))
            with col2:
                st.metric("关系总数", result.get("relationship_count", 0))
            st.subheader("节点类型")
            labels = result.get("labels", [])
            st.write(", ".join(labels) if labels else "无")
            st.subheader("关系类型")
            rel_types = result.get("relationship_types", [])
            st.write(", ".join(rel_types) if rel_types else "无")

with tab4:
    st.header("知识管理")

    if "changesets" not in st.session_state:
        st.session_state.changesets = []
    if "entity_history" not in st.session_state:
        st.session_state.entity_history = []

    st.subheader("变更集管理")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**查看变更历史**")
        if st.button("加载变更集列表"):
            url = f"{get_api_url()}/api/changesets"
            try:
                response = requests.get(url, timeout=10)
                data = response.json()
                if isinstance(data, list):
                    st.session_state.changesets = data
                elif isinstance(data, dict) and "error" in data:
                    st.error(f"API错误: {data.get('error')}")
                    st.session_state.changesets = []
                else:
                    st.session_state.changesets = []
            except Exception as e:
                st.error(f"获取失败: {e}")
                st.session_state.changesets = []

        changesets = st.session_state.changesets
        if isinstance(changesets, list) and changesets:
            display_data = []
            for cs in changesets[:10]:
                if isinstance(cs, dict):
                    display_data.append({
                        "ID": str(cs.get("change_id", ""))[:8],
                        "状态": "active" if cs.get("status") == "active" else "rolled_back",
                        "原因": str(cs.get("reason", "N/A")),
                        "时间": str(cs.get("created_at", ""))[:10]
                    })
            if display_data:
                st.dataframe(display_data, use_container_width=True, hide_index=True)
            else:
                st.info("暂无变更记录")
        else:
            st.info("暂无变更记录或未加载")

    with col2:
        st.markdown("**回滚变更**")
        rollback_id = st.text_input("变更集ID", placeholder="输入要回滚的变更集ID")
        if st.button("执行回滚", type="secondary"):
            if rollback_id:
                url = f"{get_api_url()}/api/changesets/{rollback_id}/rollback"
                try:
                    response = requests.post(url, timeout=30)
                    result = response.json()
                    if result.get("success"):
                        st.success(f"回滚成功，影响 {result.get('count', 0)} 条记录")
                    else:
                        st.error(f"回滚失败: {result.get('error', 'unknown')}")
                except Exception as e:
                    st.error(f"请求失败: {e}")
            else:
                st.warning("请输入变更集ID")

    st.subheader("实体历史")
    entity_name = st.text_input("实体名称", placeholder="输入要查询历史的实体名称")
    if st.button("查询实体历史"):
        if entity_name:
            url = f"{get_api_url()}/api/entity/{entity_name}/history"
            try:
                response = requests.get(url, timeout=10)
                st.session_state.entity_history = response.json()
            except Exception as e:
                st.error(f"查询失败: {e}")

    if st.session_state.entity_history:
        st.dataframe(st.session_state.entity_history, use_container_width=True)
    elif entity_name:
        st.info("暂无历史记录")

