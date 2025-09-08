import os
import streamlit as st
import pandas as pd
from openai import OpenAI
from Config import client

from Utils.Privacy import redact

# ---- Caching loaders ----
@st.cache_resource
def load_text_generator():
    from transformers import pipeline
    # smaller than gpt2
    return pipeline("text-generation", model="distilgpt2")

@st.cache_resource
def load_finbert():
    from Sentiment.Zero_shot import load_finbert_pipeline
    return load_finbert_pipeline()

@st.cache_resource
def lazy_index():
    # loaded on demand
    from Rag.Retriever import load_index
    try:
        return load_index()
    except Exception:
        return None

# Sidebar toggle
with st.sidebar:
    st.header("Privacy")
    PRIVACY_ON = st.toggle("Enable privacy redaction", value=True,
                           help="Masks emails, account numbers, phone numbers before any model call.")

def apply_privacy(label: str, text: str) -> str:
    """
    Redact text if privacy is ON. 
    Shows a subtle note if something was changed.
    """
    if not text:
        return text
    red = redact(text) if PRIVACY_ON else text
    if red != text:
        st.caption(f"🔒 {label}: redacted for privacy.")
    return red

def get_openai_client():
    # Try Streamlit secrets first, then environment
    api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    if not api_key:
        raise RuntimeError(
            "OpenAI API key not found. Set OPENAI_API_KEY in environment or .streamlit/secrets.toml"
        )
    return OpenAI(api_key=api_key)

def chat_completion(messages, model="gpt-4o-mini", temperature=0.7, max_tokens=400):
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=messages,
    )
    return resp.choices[0].message.content



st.set_page_config(page_title="Banking Copilot – MVP", layout="wide")
st.title("💼 Banking Copilot — Level A (MVP)")

tabs = st.tabs(["Prompt Playground", "Q&A over Docs (RAG)", "Sentiment Analysis", "NL→SQL (stub)"])

# ---------- TAB 1: Prompt Playground ----------
with tabs[0]:
    st.header("Prompt Playground (ChatGPT via OpenAI API)")

    # Controls
    col_top = st.columns(3)
    with col_top[0]:
        model_name = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"])
    with col_top[1]:
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05)
    with col_top[2]:
        max_tokens = st.slider("Max tokens", 64, 2000, 400, 32)

    # System prompt (optional; keeps answers concise + bank tone)
    system_prompt = st.text_area(
        "System prompt (optional)",
        value="You are a helpful banking research assistant. Be concise, accurate, and cite if context is provided.",
        height=80
    )

    # Two columns: Free-form (left) and Templates (right)
    colA, colB = st.columns(2)

    # -------- Free-form ----------
    with colA:
        st.subheader("Free-form")
        user_prompt = st.text_area(
            "Enter a prompt",
            value="Summarize Basel III in 2 bullet points."
        )
        if st.button("Generate with ChatGPT", key="openai_free"):
            try:
                safe_user = apply_privacy("Prompt", user_prompt)
                msgs = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": safe_user}]
                out = chat_completion(
                    msgs, model=model_name, temperature=temperature, max_tokens=max_tokens
                )
                st.write(out)
            except Exception as e:
                st.error(f"OpenAI call failed: {e}")

    # -------- Templates ----------
    with colB:
        st.subheader("Templates")
        base_text = st.text_area(
            "Paste text to transform",
            value="The bank improved its CET1 ratio and reduced risk-weighted assets."
        )
        style = st.selectbox("Style", ["Executive bullets", "Explain like I'm five", "Risks only"])
        if st.button("Apply Template with ChatGPT", key="openai_templ"):
            try:
                if style == "Executive bullets":
                    prompt = f"Summarize this in exactly 3 concise executive bullet points:\n\n{base_text}"
                elif style == "Explain like I'm five":
                    prompt = f"Explain this simply, like I'm five. Avoid jargon:\n\n{base_text}"
                else:
                    prompt = f"List only the risks mentioned in the following text. Do not add anything not present:\n\n{base_text}"
                safe_prompt = apply_privacy("Prompt", prompt)
                msgs = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": safe_prompt}]
                out = chat_completion(
                    msgs, model=model_name, temperature=temperature, max_tokens=max_tokens
                )
                st.write(out)
            except Exception as e:
                st.error(f"OpenAI call failed: {e}")

# ---------- TAB 2: Q&A over Docs (RAG) ----------
with tabs[1]:
    st.header("RAG: Ask questions about your PDFs")
    st.caption("Put PDFs in **data/filings/**. Click 'Build/Refresh Index' after adding files.")

    col1, col2 = st.columns([1,2])
    with col1:
        if st.button("Build / Refresh Index"):
            from rag.ingest import build_index
            try:
                path, n = build_index()
                st.success(f"Index built: {path} — {n} chunks")
                st.cache_resource.clear()
            except Exception as e:
                st.error(str(e))

        idx = lazy_index()
        if idx is None:
            st.warning("No index found yet. Add PDFs and click 'Build / Refresh Index'.")

        # NEW: relevance threshold control
        min_score = st.slider(
            "Minimum relevance (cosine similarity)",
            min_value=0.0, max_value=1.0, value=0.45, step=0.05,
            help="Higher = stricter matching. If no result exceeds this, the app will refuse to answer."
        )

    with col2:
        q = st.text_input("Your question", value="What is the Tier 1 capital ratio requirement?")
        if st.button("Search", disabled=(lazy_index() is None)):
            from Rag.Retriever import cosine_topk
            q_safe = apply_privacy("Question", q)
            results = cosine_topk(q_safe, lazy_index(), k=3, min_score=min_score)

            if not results:
                st.info("🙅 No sufficiently relevant information found in the documents (above the threshold). "
                        "Try rephrasing your question or lowering the relevance threshold.")
            else:
                st.write("**Top results (with citations):**")
                for r in results:
                    shown_text = apply_privacy("Context", r["text"])
                    st.markdown(f"> {shown_text}\n\n— *{r['source']}*, page {r['page']}  (score: {r['score']:.3f})")

                top = results[0]
                st.success(f"**Answer (from documents):** {top['text'][:400]} ...  \n\n*Cited:* {top['source']} p.{top['page']}")

# ---------- TAB 3: Sentiment ----------
with tabs[2]:
    st.header("Finance Sentiment (FinBERT)")
    text = st.text_area("Paste news/headline", value="Inflation hits a new high, hurting markets.")
    if st.button("Classify sentiment"):
        nlp = load_finbert()
        from Sentiment.Zero_shot import classify_sentiment
        label, score = classify_sentiment(nlp, text)
        st.write(f"**Label:** {label} — **Confidence:** {score:.3f}")


# ---------- TAB 4: NL→SQL (Level B) ----------
with tabs[3]:
    st.header("NL→SQL (natural language → safe SQL on SQLite)")

    st.caption("DB: data/transactions.db  •  Table: transactions(id, ts, amount, ccy, counterparty, book)")
    user_q = st.text_input("Ask a database question", value="Show the total USD amount per counterparty.")

    col = st.columns(3)
    with col[0]:
        model_name = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"], index=0, key="nl2sql_model")

    with col[1]:
        run_btn = st.button("Generate & Run SQL")
    with col[2]:
        st.caption("Guardrails: SELECT-only, LIMIT added, DDL/DML blocked.")

    if run_btn:
        from Nl2sql.Generate_sql import nl2sql_run
        safe_query = apply_privacy("NL Query", user_q)
        sql, df, err = nl2sql_run(safe_query, model=model_name)

        if sql:
            st.subheader("Generated SQL")
            st.code(sql, language="sql")
        if err:
            st.error(err)
        elif df is not None:
            st.subheader("Results")
            st.dataframe(df)

