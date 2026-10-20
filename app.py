"""JobAgent in the browser. Start it with the Start launcher, or `streamlit run app.py`.

Everything runs on this computer: the page is served on 127.0.0.1 only and the
AI is the local Ollama app. The same pipeline as orchestrator.py is used.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

import webapp  # noqa: E402

st.set_page_config(page_title="JobAgent", page_icon="💼", layout="centered")
nav = st.navigation(webapp.make_pages())
with st.sidebar:
    webapp.sidebar_status()
nav.run()
