"""
ui.py — Gradio chat interface for the single-agent legal RAG system.
Run:
    python ui.py
"""

import logging
import os
import inspect
import threading

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / "Apikey.env")
DEBUG_MODE = os.environ.get("LEGAL_RAG_DEBUG", "false").strip().lower() == "true"

# Some Conda/IDE terminals leave SSL_CERT_FILE pointing to an environment or
# certificate bundle that no longer exists. httpx reads it while Gradio is
# imported, so repair it before importing Gradio rather than disabling TLS.
configured_ca_bundle = os.environ.get("SSL_CERT_FILE")
if configured_ca_bundle and not Path(configured_ca_bundle).is_file():
    try:
        import certifi
    except ImportError:
        # With no explicit path, httpx/ssl can fall back to their normal CA
        # discovery. Keeping a known-invalid path would always crash.
        os.environ.pop("SSL_CERT_FILE", None)
    else:
        certifi_ca_bundle = Path(certifi.where())
        if certifi_ca_bundle.is_file():
            os.environ["SSL_CERT_FILE"] = str(certifi_ca_bundle)
        else:
            os.environ.pop("SSL_CERT_FILE", None)

hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token

import gradio as gr
from agent import SingleAgentRAG
import html
import re

def _plain(value):
    """Escape source metadata so it cannot introduce HTML or Markdown links."""
    return re.sub(r"([\\`*_{}\[\]()#+.!|>~-])", r"\\\1", html.escape(str(value)))


def format_answer(answer, documents):
    if not documents:
        return answer
    lines = []
    seen = set()
    for doc in documents:
        label = doc.get("citation_label") or "Source"
        source = doc.get("source") or doc.get("document_name") or "Unknown source"
        key = (label, source)
        if key in seen:
            continue
        seen.add(key)
        name = str(source).replace("\\", "/").rsplit("/", 1)[-1]
        country = doc.get("country") or ""
        lines.append(f"- **{_plain(label)}** — {_plain(name)}"
                     + (f" ({_plain(country)})" if country else ""))
        excerpt = doc.get("excerpt") or doc.get("text") or ""
        if excerpt:
            excerpt = " ".join(str(excerpt).split())
            if len(excerpt) > 500:
                excerpt = excerpt[:500].rsplit(" ", 1)[0] + "…"
            lines.append(f"\n  {_plain(excerpt)}\n")
    return answer + "\n\n---\n\n### Retrieved sources\n\n" + "\n".join(lines)


def turns_to_chatbot(turns):
    messages = []
    for turn in turns:
        messages.extend([
            {"role": "user", "content": turn["query"]},
            {"role": "assistant", "content": format_answer(
                turn["answer"], turn.get("retrieved_documents") or []
            )},
        ])
    return messages


print("Gradio version:", gr.__version__)

logging.basicConfig(level=logging.DEBUG if DEBUG_MODE else logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# INITIALISE SYSTEM
# ---------------------------------------------------------------------------

print("Building system ...")
data_dir = Path(__file__).parent / "data"
rag = SingleAgentRAG(data_dir=data_dir)
rag_lock = threading.RLock()
print("System ready.")


def _ltm_status():
    try:
        return f"{rag.ltm.stats()['qa_pairs']} saved summaries"
    except Exception:
        logger.exception("Unable to read memory status")
        return "Temporarily unavailable"


# ---------------------------------------------------------------------------
# CHAT HISTORY HELPERS
# ---------------------------------------------------------------------------

def _session_choices():
    """
    (label, value) pairs for the sidebar list, most recently updated first.
    Label shows the title and turn count, value is the session_id used to
    reload it.
    """
    with rag_lock:
        sessions = rag.history.list_sessions()
    return [
        (f"{s['title']}  ({s['turn_count']} turns)", s["session_id"])
        for s in sessions
    ]


def _turns_to_chatbot(turns_data):
    """Convert ChatHistoryStore turns into Gradio's messages format."""
    return turns_to_chatbot(turns_data)


# ---------------------------------------------------------------------------
# CHAT LOGIC
# ---------------------------------------------------------------------------

def bot_logic(user_input, history):

    # Work on a fresh list so a yielded Gradio value is not mutated later.
    history = list(history or [])

    query = user_input.strip()

    if not query:
        yield history
        return

    # Add user message
    history.append(
        {
            "role": "user",
            "content": query,
        }
    )

    # Temporary assistant message
    history.append(
        {
            "role": "assistant",
            "content": "⏳ Thinking...",
        }
    )

    yield history

    # Serialize operations because the local UI shares one active conversation.
    try:
        with rag_lock:
            result = rag.ask(query)
            answer = format_answer(result["answer"], result["retrieved_documents"])
    except Exception:
        logger.exception("Unable to answer UI request")
        answer = (
            "A temporary error prevented the request from completing. "
            "Please try again."
        )

    # Replace temporary answer
    history[-1] = {
        "role": "assistant",
        "content": answer,
    }

    yield history


def handle_submit(user_input, history):

    for updated_history in bot_logic(user_input, history):

        stm_text = f"{len(rag.stm)} turns in session"
        ltm_text = _ltm_status()

        yield (
            updated_history,
            "",
            stm_text,
            ltm_text,
            gr.update(choices=_session_choices(), value=rag.session_id),
        )


def new_session():
    """'New Session' button: really starts a new chat_history session
    (not just an in-RAM stm reset), so the previous chat stays in the
    sidebar instead of being overwritten."""
    try:
        with rag_lock:
            rag.new_session()
    except Exception:
        logger.exception("Unable to start a new chat session")
        gr.Warning("Could not start a new chat session.")
        return (
            gr.update(),
            gr.update(),
            f"{len(rag.stm)} turns in session",
            _ltm_status(),
            gr.update(),
        )

    return (
        [],                 # chatbot history
        "",                 # textbox
        "0 turns in session",
        _ltm_status(),
        gr.update(choices=_session_choices(), value=rag.session_id),
    )


def load_selected_session(session_id):
    """Sidebar click: reload a past session verbatim and make it active,
    so any follow-up question continues that conversation."""
    if not session_id:
        return gr.update(), "", ""

    try:
        with rag_lock:
            turns_data = rag.load_session(session_id)
    except Exception:
        logger.exception("Unable to load chat session %s", session_id)
        gr.Warning("The selected chat session is unavailable.")
        return (
            gr.update(),
            f"{len(rag.stm)} turns in session",
            _ltm_status(),
        )

    return (
        _turns_to_chatbot(turns_data),
        f"{len(rag.stm)} turns in session",
        _ltm_status(),
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

theme = gr.themes.Soft()
blocks_options = {"title": "Legal RAG"}
if "theme" in inspect.signature(gr.Blocks).parameters:
    blocks_options["theme"] = theme

with gr.Blocks(**blocks_options) as demo:

    gr.Markdown("# ⚖️ Single-Agent Legal RAG")

    gr.Markdown(
        "Ask questions about family law and inheritance in "
        "**Italy, Estonia, and Slovenia**."
    )

    with gr.Row():

        # ---------------------------------------------------------------
        # Sidebar
        # ---------------------------------------------------------------

        with gr.Column(scale=1):

            gr.Markdown("### 💬 Chat history")

            session_list = gr.Radio(
                choices=_session_choices(),
                value=rag.session_id,
                label=None,
                show_label=False,
                interactive=True,
            )

            new_session_btn = gr.Button("🆕 New chat")

            gr.Markdown("---")
            gr.Markdown("### 🧠 System Status")

            stm_status = gr.Textbox(
                label="Short-term memory",
                value="0 turns in session",
                interactive=False,
            )

            ltm_status = gr.Textbox(
                label="Long-term memory",
                value=_ltm_status(),
                interactive=False,
            )

            gr.Markdown("---")

            gr.Markdown("Answers use the available legal sources when needed.")

        # ---------------------------------------------------------------
        # Chat
        # ---------------------------------------------------------------

        with gr.Column(scale=3):

            chatbot_options = {
                "value": [],
                "label": "Conversation",
                "height": 520,
            }
            # Gradio 4/5 needs type="messages" for role/content dictionaries;
            # Gradio 6 only supports messages and removed the type parameter.
            if "type" in inspect.signature(gr.Chatbot).parameters:
                chatbot_options["type"] = "messages"
            chatbot = gr.Chatbot(**chatbot_options)

            msg_input = gr.Textbox(
                label="Your question",
                placeholder="Ask a legal question...",
                lines=2,
            )

            with gr.Row():

                submit_btn = gr.Button(
                    "Send",
                    variant="primary",
                )

                clear_btn = gr.Button(
                    "🧹 New Session",
                )

    # -------------------------------------------------------------------
    # Events
    # -------------------------------------------------------------------

    submit_outputs = [
        chatbot,
        msg_input,
        stm_status,
        ltm_status,
        session_list,
    ]

    submit_btn.click(
        fn=handle_submit,
        inputs=[msg_input, chatbot],
        outputs=submit_outputs,
        concurrency_id="single_agent",
        concurrency_limit=1,
    )

    msg_input.submit(
        fn=handle_submit,
        inputs=[msg_input, chatbot],
        outputs=submit_outputs,
        concurrency_id="single_agent",
        concurrency_limit=1,
    )

    clear_btn.click(
        fn=new_session,
        outputs=[
            chatbot,
            msg_input,
            stm_status,
            ltm_status,
            session_list,
        ],
        concurrency_id="single_agent",
        concurrency_limit=1,
    )

    new_session_btn.click(
        fn=new_session,
        outputs=[
            chatbot,
            msg_input,
            stm_status,
            ltm_status,
            session_list,
        ],
        concurrency_id="single_agent",
        concurrency_limit=1,
    )

    session_list.select(
        fn=load_selected_session,
        inputs=[session_list],
        outputs=[
            chatbot,
            stm_status,
            ltm_status,
        ],
        concurrency_id="single_agent",
        concurrency_limit=1,
    )

    demo.queue(default_concurrency_limit=1)


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    launch_options = {"share": False, "debug": DEBUG_MODE, "server_name": "127.0.0.1"}
    if "theme" in inspect.signature(demo.launch).parameters:
        launch_options["theme"] = theme
    demo.launch(**launch_options)


