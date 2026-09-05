"""Run a local single-agent chat: python main.py."""

import logging
from pathlib import Path

from agent import SingleAgentRAG


def main():
    logging.basicConfig(level=logging.INFO)
    data_dir = Path(__file__).parent / "data"
    rag = SingleAgentRAG(data_dir=data_dir)
    print("Legal RAG. Use /new for a new chat or /quit to exit.")
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if query == "/quit":
            break
        if query == "/new":
            rag.new_session()
            continue
        if not query:
            continue
        try:
            print("Assistant:", rag.ask(query)["answer"])
        except Exception as exc:
            logger = logging.getLogger(__name__)
            logger.error("Could not complete this turn: %s", exc)


if __name__ == "__main__":
    main()
