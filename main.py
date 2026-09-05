"""Run a local single-agent chat: python main.py [--long-term]."""

import argparse
import logging
from pathlib import Path

from agent import SingleAgentRAG


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--long-term", action="store_true",
                        help="Enable persistent Chroma Q&A memory")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    data_dir = Path(__file__).parent / "data"
    ltm = None
    if args.long_term:
        from memory.long_term import LongTermMemory
        ltm = LongTermMemory(db_dir=str(data_dir / "chroma_db"))
    rag = SingleAgentRAG(long_term=ltm, data_dir=data_dir)
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
