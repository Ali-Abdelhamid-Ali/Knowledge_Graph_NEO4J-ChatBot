from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neo4j_factual_chatbot.core.config import get_settings
from neo4j_factual_chatbot.services.chat_service import Neo4jFactualChatbotService

EXIT_COMMANDS = {"exit", "quit", "q", "\u062e\u0631\u0648\u062c", "\u0627\u0646\u0647\u0627\u0621"}


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

    settings = get_settings()
    service = Neo4jFactualChatbotService(settings)

    try:
        service.verify_connection()
    except Exception as exc:
        service.close()
        raise SystemExit(f"Failed to connect to Neo4j Aura: {exc}") from exc

    print("Neo4j factual chatbot is ready.")
    print("This is an isolated project separate from the SQL API app.")
    print("Supported actions: add, inquire, update, delete.")
    print("Run: python neo4j_factual_chatbot/main.py")
    print("Type exit, quit, or q to stop. Arabic exit commands are also supported.")

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                print("\nAssistant: Goodbye.")
                break

            if not user_input:
                continue

            if user_input.lower() in EXIT_COMMANDS or user_input in EXIT_COMMANDS:
                print("Assistant: Goodbye.")
                break

            try:
                assistant_response = service.handle_message(user_input)
            except Exception:
                logging.getLogger(__name__).exception("Chatbot turn failed")
                assistant_response = service._default_error_message(user_input)

            print(f"Assistant: {assistant_response}")
    except KeyboardInterrupt:
        print("\nAssistant: Goodbye.")
    finally:
        service.close()


if __name__ == "__main__":
    main()
