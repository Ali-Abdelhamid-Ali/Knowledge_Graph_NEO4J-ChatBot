# Knowledge_Graph_NEO4J-ChatBot

# Neo4j Factual Chatbot

Standalone CLI project for a Neo4j + Cohere + LangGraph factual chatbot.

## Architecture

```text

├── main.py                     # CLI entrypoint
├── core/config.py             # Environment configuration
├── schemas/chat.py            # Pydantic models and workflow state
├── llm/parsers.py             # LLM response parsing helpers
├── prompts/chatbot.py         # Cohere prompt templates
├── graph/repository.py        # Neo4j CRUD operations
├── workflow/langgraph_flow.py # LangGraph graph wiring
└── services/chat_service.py   # Service orchestration
```

## Run

```bash
python main.py
```

Required env vars:

- `COHERE_API_KEY`
- `NEO4J_URI`
- `NEO4J_USERNAME`
- `NEO4J_PASSWORD`
- `NEO4J_DATABASE`
