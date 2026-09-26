import json
import os
import chromadb
import ollama

MODEL_NAME = "llama3.2:1b"
EXIT_WORDS = {"bye", "exit", "quit", "goodbye"}
HISTORY_FILE = "chat_history.json"
MAX_MESSAGES = 20
RELEVANCE_THRESHOLD = 0.45

SYSTEM_PROMPT = (
    "You are StudyBot, an AI assistant — you are NOT the user. Keep answers "
    "clear and concise (2-4 sentences unless asked for more). Remember and use "
    "what the user has told you earlier in this conversation. If a note is "
    "given below a question and it answers that question, mention you used "
    "your saved notes. Otherwise just answer normally."
)

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="it_notes_v2",
    metadata={"hnsw:space": "cosine"},
)


def retrieve_notes(question: str) -> str:
    results = collection.query(query_texts=[question], n_results=1)
    documents = results["documents"][0] if results["documents"] else []
    distances = results["distances"][0] if results["distances"] else []

    if not documents or distances[0] > RELEVANCE_THRESHOLD:
        return ""

    return f"\n\n[Note: {documents[0]}]"


def load_history() -> list[dict]:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def save_history(history: list[dict]) -> None:
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def trim_history(history: list[dict]) -> list[dict]:
    system_message = history[0]
    recent_messages = history[1:][-MAX_MESSAGES:]
    return [system_message] + recent_messages


def main():
    conversation_history = load_history()
    greeting = "Welcome back!" if len(conversation_history) > 1 else "Hi! I'm StudyBot, your local IT assistant."
    print(f"Chatbot: {greeting} Type 'bye' to exit.")

    while True:
        user_input = input("You: ")
        if user_input.lower().strip() in EXIT_WORDS:
            save_history(conversation_history)
            print("Chatbot: Goodbye! I've saved our conversation for next time.")
            break

        notes_context = retrieve_notes(user_input)
        conversation_history.append({"role": "user", "content": user_input + notes_context})
        conversation_history = trim_history(conversation_history)

        response = ollama.chat(
            model=MODEL_NAME,
            messages=conversation_history,
            options={"temperature": 0.3},
        )
        reply = response["message"]["content"]

        conversation_history.append({"role": "assistant", "content": reply})
        print(f"Chatbot: {reply}")
        save_history(conversation_history)


if __name__ == "__main__":
    main()