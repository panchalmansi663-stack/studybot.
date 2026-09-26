import json
import os
import ollama

MODEL_NAME = "llama3.2:1b"
EXIT_WORDS = {"bye", "exit", "quit", "goodbye"}
HISTORY_FILE = "chat_history.json"
MAX_MESSAGES = 20  # keep the last 20 messages (10 exchanges) to avoid slowdown over time

SYSTEM_PROMPT = (
    "You are a friendly study-help chatbot. Keep answers clear and reasonably "
    "concise unless the user asks for more detail. If you're not fully sure "
    "about a specific fact, say so instead of guessing confidently. "
    "You DO have access to everything the user has told you earlier in this "
    "same conversation — use it confidently and don't claim you can't remember "
    "things they've already told you in this chat."
)


def load_history() -> list[dict]:
    """Load saved conversation from disk, or start fresh if none exists."""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def save_history(history: list[dict]) -> None:
    """Write the current conversation to disk so it survives closing the program."""
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def trim_history(history: list[dict]) -> list[dict]:
    """Keep the system prompt + only the most recent messages, so it doesn't grow forever."""
    system_message = history[0]
    recent_messages = history[1:][-MAX_MESSAGES:]
    return [system_message] + recent_messages


def main():
    conversation_history = load_history()

    if len(conversation_history) > 1:
        print("Chatbot: Welcome back! I remember our earlier conversation. Type 'bye' to exit.")
    else:
        print("Chatbot: Hi! I'm your local study assistant. Type 'bye' to exit.")

    while True:
        user_input = input("You: ")
        if user_input.lower().strip() in EXIT_WORDS:
            save_history(conversation_history)
            print("Chatbot: Goodbye! I've saved our conversation for next time.")
            break

        conversation_history.append({"role": "user", "content": user_input})
        conversation_history = trim_history(conversation_history)

        print("Chatbot: thinking...")
        response = ollama.chat(model=MODEL_NAME, messages=conversation_history)
        reply = response["message"]["content"]

        conversation_history.append({"role": "assistant", "content": reply})
        print(f"Chatbot: {reply}")

        save_history(conversation_history)  # save after every message, not just on exit


if __name__ == "__main__":
    main()