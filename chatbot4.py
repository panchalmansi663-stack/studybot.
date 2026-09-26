import ollama

MODEL_NAME = "llama3.2:1b"
EXIT_WORDS = {"bye", "exit", "quit", "goodbye"}

SYSTEM_PROMPT = (
    "You are a friendly study-help chatbot. Keep answers clear and reasonably "
    "concise unless the user asks for more detail. If you're not fully sure "
    "about a specific fact, say so instead of guessing confidently. "
    "You DO have access to everything the user has told you earlier in this "
    "same conversation — use it confidently and don't claim you can't remember "
    "things they've already told you in this chat."
)


def main():
    # conversation_history holds every message so far — this is what gives it "memory"
    conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("Chatbot: Hi! I'm your local study assistant. Type 'bye' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower().strip() in EXIT_WORDS:
            print("Chatbot: Goodbye!")
            break

        conversation_history.append({"role": "user", "content": user_input})

        print("Chatbot: thinking...")
        response = ollama.chat(model=MODEL_NAME, messages=conversation_history)
        reply = response["message"]["content"]

        # Add the bot's own reply to history too, so IT remembers what it said
        conversation_history.append({"role": "assistant", "content": reply})

        print(f"Chatbot: {reply}")


if __name__ == "__main__":
    main()