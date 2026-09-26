import ollama

MODEL_NAME = "llama3.2:1b"
EXIT_WORDS = {"bye", "exit", "quit", "goodbye"}


def get_bot_response(user_input: str) -> str:
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": user_input}],
    )
    return response["message"]["content"]


def main():
    print("Chatbot: Hi! I'm running on a local AI model now. Type 'bye' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower().strip() in EXIT_WORDS:
            print("Chatbot: Goodbye!")
            break

        print("Chatbot: thinking...")
        reply = get_bot_response(user_input)
        print(f"Chatbot: {reply}")


if __name__ == "__main__":
    main()