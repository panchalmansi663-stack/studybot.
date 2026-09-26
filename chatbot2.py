from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Knowledge base: each entry has example phrasings + one answer
KNOWLEDGE_BASE = [
    {
        "examples": ["hi", "hello", "hey there", "good morning"],
        "response": "Hello! How can I help you today?",
    },
    {
        "examples": ["what is your name", "who are you", "what are you called"],
        "response": "I'm a chatbot you're building from scratch in Python!",
    },
    {
        "examples": ["how are you", "how are you doing", "how's it going"],
        "response": "I'm just code, but I'm running fine. How about you?",
    },
    {
        "examples": ["thank you", "thanks a lot", "i appreciate it"],
        "response": "You're welcome!",
    },
    {
        "examples": ["what can you do", "help me", "what are your features"],
        "response": "I can chat, remember your name, and match things you say even if worded differently.",
    },
]

EXIT_WORDS = {"bye", "exit", "quit", "goodbye"}

# Flatten all examples into one list, remembering which knowledge-base entry each belongs to
all_examples = []
example_owner = []  # parallel list: which KNOWLEDGE_BASE index each example belongs to
for i, entry in enumerate(KNOWLEDGE_BASE):
    for example in entry["examples"]:
        all_examples.append(example)
        example_owner.append(i)

# Build the TF-IDF model once, on startup, from all known examples
vectorizer = TfidfVectorizer()
example_vectors = vectorizer.fit_transform(all_examples)


def extract_name(text: str) -> str | None:
    match = re.search(r"(?:my name is|i am|i'm|call me)\s+([a-zA-Z]+)", text, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    return None


def find_best_response(user_text: str, threshold: float = 0.3) -> str | None:
    """Turn user_text into a vector, compare it to every known example, return the best match's response."""
    user_vector = vectorizer.transform([user_text])
    similarities = cosine_similarity(user_vector, example_vectors)[0]

    best_index = similarities.argmax()
    best_score = similarities[best_index]

    if best_score < threshold:
        return None  # nothing close enough

    matched_entry_index = example_owner[best_index]
    return KNOWLEDGE_BASE[matched_entry_index]["response"]


def get_bot_response(user_input: str, user_name: str | None) -> tuple[str | None, str | None]:
    text = user_input.strip()

    if text.lower() in EXIT_WORDS:
        return None, user_name

    detected_name = extract_name(text)
    if detected_name:
        return f"Nice to meet you, {detected_name}! I'll remember that.", detected_name

    response = find_best_response(text)
    if response:
        if user_name and "Hello!" in response:
            response = f"Hello again, {user_name}! How can I help?"
        return response, user_name

    return "I'm not sure how to respond to that yet — try rephrasing, or ask me something else.", user_name


def main():
    user_name = None
    print("Chatbot: Hi! Type 'bye' to exit.")
    while True:
        user_input = input("You: ")
        reply, user_name = get_bot_response(user_input, user_name)
        if reply is None:
            print(f"Chatbot: Goodbye{', ' + user_name if user_name else ''}!")
            break
        print(f"Chatbot: {reply}")


if __name__ == "__main__":
    main()