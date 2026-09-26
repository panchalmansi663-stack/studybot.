import difflib
import re

# Bigger rule set: each key is a "topic", each value is a list of trigger phrases + the reply
RULES = {
    "greeting": {
        "triggers": ["hi", "hello", "hey", "good morning", "good evening"],
        "response": "Hello! How can I help you today?",
    },
    "bot_name": {
        "triggers": ["your name", "who are you", "what are you"],
        "response": "I'm a chatbot you're building from scratch in Python!",
    },
    "wellbeing": {
        "triggers": ["how are you", "how're you", "how you doing"],
        "response": "I'm just code, but I'm running fine. How about you?",
    },
    "thanks": {
        "triggers": ["thank you", "thanks", "appreciate it"],
        "response": "You're welcome!",
    },
    "capabilities": {
        "triggers": ["what can you do", "help me", "what do you do"],
        "response": "Right now I can chat a little, remember your name, and understand close variations of what I know. I'll get smarter as you keep building me!",
    },
}

EXIT_WORDS = {"bye", "exit", "quit", "goodbye"}


def extract_name(text: str) -> str | None:
    """Look for patterns like 'my name is X' or 'I am X' and pull out X."""
    match = re.search(r"(?:my name is|i am|i'm|call me)\s+([a-zA-Z]+)", text, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    return None


def find_best_topic(text: str) -> str | None:
    """Check exact keyword matches first, then fall back to fuzzy matching."""
    text_lower = text.lower()

    # 1. Exact substring match (fast, precise)
    for topic, data in RULES.items():
        for trigger in data["triggers"]:
            if trigger in text_lower:
                return topic

    # 2. Fuzzy match against all known triggers (catches typos/variations)
    all_triggers = [t for data in RULES.values() for t in data["triggers"]]
    close_matches = difflib.get_close_matches(text_lower, all_triggers, n=1, cutoff=0.6)
    if close_matches:
        matched_trigger = close_matches[0]
        for topic, data in RULES.items():
            if matched_trigger in data["triggers"]:
                return topic

    return None


def get_bot_response(user_input: str, user_name: str | None) -> tuple[str | None, str | None]:
    """Returns (reply, updated_user_name). Reply of None means 'stop the program'."""
    text = user_input.strip()

    if text.lower() in EXIT_WORDS:
        farewell = f"Goodbye, {user_name}!" if user_name else "Goodbye!"
        return None, user_name

    detected_name = extract_name(text)
    if detected_name:
        return f"Nice to meet you, {detected_name}! I'll remember that.", detected_name

    topic = find_best_topic(text)
    if topic:
        reply = RULES[topic]["response"]
        if user_name and topic == "greeting":
            reply = f"Hello again, {user_name}! How can I help?"
        return reply, user_name

    return "I'm not sure how to respond to that yet — I only know a few topics so far.", user_name


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