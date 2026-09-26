import json
import os
import uuid
import chromadb
import ollama
from flask import Flask, render_template_string, request, jsonify

MODEL_NAME = "llama3.2:1b"
SESSIONS_DIR = "chat_sessions"
MAX_MESSAGES = 20
RELEVANCE_THRESHOLD = 0.45

SYSTEM_PROMPT = (
    "You are StudyBot, an AI assistant — you are NOT the user. Keep answers "
    "clear and concise (2-4 sentences unless asked for more). Remember and use "
    "what the user has told you earlier in this conversation. If a note is "
    "given below a question and it answers that question, mention you used "
    "your saved notes. Otherwise just answer normally."
)

os.makedirs(SESSIONS_DIR, exist_ok=True)

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="it_notes_v2",
    metadata={"hnsw:space": "cosine"},
)

app = Flask(__name__)


def session_path(session_id):
    return os.path.join(SESSIONS_DIR, f"{session_id}.json")


def load_session(session_id):
    path = session_path(session_id)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "id": session_id,
        "title": "New chat",
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}],
    }


def save_session(session_data):
    with open(session_path(session_data["id"]), "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=2)


def list_sessions():
    sessions = []
    for filename in os.listdir(SESSIONS_DIR):
        if filename.endswith(".json"):
            full_path = os.path.join(SESSIONS_DIR, filename)
            with open(full_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            sessions.append({
                "id": data["id"],
                "title": data.get("title", "New chat"),
                "updated": os.path.getmtime(full_path),
            })
    sessions.sort(key=lambda s: s["updated"], reverse=True)
    return sessions


def trim_messages(messages):
    system_message = messages[0]
    recent = messages[1:][-MAX_MESSAGES:]
    return [system_message] + recent


def retrieve_notes(question):
    results = collection.query(query_texts=[question], n_results=1)
    documents = results["documents"][0] if results["documents"] else []
    distances = results["distances"][0] if results["distances"] else []
    if not documents or distances[0] > RELEVANCE_THRESHOLD:
        return ""
    return f"\n\n[Note: {documents[0]}]"


PAGE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>StudyBot</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
        font-family: 'Inter', sans-serif;
        background: #ffffff;
        color: #2b2b2b;
        height: 100vh;
        display: flex;
        overflow: hidden;
    }
    #sidebar {
        width: 260px;
        background: #faf9f6;
        border-right: 1px solid #ece9e3;
        display: flex;
        flex-direction: column;
        padding: 16px 12px;
    }
    #sidebar h1 {
        font-size: 16px;
        font-weight: 700;
        color: #da7756;
        padding: 4px 8px 16px 8px;
    }
    #new-chat-btn {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 10px 12px;
        background: #da7756;
        color: white;
        border: none;
        border-radius: 8px;
        font-size: 13.5px;
        font-weight: 600;
        cursor: pointer;
        margin-bottom: 16px;
        transition: background 0.15s;
    }
    #new-chat-btn:hover { background: #c9663f; }
    #session-list {
        flex: 1;
        overflow-y: auto;
        display: flex;
        flex-direction: column;
        gap: 4px;
    }
    .session-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 6px;
        padding: 9px 8px 9px 12px;
        border-radius: 8px;
        font-size: 13px;
        color: #4a4a4a;
        cursor: pointer;
        transition: background 0.12s;
    }
    .session-item:hover { background: #f0ede7; }
    .session-item.active { background: #ece5db; color: #1a1a1a; font-weight: 600; }
    .session-title {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        flex: 1;
    }
    .delete-btn {
        opacity: 1;
        background: #f0ede7;
        border: none;
        color: #8a7a68;
        font-size: 13px;
        font-weight: bold;
        cursor: pointer;
        padding: 3px 8px;
        border-radius: 4px;
        flex-shrink: 0;
    }
    .delete-btn:hover { background: #e0533e; color: white; }
    #main {
        flex: 1;
        display: flex;
        flex-direction: column;
        height: 100vh;
    }
    #chat-box {
        flex: 1;
        overflow-y: auto;
        padding: 32px 0;
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .msg-wrapper {
        width: 100%;
        max-width: 680px;
        padding: 0 24px;
        margin-bottom: 22px;
        animation: fadeIn 0.25s ease;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .msg-wrapper.user { display: flex; justify-content: flex-end; }
    .user .bubble {
        background: #f2ede4;
        color: #2b2b2b;
        padding: 10px 16px;
        border-radius: 16px;
        max-width: 75%;
        font-size: 14.5px;
        line-height: 1.5;
        white-space: pre-wrap;
    }
    .bot-row { display: flex; gap: 12px; align-items: flex-start; }
    .bot-avatar {
        width: 28px; height: 28px;
        border-radius: 50%;
        background: #da7756;
        color: white;
        display: flex; align-items: center; justify-content: center;
        font-size: 14px;
        flex-shrink: 0;
        margin-top: 2px;
    }
    .bot-text {
        font-size: 14.5px;
        line-height: 1.6;
        color: #2b2b2b;
        max-width: calc(100% - 40px);
    }
    .bot-text p { margin-bottom: 10px; }
    .bot-text ul, .bot-text ol { margin: 8px 0 8px 22px; }
    .bot-text li { margin-bottom: 4px; }
    .bot-text code {
        background: #f2ede4;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 13px;
        font-family: 'Courier New', monospace;
    }
    .bot-text pre {
        background: #26263f;
        color: #e4e4e7;
        padding: 12px 16px;
        border-radius: 8px;
        overflow-x: auto;
        margin: 8px 0;
    }
    .bot-text pre code { background: none; padding: 0; color: inherit; }
    .bot-text strong { font-weight: 700; }
    .typing-dots { display: flex; gap: 4px; padding: 6px 0; }
    .typing-dots span {
        width: 6px; height: 6px; border-radius: 50%;
        background: #b8b0a3;
        animation: bounce 1.2s infinite ease-in-out;
    }
    .typing-dots span:nth-child(2) { animation-delay: 0.15s; }
    .typing-dots span:nth-child(3) { animation-delay: 0.3s; }
    @keyframes bounce {
        0%, 60%, 100% { transform: translateY(0); opacity: 0.5; }
        30% { transform: translateY(-5px); opacity: 1; }
    }
    #input-area {
        padding: 16px 24px 24px 24px;
        display: flex;
        justify-content: center;
    }
    #input-row {
        width: 100%;
        max-width: 680px;
        display: flex;
        gap: 10px;
        border: 1px solid #e2ded5;
        border-radius: 26px;
        padding: 6px 6px 6px 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
    }
    #user-input {
        flex: 1;
        border: none;
        outline: none;
        font-size: 14.5px;
        font-family: inherit;
        background: transparent;
        padding: 8px 0;
    }
    #send-btn {
        width: 38px; height: 38px;
        border-radius: 50%;
        border: none;
        background: #da7756;
        color: white;
        font-size: 16px;
        cursor: pointer;
        display: flex; align-items: center; justify-content: center;
        transition: background 0.15s;
    }
    #send-btn:hover { background: #c9663f; }
</style>
</head>
<body>
    <div id="sidebar">
        <h1>StudyBot</h1>
        <button id="new-chat-btn" onclick="newChat()">+ New chat</button>
        <div id="session-list"></div>
    </div>

    <div id="main">
        <div id="chat-box"></div>
        <div id="input-area">
            <div id="input-row">
                <input type="text" id="user-input" placeholder="Ask something..." autofocus>
                <button id="send-btn" onclick="sendMessage()">➤</button>
            </div>
        </div>
    </div>

    <script>
        let currentSessionId = null;
        const chatBox = document.getElementById('chat-box');
        const sessionList = document.getElementById('session-list');
        const userInput = document.getElementById('user-input');

        function renderMessage(text, sender) {
            const wrapper = document.createElement('div');
            wrapper.className = 'msg-wrapper ' + sender;
            if (sender === 'user') {
                wrapper.innerHTML = `<div class="bubble"></div>`;
                wrapper.querySelector('.bubble').textContent = text;
            } else {
                wrapper.innerHTML = `
                    <div class="bot-row">
                        <div class="bot-avatar">🤖</div>
                        <div class="bot-text"></div>
                    </div>`;
                wrapper.querySelector('.bot-text').innerHTML = marked.parse(text);
            }
            chatBox.appendChild(wrapper);
            chatBox.scrollTop = chatBox.scrollHeight;
            return wrapper;
        }

        function renderTyping() {
            const wrapper = document.createElement('div');
            wrapper.className = 'msg-wrapper bot';
            wrapper.id = 'typing-wrapper';
            wrapper.innerHTML = `
                <div class="bot-row">
                    <div class="bot-avatar">🤖</div>
                    <div class="typing-dots"><span></span><span></span><span></span></div>
                </div>`;
            chatBox.appendChild(wrapper);
            chatBox.scrollTop = chatBox.scrollHeight;
        }

        async function loadSessions() {
            const res = await fetch('/api/sessions');
            const sessions = await res.json();
            sessionList.innerHTML = '';
            sessions.forEach(s => {
                const item = document.createElement('div');
                item.className = 'session-item' + (s.id === currentSessionId ? ' active' : '');
                item.innerHTML = `
                    <span class="session-title">${s.title}</span>
                    <button class="delete-btn" title="Delete chat">✕</button>
                `;
                item.querySelector('.session-title').onclick = () => loadSession(s.id);
                item.querySelector('.delete-btn').onclick = (e) => {
                    e.stopPropagation();
                    deleteSession(s.id);
                };
                sessionList.appendChild(item);
            });
            return sessions;
        }

        async function deleteSession(sessionId) {
            if (!confirm('Delete this chat? This cannot be undone.')) return;
            await fetch('/api/session/' + sessionId, { method: 'DELETE' });
            if (sessionId === currentSessionId) {
                const remaining = await loadSessions();
                if (remaining.length > 0) {
                    loadSession(remaining[0].id);
                } else {
                    newChat();
                }
            } else {
                loadSessions();
            }
        }

        async function loadSession(sessionId) {
            currentSessionId = sessionId;
            const res = await fetch('/api/session/' + sessionId);
            const data = await res.json();
            chatBox.innerHTML = '';
            data.messages.forEach(m => renderMessage(m.content, m.role === 'user' ? 'user' : 'bot'));
            loadSessions();
        }

        async function newChat() {
            const res = await fetch('/api/new_session', { method: 'POST' });
            const data = await res.json();
            currentSessionId = data.id;
            chatBox.innerHTML = '';
            renderMessage("Hi! I'm StudyBot, your local IT assistant. Ask me anything!", 'bot');
            loadSessions();
        }

        async function sendMessage() {
            const message = userInput.value.trim();
            if (!message || !currentSessionId) return;
            renderMessage(message, 'user');
            userInput.value = '';
            renderTyping();

            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ session_id: currentSessionId, message: message })
            });
            const data = await response.json();

            document.getElementById('typing-wrapper').remove();
            renderMessage(data.reply, 'bot');
            loadSessions();
        }

        userInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') sendMessage();
        });

        (async () => {
            const sessions = await loadSessions();
            if (sessions.length > 0) {
                loadSession(sessions[0].id);
            } else {
                newChat();
            }
        })();
    </script>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(PAGE_HTML)


@app.route("/api/sessions")
def api_sessions():
    return jsonify(list_sessions())


@app.route("/api/new_session", methods=["POST"])
def api_new_session():
    session_id = str(uuid.uuid4())[:8]
    session_data = {
        "id": session_id,
        "title": "New chat",
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}],
    }
    save_session(session_data)
    return jsonify({"id": session_id})


@app.route("/api/session/<session_id>")
def api_get_session(session_id):
    data = load_session(session_id)
    visible = [m for m in data["messages"] if m["role"] != "system"]
    return jsonify({"id": data["id"], "title": data["title"], "messages": visible})


@app.route("/api/session/<session_id>", methods=["DELETE"])
def api_delete_session(session_id):
    path = session_path(session_id)
    if os.path.exists(path):
        os.remove(path)
        return jsonify({"deleted": True})
    return jsonify({"deleted": False}), 404


@app.route("/api/chat", methods=["POST"])
def api_chat():
    body = request.json
    session_id = body["session_id"]
    user_message = body["message"]

    session_data = load_session(session_id)
    session_data["messages"].append({"role": "user", "content": user_message})

    notes_context = retrieve_notes(user_message)
    messages_for_model = session_data["messages"][:-1] + [
        {"role": "user", "content": user_message + notes_context}
    ]
    messages_for_model = trim_messages(messages_for_model)

    response = ollama.chat(
        model=MODEL_NAME,
        messages=messages_for_model,
        options={"temperature": 0.3},
    )
    reply = response["message"]["content"]

    session_data["messages"].append({"role": "assistant", "content": reply})
    session_data["messages"] = trim_messages(session_data["messages"])

    if session_data["title"] == "New chat":
        session_data["title"] = user_message[:40] + ("..." if len(user_message) > 40 else "")

    save_session(session_data)
    return jsonify({"reply": reply, "title": session_data["title"]})


if __name__ == "__main__":
    app.run(debug=True, port=5000)