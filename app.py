"""
Autonomous AI Debate Chamber — Flask + SocketIO Server
Zenvyro Labs Architecture

Routes match the reference project's REST API design:
  POST /api/debate/start       → initialize debate
  POST /api/debate/next-turn   → manual turn trigger (legacy)
  POST /api/machine-learning/train    → retrain the RF model
  POST /api/machine-learning/evaluate → score two text inputs
  GET  /                       → serve the UI

SocketIO is added on top for live real-time debate streaming
so the frontend never needs to poll or refresh.
"""
import threading
import time
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
from flask_cors import CORS

from d_services.ai_service import DebateConductor
from d_services.feature_extractor import FeatureExtractor
from d_services.ml_judge import DebateRegressionJudge

# ── App Setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = "zenvyro-debate-chamber"
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ── Shared Singletons ─────────────────────────────────────────────────────────
extractor = FeatureExtractor()
ml_judge  = DebateRegressionJudge()


class DebateState:
    """Holds all live debate state in one clean object (no globals)."""
    def __init__(self):
        self.running   = False
        self.topic     = ""
        self.duration  = 120
        self.round     = 0
        self.result    = None
        self.conductor: DebateConductor | None = None


state = DebateState()

# ── UI Route ──────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

# ── Debate REST API (matches Zenvyro reference routes) ────────────────────────
@app.route("/api/debate/start", methods=["POST"])
def start_debate_api():
    """Initialises the debate (REST endpoint, mirrors reference project)."""
    data  = request.json or {}
    topic = data.get("topic", "").strip()
    if not topic:
        return jsonify({"error": "Topic is required"}), 400
    return jsonify({"status": "active", "topic": topic})


@app.route("/api/debate/next-turn", methods=["POST"])
def next_turn_api():
    """Manual turn trigger — retained for compatibility with reference UI."""
    if not state.conductor or not state.topic:
        return jsonify({"error": "No debate running"}), 400

    data         = request.json or {}
    last_speaker = data.get("last_speaker", "A")
    agent        = "B" if last_speaker == "A" else "A"

    if agent == "A":
        text = state.conductor.generate_agent_a_response(state.topic)
    else:
        text = state.conductor.generate_agent_b_response(state.topic)

    state.round += 1
    return jsonify({
        "agent":   agent,
        "message": text,
        "round":   state.round,
        "memory":  state.conductor.memory.count()
    })


@app.route("/api/machine-learning/train", methods=["POST"])
def trigger_training():
    """Retrain the SciKit-Learn RandomForestRegressor from the CSV."""
    try:
        _, metrics = ml_judge.train_model("historical_debates.csv")
        return jsonify({"status": "Training Completed", "metrics": metrics})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/machine-learning/evaluate", methods=["POST"])
def evaluate_debate():
    """Score two text inputs and return winner (standalone REST endpoint)."""
    data           = request.json or {}
    advocate_text  = data.get("advocate_text",   "")
    challenger_text = data.get("challenger_text", "")

    feat_a = extractor.to_dataframe(extractor.extract(advocate_text  or "no text"))
    feat_b = extractor.to_dataframe(extractor.extract(challenger_text or "no text"))
    result = ml_judge.judge(feat_a, feat_b)

    return jsonify({
        "winner":           result["winner"],
        "advocate_score":   result["score_a"],
        "challenger_score": result["score_b"],
        "confidence":       result["confidence"],
        "reason":           result["reason"]
    })


@app.route("/status")
def status():
    elapsed   = time.time() - (state._start_time if hasattr(state, "_start_time") else 0)
    remaining = max(0, state.duration - int(elapsed)) if state.running else 0
    return jsonify({
        "running":        state.running,
        "round":          state.round,
        "remaining":      remaining,
        "memory_entries": state.conductor.memory.count() if state.conductor else 0,
    })


@app.route("/winner")
def winner():
    if state.result:
        return jsonify(state.result)
    return jsonify({"error": "Debate not finished yet"}), 400


# ── SocketIO Debate Start ─────────────────────────────────────────────────────
@app.route("/start", methods=["POST"])
def start():
    """Start the autonomous debate loop via SocketIO live streaming."""
    data     = request.get_json() or {}
    topic    = data.get("topic", "").strip()
    duration = int(data.get("duration", 120))
    model    = data.get("model", "llama3.2")

    if not topic:
        return jsonify({"error": "Please enter a debate topic."}), 400
    if duration < 30:
        return jsonify({"error": "Duration must be at least 30 seconds."}), 400
    if state.running:
        return jsonify({"error": "A debate is already running."}), 400

    state.topic     = topic
    state.duration  = duration
    state.round     = 0
    state.result    = None
    state.conductor = DebateConductor(model=model)

    if not state.conductor.is_ready():
        return jsonify({"error": "Ollama is offline. Run: ollama serve"}), 503

    threading.Thread(target=_run_debate, daemon=True).start()
    return jsonify({"status": "started", "topic": topic, "duration": duration})


# ── Debate Loop ───────────────────────────────────────────────────────────────
def _run_debate():
    """
    Main autonomous debate loop — runs in a background thread.
    Emits SocketIO events so the frontend updates live without polling.
    """
    state.running    = True
    state._start_time = time.time()
    texts_a, texts_b = [], []

    socketio.emit("debate_started", {"topic": state.topic, "duration": state.duration})

    while True:
        elapsed = time.time() - state._start_time
        if elapsed >= state.duration:
            break

        state.round += 1
        socketio.emit("round_update", {"round": state.round})

        # — Agent A (Advocate) —
        socketio.emit("typing", {"agent": "A"})
        resp_a = state.conductor.generate_agent_a_response(state.topic)
        texts_a.append(resp_a)
        socketio.emit("message", {
            "agent": "A", "text": resp_a,
            "round": state.round,
            "memory": state.conductor.memory.count()
        })
        time.sleep(0.8)

        if time.time() - state._start_time >= state.duration:
            break

        # — Agent B (Challenger) —
        socketio.emit("typing", {"agent": "B"})
        resp_b = state.conductor.generate_agent_b_response(state.topic)
        texts_b.append(resp_b)
        socketio.emit("message", {
            "agent": "B", "text": resp_b,
            "round": state.round,
            "memory": state.conductor.memory.count()
        })
        time.sleep(0.8)

    # — ML Judge —
    combined_a = " ".join(texts_a) or "No argument provided."
    combined_b = " ".join(texts_b) or "No argument provided."

    feat_a = extractor.to_dataframe(extractor.extract(combined_a))
    feat_b = extractor.to_dataframe(extractor.extract(combined_b))
    result = ml_judge.judge(feat_a, feat_b)

    result["features_a"] = extractor.extract(combined_a)
    result["features_b"] = extractor.extract(combined_b)
    state.result  = result
    state.running = False

    socketio.emit("debate_ended", result)


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 AI Debate Chamber — http://127.0.0.1:5000")
    print("Ensure Ollama is running locally on port 11434!")
    socketio.run(app, debug=True, host="0.0.0.0", port=5000)
