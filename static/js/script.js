// ============================================================
// AI DEBATE CHAMBER — Frontend Controller
// Matches Zenvyro Labs UI: blue advocate vs red challenger,
// real SocketIO live updates, Zenvyro-style judge overlay.
// ============================================================

const socket = io();

// DOM references
const clock        = document.getElementById('clock');
const startBtn     = document.getElementById('startBtn');
const feedA        = document.getElementById('feedA');
const feedB        = document.getElementById('feedB');
const dotA         = document.getElementById('dotA');
const dotB         = document.getElementById('dotB');
const globalDot    = document.getElementById('globalDot');
const statusText   = document.getElementById('statusText');
const displayTopic = document.getElementById('displayTopic');
const roundNum     = document.getElementById('roundNum');
const msgCount     = document.getElementById('msgCount');
const wordCount    = document.getElementById('wordCount');
const memCount     = document.getElementById('memCount');

let timerInterval = null;
let totalDuration  = 120;
let msgs = 0, words = 0;

// ── Socket Events ──────────────────────────────────────────────

socket.on('debate_started', ({ topic, duration }) => {
    totalDuration = duration;
    msgs = 0; words = 0;
    feedA.innerHTML = '';
    feedB.innerHTML = '';
    roundNum.textContent  = '0';
    msgCount.textContent  = '0';
    wordCount.textContent = '0';
    memCount.textContent  = '0';
    displayTopic.textContent = `"${topic}"`;
    globalDot.classList.add('live');
    statusText.textContent = 'Debate initialized. Agent A computing opening statement...';
    startClock(duration);
});

socket.on('round_update', ({ round }) => {
    roundNum.textContent = round;
});

socket.on('typing', ({ agent }) => {
    const feed = agent === 'A' ? feedA : feedB;
    removeTyping(feed);
    const div = document.createElement('div');
    div.className = 'typing-wrapper';
    div.id = `typing${agent}`;
    div.innerHTML = `<div class="typing-indicator">
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
    </div>`;
    feed.appendChild(div);
    feed.scrollTop = feed.scrollHeight;

    if (agent === 'A') {
        dotA.classList.add('active');
        dotB.classList.remove('active');
        statusText.textContent = `Agent A (Advocate) is generating response...`;
    } else {
        dotB.classList.add('active');
        dotA.classList.remove('active');
        statusText.textContent = `Agent B (Challenger) is generating counter-argument...`;
    }
});

socket.on('message', ({ agent, text, round, memory }) => {
    const typingEl = document.getElementById(`typing${agent}`);
    if (typingEl) typingEl.remove();

    msgs++;
    words += text.split(' ').length;
    msgCount.textContent  = msgs;
    wordCount.textContent = words;
    memCount.textContent  = memory;

    const feed = agent === 'A' ? feedA : feedB;
    const cls  = agent === 'A' ? 'msg-adv' : 'msg-chal';
    const role = agent === 'A' ? 'Advocate' : 'Challenger';

    const div = document.createElement('div');
    div.className = `msg ${cls}`;
    div.innerHTML = `
        <div class="round-tag">Round ${round} · ${role}</div>
        <div class="msg-text">${escHtml(text)}</div>
    `;
    feed.appendChild(div);
    feed.scrollTop = feed.scrollHeight;
});

socket.on('debate_ended', (result) => {
    clearInterval(timerInterval);
    clock.textContent = '00:00';
    dotA.classList.remove('active');
    dotB.classList.remove('active');
    globalDot.classList.remove('live');
    statusText.textContent = 'Debate concluded. ML Judge computing verdict...';
    startBtn.disabled = false;
    startBtn.textContent = '▶ Start Debate';

    setTimeout(() => showJudge(result), 1500);
});

// ── Start Button ───────────────────────────────────────────────

startBtn.addEventListener('click', async () => {
    const topic    = document.getElementById('topicInput').value.trim();
    const duration = parseInt(document.getElementById('durationSelect').value);
    const model    = document.getElementById('modelSelect').value;

    if (!topic) { showError('Please enter a debate topic.'); return; }

    startBtn.disabled = true;
    startBtn.textContent = '● LIVE';

    try {
        const res  = await fetch('/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ topic, duration, model })
        });
        const data = await res.json();
        if (!res.ok) {
            showError(data.error || 'Failed to start debate.');
            startBtn.disabled = false;
            startBtn.textContent = '▶ Start Debate';
        }
    } catch (e) {
        showError('Cannot reach server.');
        startBtn.disabled = false;
        startBtn.textContent = '▶ Start Debate';
    }
});

// ── Clock ──────────────────────────────────────────────────────

function startClock(seconds) {
    clearInterval(timerInterval);
    let remaining = seconds;
    updateClock(remaining);

    timerInterval = setInterval(() => {
        remaining--;
        updateClock(remaining);
        if (remaining <= 60) clock.classList.add('danger');
        if (remaining <= 0) { clearInterval(timerInterval); clock.textContent = '00:00'; }
    }, 1000);
}

function updateClock(s) {
    const m   = String(Math.floor(s / 60)).padStart(2, '0');
    const sec = String(Math.max(s % 60, 0)).padStart(2, '0');
    clock.textContent = `${m}:${sec}`;
}

// ── Judge Overlay ──────────────────────────────────────────────

function showJudge(result) {
    const winnerName = result.winner === 'Agent A'
        ? 'Agent A · The Advocate'
        : result.winner === 'Agent B'
        ? 'Agent B · The Challenger'
        : 'Both Agents (Tie)';

    document.getElementById('scoreA').textContent = result.score_a + ' / 10';
    document.getElementById('scoreB').textContent = result.score_b + ' / 10';

    // Animate bars after short delay
    setTimeout(() => {
        document.getElementById('barA').style.width = (result.score_a * 10) + '%';
        document.getElementById('barB').style.width = (result.score_b * 10) + '%';
    }, 200);

    const feat = result.features_a;
    document.getElementById('verdictBox').innerHTML = `
        <strong>Winner: ${winnerName}</strong><br/><br/>
        The SciKit-Learn RandomForestRegressor analyzed ${msgs} arguments 
        across ${roundNum.textContent} rounds. ${winnerName} demonstrated 
        superior ${result.reason}.
    `;

    document.getElementById('judgeMetrics').innerHTML = `
        <div class="metric">
            <div class="label">Model Used</div>
            <div class="val">RandomForestRegressor</div>
        </div>
        <div class="metric">
            <div class="label">Features Extracted</div>
            <div class="val">10 NLP Features</div>
        </div>
        <div class="metric">
            <div class="label">Agent A Score</div>
            <div class="val" style="color:#3b82f6">${result.score_a} / 10</div>
        </div>
        <div class="metric">
            <div class="label">Agent B Score</div>
            <div class="val" style="color:#ef4444">${result.score_b} / 10</div>
        </div>
        <div class="metric">
            <div class="label">Confidence Delta</div>
            <div class="val">${result.confidence}%</div>
        </div>
        <div class="metric">
            <div class="label">Memory Entries</div>
            <div class="val">${memCount.textContent}</div>
        </div>
    `;

    document.getElementById('judgeOverlay').style.display = 'flex';
}

// ── Reset ──────────────────────────────────────────────────────

function resetDebate() {
    document.getElementById('judgeOverlay').style.display = 'none';
    clock.textContent = '--:--';
    clock.classList.remove('danger');
    displayTopic.textContent = 'Awaiting initialization...';
    feedA.innerHTML = '<p style="color:#64748b;text-align:center;margin-top:60px;font-style:italic;">Awaiting debate initialization...</p>';
    feedB.innerHTML = '<p style="color:#64748b;text-align:center;margin-top:60px;font-style:italic;">Awaiting debate initialization...</p>';
    statusText.textContent = 'Awaiting initialization...';
    roundNum.textContent = '0';
    msgCount.textContent = '0';
    wordCount.textContent = '0';
    memCount.textContent = '0';
    startBtn.disabled = false;
    startBtn.textContent = '▶ Start Debate';
    document.getElementById('topicInput').value = '';
}

// ── Helpers ────────────────────────────────────────────────────

function removeTyping(feed) {
    ['typingA', 'typingB'].forEach(id => {
        const el = document.getElementById(id);
        if (el && feed.contains(el)) el.remove();
    });
}

function showError(msg) {
    const toast = document.getElementById('errorToast');
    toast.textContent = msg;
    toast.style.display = 'block';
    setTimeout(() => { toast.style.display = 'none'; }, 4000);
}

function escHtml(str) {
    return str
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
}
