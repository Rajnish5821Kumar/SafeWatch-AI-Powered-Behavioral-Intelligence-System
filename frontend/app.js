/**
 * SafeWatch Dashboard — Real-Time Frontend Application
 * =====================================================
 * WebSocket client, Chart.js visualizations, live DOM updates,
 * and video upload/analysis flow.
 */

'use strict';

// ── Configuration ──────────────────────────────────────────────────────────
const API_BASE   = 'http://localhost:8000';
const WS_URL     = 'ws://localhost:8000/ws/stream';
const POLL_INTERVAL = 8000;  // ms — analytics polling fallback

// ── State ───────────────────────────────────────────────────────────────────
let socket        = null;
let isConnected   = false;
let alertCount    = 0;
let frameCount    = 0;
let selectedFile  = null;
let timelineChart = null;
let emotionChart  = null;

// ── Initialise on DOM ready ──────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initCharts();
  initUpload();
  connectWebSocket();
  startAnalyticsPoll();
  seedDemoState();
  loadAlertHistory();
});

// ═══════════════════════════════════════════════════════════════════════════
// WebSocket
// ═══════════════════════════════════════════════════════════════════════════

function connectWebSocket() {
  try {
    socket = new WebSocket(WS_URL);

    socket.onopen = () => {
      isConnected = true;
      setLiveStatus(true);
      setWsDot('green');
      console.log('[SafeWatch] WebSocket connected');
    };

    socket.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data);
        if (msg.type === 'frame_result') handleFrameResult(msg.data);
        if (msg.type === 'heartbeat') updateTimestamp(msg.ts);
      } catch (e) {
        console.warn('[SafeWatch] WS parse error:', e);
      }
    };

    socket.onclose = () => {
      isConnected = false;
      setLiveStatus(false);
      setWsDot('amber');
      console.log('[SafeWatch] WebSocket disconnected — retrying in 4s');
      setTimeout(connectWebSocket, 4000);
    };

    socket.onerror = (e) => {
      setWsDot('red');
      console.warn('[SafeWatch] WebSocket error:', e);
    };

  } catch (e) {
    console.warn('[SafeWatch] Cannot connect WebSocket:', e);
    setLiveStatus(false);
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Frame Result Handler
// ═══════════════════════════════════════════════════════════════════════════

function handleFrameResult(data) {
  frameCount++;

  // Header stats
  setEl('hdrPersons', data.n_persons ?? '—');
  setEl('hdrFps', data.fps ? data.fps.toFixed(1) : '—');

  // Stage latencies
  const lat = data.stage_latencies_ms || {};
  setEl('latDetect',  lat.detection  ? lat.detection + 'ms'  : '—');
  setEl('latPose',    lat.pose        ? lat.pose + 'ms'       : '—');
  setEl('latAnomaly', lat.anomaly     ? lat.anomaly + 'ms'    : '—');
  setEl('latTotal',   lat.total       ? lat.total + 'ms'      : '—');

  // Tracked person cards
  updatePersonsList(data.tracked_ids || [], data.anomalies || {});

  // Alerts
  if (data.alerts && data.alerts.length > 0) {
    data.alerts.forEach(addAlertToFeed);
    alertCount += data.alerts.length;
    setEl('hdrAlerts', alertCount);
    setEl('pillAlert', `⚠ ${alertCount} Alert${alertCount > 1 ? 's' : ''}`);
    el('pillAlert').classList.remove('hidden');
  }

  // Emotion distribution
  if (data.emotion_summary) updateEmotionChart(data.emotion_summary);

  // Timeline
  if (data.fps > 0 && data.n_persons >= 0) {
    pushTimelinePoint(data);
  }

  // Pill
  el('pillPersons').textContent = `${data.n_persons || 0} Persons`;

  // Timestamp
  updateTimestamp(data.timestamp);
}

// ═══════════════════════════════════════════════════════════════════════════
// Persons List
// ═══════════════════════════════════════════════════════════════════════════

function updatePersonsList(trackIds, anomalies) {
  const container = el('personsList');
  container.innerHTML = '';

  if (!trackIds.length) {
    container.innerHTML = '<div class="empty-state">No active tracks</div>';
    return;
  }

  trackIds.slice(0, 12).forEach(tid => {
    const anom = anomalies[String(tid)] || {};
    const score = anom.score || 0;
    const sev   = anom.severity || 'none';
    const isAlert = anom.alert;

    const card = document.createElement('div');
    card.className = `person-card${isAlert ? ' alert' : ''}`;
    card.innerHTML = `
      <div class="person-avatar">${tid}</div>
      <div class="person-info">
        <div class="person-id">Track #${tid}</div>
        <div class="person-status">${isAlert ? '⚠ ' + sev.toUpperCase() : '● Normal'}</div>
      </div>
      <div class="person-score ${scoreClass(score)}">${(score * 100).toFixed(0)}%</div>
    `;
    container.appendChild(card);
  });
}

function scoreClass(s) {
  if (s < 0.5)  return 'score-normal';
  if (s < 0.72) return 'score-normal';
  if (s < 0.82) return 'score-low';
  if (s < 0.92) return 'score-medium';
  return 'score-high';
}

// ═══════════════════════════════════════════════════════════════════════════
// Alert Feed
// ═══════════════════════════════════════════════════════════════════════════

function addAlertToFeed(alert) {
  const feed = el('alertsFeed');
  const emptyState = feed.querySelector('.empty-state');
  if (emptyState) emptyState.remove();

  const item = document.createElement('div');
  item.className = `alert-item ${alert.severity}`;
  item.setAttribute('title', (alert.evidence || []).join('\n'));

  const ts = alert.timestamp ? new Date(alert.timestamp * 1000).toLocaleTimeString() : '—';
  const evidence = (alert.evidence || []).slice(0, 1).join(' ');

  item.innerHTML = `
    <div class="alert-top">
      <span class="alert-badge">${alert.severity || 'low'}</span>
      <span class="alert-time">${ts} · Track #${alert.track_id}</span>
    </div>
    <div class="alert-headline">${escHtml(alert.headline || 'Behavioral anomaly detected')}</div>
    <div class="alert-evidence">${escHtml(evidence)}</div>
  `;

  feed.insertBefore(item, feed.firstChild);

  // Keep max 20 alerts visible
  const items = feed.querySelectorAll('.alert-item');
  if (items.length > 20) items[items.length - 1].remove();
}

document.addEventListener('DOMContentLoaded', () => {
  el('clearAlertsBtn').addEventListener('click', async () => {
    el('alertsFeed').innerHTML = '<div class="empty-state">No alerts — all clear ✓</div>';
    alertCount = 0;
    setEl('hdrAlerts', 0);
    el('pillAlert').classList.add('hidden');
    try { await fetch(`${API_BASE}/api/alerts`, { method: 'DELETE' }); } catch(e) {}
  });
});

async function loadAlertHistory() {
  try {
    const res = await fetch(`${API_BASE}/api/alerts?limit=10`);
    if (!res.ok) return;
    const data = await res.json();
    (data.alerts || []).forEach(a => addAlertToFeed(a));
    alertCount = data.count || 0;
    setEl('hdrAlerts', alertCount);
  } catch (e) {
    // API not running — no-op (demo mode)
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Charts
// ═══════════════════════════════════════════════════════════════════════════

const EMOTION_COLORS = {
  neutral:  '#94A3B8',
  happy:    '#34D399',
  sad:      '#818CF8',
  angry:    '#F87171',
  fear:     '#FBBF24',
  surprise: '#6EE7F7',
  disgust:  '#A78BFA',
};
const EMOTION_LABELS = ['neutral', 'happy', 'sad', 'angry', 'fear', 'surprise', 'disgust'];

function initCharts() {
  // Emotion donut
  const emotionCtx = el('emotionChart').getContext('2d');
  emotionChart = new Chart(emotionCtx, {
    type: 'doughnut',
    data: {
      labels: EMOTION_LABELS,
      datasets: [{
        data: EMOTION_LABELS.map(e => (e === 'neutral') ? 1 : 0),
        backgroundColor: EMOTION_LABELS.map(e => EMOTION_COLORS[e] + 'CC'),
        borderColor: EMOTION_LABELS.map(e => EMOTION_COLORS[e]),
        borderWidth: 1.5,
        hoverOffset: 6,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '68%',
      plugins: { legend: { display: false }, tooltip: { enabled: true } },
      animation: { duration: 600, easing: 'easeInOutQuart' },
    }
  });

  buildEmotionLegend();

  // Timeline chart
  const tlCtx = el('timelineChart').getContext('2d');
  timelineChart = new Chart(tlCtx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          label: 'Engagement',
          data: [],
          borderColor: '#6EE7F7',
          backgroundColor: 'rgba(110,231,247,0.06)',
          fill: true,
          tension: 0.45,
          pointRadius: 2,
          borderWidth: 2,
        },
        {
          label: 'Anomaly Rate',
          data: [],
          borderColor: '#F87171',
          backgroundColor: 'rgba(248,113,113,0.06)',
          fill: true,
          tension: 0.45,
          pointRadius: 2,
          borderWidth: 2,
        },
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          ticks: { color: '#475569', maxTicksLimit: 8, font: { size: 10 } },
          grid: { color: 'rgba(255,255,255,0.04)' },
        },
        y: {
          min: 0, max: 1,
          ticks: { color: '#475569', font: { size: 10 }, callback: v => (v * 100).toFixed(0) + '%' },
          grid: { color: 'rgba(255,255,255,0.04)' },
        },
      },
      plugins: {
        legend: {
          labels: { color: '#64748B', font: { size: 10 }, boxWidth: 10 }
        }
      },
      animation: { duration: 300 },
    }
  });

  // Pre-load timeline from API
  loadTimelineData();
}

function buildEmotionLegend() {
  const legend = el('emotionLegend');
  EMOTION_LABELS.forEach(emo => {
    const item = document.createElement('div');
    item.className = 'legend-item';
    item.innerHTML = `<span class="legend-dot" style="background:${EMOTION_COLORS[emo]}"></span>${emo}`;
    legend.appendChild(item);
  });
}

function updateEmotionChart(dist) {
  if (!emotionChart) return;
  const data = EMOTION_LABELS.map(e => dist[e] || 0);
  emotionChart.data.datasets[0].data = data;
  emotionChart.update('active');

  // Update dominant emotion in donut center
  const dominant = EMOTION_LABELS.reduce((a, b) => (dist[a] || 0) > (dist[b] || 0) ? a : b);
  el('donutCenter').querySelector('.donut-val').textContent = dominant;
}

function pushTimelinePoint(data) {
  if (!timelineChart) return;
  const now = new Date().toLocaleTimeString([], { hour:'2-digit', minute:'2-digit', second:'2-digit' });
  const engagement = data.engagement_score || (1 - (data.anomaly_rate || 0.05));
  const anomalyRate = data.anomaly_rate || 0.0;

  const labels = timelineChart.data.labels;
  const eng    = timelineChart.data.datasets[0].data;
  const anom   = timelineChart.data.datasets[1].data;

  labels.push(now);
  eng.push(Math.min(1, Math.max(0, engagement)));
  anom.push(Math.min(1, Math.max(0, anomalyRate)));

  // Keep 60 points
  if (labels.length > 60) {
    labels.shift();
    eng.shift();
    anom.shift();
  }

  timelineChart.update('none');
}

async function loadTimelineData() {
  try {
    const res = await fetch(`${API_BASE}/api/analytics/timeline?points=40`);
    if (!res.ok) return;
    const data = await res.json();
    (data.data || []).forEach(pt => {
      const t = new Date(pt.t * 1000).toLocaleTimeString([], { hour:'2-digit', minute:'2-digit' });
      timelineChart.data.labels.push(t);
      timelineChart.data.datasets[0].data.push(pt.engagement);
      timelineChart.data.datasets[1].data.push(pt.anomaly_rate);
    });
    timelineChart.update();
  } catch (e) {
    seedDemoTimeline();
  }
}

function seedDemoTimeline() {
  if (!timelineChart) return;
  for (let i = 30; i >= 0; i--) {
    const t = new Date(Date.now() - i * 10000).toLocaleTimeString([], { hour:'2-digit', minute:'2-digit', second:'2-digit' });
    timelineChart.data.labels.push(t);
    timelineChart.data.datasets[0].data.push(+(0.6 + 0.2 * Math.sin(i * 0.3) + (Math.random() - 0.5) * 0.05).toFixed(3));
    timelineChart.data.datasets[1].data.push(+(Math.max(0, 0.05 + 0.07 * Math.cos(i * 0.25) + (Math.random() * 0.02))).toFixed(3));
  }
  timelineChart.update();
}

// ═══════════════════════════════════════════════════════════════════════════
// Analytics Polling (fallback when no active WebSocket data)
// ═══════════════════════════════════════════════════════════════════════════

function startAnalyticsPoll() {
  setInterval(async () => {
    if (!isConnected) return;
    try {
      const res = await fetch(`${API_BASE}/api/analytics/summary`);
      if (!res.ok) return;
      const data = await res.json();
      setEl('hdrPersons', data.active_persons ?? '—');
      setEl('hdrFps', data.avg_fps ? data.avg_fps.toFixed(1) : '—');
      if (data.emotion_distribution) updateEmotionChart(data.emotion_distribution);
    } catch (e) {}
  }, POLL_INTERVAL);
}

// ═══════════════════════════════════════════════════════════════════════════
// Upload Flow
// ═══════════════════════════════════════════════════════════════════════════

function initUpload() {
  const area     = el('uploadArea');
  const fileInp  = el('videoFile');
  const btn      = el('analyzeBtn');

  // Click to open file dialog
  area.addEventListener('click', () => fileInp.click());

  // Drag-and-drop
  area.addEventListener('dragover', (e) => { e.preventDefault(); area.classList.add('drag-over'); });
  area.addEventListener('dragleave', () => area.classList.remove('drag-over'));
  area.addEventListener('drop', (e) => {
    e.preventDefault();
    area.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('video/')) setSelectedFile(file);
  });

  fileInp.addEventListener('change', () => {
    if (fileInp.files[0]) setSelectedFile(fileInp.files[0]);
  });

  btn.addEventListener('click', uploadAndAnalyze);
}

function setSelectedFile(file) {
  selectedFile = file;
  el('uploadArea').querySelector('.upload-hint').textContent = `✓ ${file.name}`;
  el('uploadArea').querySelector('.upload-sub').textContent =
    `${(file.size / 1024 / 1024).toFixed(1)} MB · ${file.type}`;
  el('analyzeBtn').disabled = false;
}

async function uploadAndAnalyze() {
  if (!selectedFile) return;

  el('analyzeBtn').disabled = true;
  el('analyzeBtn').innerHTML = '<span class="btn-icon">⏳</span> Uploading…';
  el('progressWrap').classList.remove('hidden');
  el('progressBar').style.width = '25%';
  el('progressLabel').textContent = 'Uploading video…';

  try {
    const form = new FormData();
    form.append('file', selectedFile);

    const res = await fetch(`${API_BASE}/api/upload`, { method: 'POST', body: form });

    el('progressBar').style.width = '60%';
    el('progressLabel').textContent = 'Analysis running via WebSocket…';

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Upload failed' }));
      throw new Error(err.detail || 'Upload failed');
    }

    el('analyzeBtn').innerHTML = '<span class="btn-icon">📡</span> Streaming…';
    el('progressBar').style.width = '100%';
    el('progressLabel').textContent = 'Results streaming live ➜ watch the dashboard';

    // Show video placeholder message
    el('videoPlaceholder').innerHTML = `
      <div class="placeholder-icon">🔄</div>
      <p>Processing: ${escHtml(selectedFile.name)}</p>
      <span>Live results appear on this dashboard via WebSocket</span>
    `;

  } catch (e) {
    el('progressLabel').textContent = `Error: ${e.message}`;
    el('progressBar').style.background = 'var(--rose)';
    el('analyzeBtn').disabled = false;
    el('analyzeBtn').innerHTML = '<span class="btn-icon">▶</span> Retry';
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Demo / Seed State
// ═══════════════════════════════════════════════════════════════════════════

function seedDemoState() {
  // Pre-populate persons list with demo data
  setTimeout(() => {
    if (frameCount === 0) {
      const demoAnoms = {};
      [1, 4, 7, 12, 19].forEach((id, i) => {
        const score = [0.12, 0.78, 0.91, 0.34, 0.56][i];
        demoAnoms[id] = {
          score,
          severity: score > 0.82 ? 'high' : score > 0.72 ? 'medium' : 'none',
          alert: score > 0.72,
        };
      });
      updatePersonsList([1, 4, 7, 12, 19], demoAnoms);
      setEl('hdrPersons', 28);
      setEl('hdrFps', '13.4');

      // Demo emotion distribution
      updateEmotionChart({
        neutral: 0.42, happy: 0.22, sad: 0.11,
        angry: 0.08, fear: 0.05, surprise: 0.09, disgust: 0.03
      });
    }
  }, 800);
}

// ═══════════════════════════════════════════════════════════════════════════
// UI Utilities
// ═══════════════════════════════════════════════════════════════════════════

function el(id) { return document.getElementById(id); }
function setEl(id, text) {
  const node = document.getElementById(id);
  if (node) node.textContent = text;
}

function setLiveStatus(online) {
  const badge = el('liveBadge');
  const label = el('liveStatus');
  if (online) {
    badge.classList.add('online');
    label.textContent = 'LIVE';
  } else {
    badge.classList.remove('online');
    label.textContent = 'OFFLINE';
  }
}

function setWsDot(state) {
  const dot = el('dotWs');
  dot.className = `status-dot dot-${state === 'green' ? 'green' : state === 'red' ? 'red' : 'amber'}`;
}

function updateTimestamp(ts) {
  if (!ts) return;
  const d = new Date(ts > 1e10 ? ts : ts * 1000);
  const label = el('videoTimestamp');
  if (label) label.textContent = d.toLocaleTimeString();
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
