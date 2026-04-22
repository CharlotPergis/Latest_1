// ----------------------------
// Breaker Monitoring Dashboard JS - WORKING WITH LOCALSTORAGE
// ----------------------------

const BACKEND_URL = '';

// Add alert styles
const alertStyle = document.createElement('style');
alertStyle.textContent = `
@keyframes slideIn {
    from { transform: translateX(100%); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
}
@keyframes slideOut {
    from { transform: translateX(0); opacity: 1; }
    to { transform: translateX(100%); opacity: 0; }
}
`;
document.head.appendChild(alertStyle);

// Data storage
let tempHistory = [];
let currentHistory = [];
let historyData = [];
const MAX_HISTORY = 20;

// Canvas elements
let tempCanvas, currentCanvas;
let tempCtx, currentCtx;

// Initialize graphs
function initGraphs() {
  tempCanvas = document.getElementById("tempGraph");
  currentCanvas = document.getElementById("currentGraph");
  
  if (tempCanvas && currentCanvas) {
    tempCtx = tempCanvas.getContext("2d");
    currentCtx = currentCanvas.getContext("2d");
    
    const resizeCanvas = () => {
      const width = tempCanvas.parentElement.clientWidth;
      tempCanvas.width = width;
      tempCanvas.height = 50;
      currentCanvas.width = width;
      currentCanvas.height = 50;
      drawGraphs();
    };
    
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
  }
}

function drawMinimalGraph(ctx, data, color, maxValue) {
  if (!ctx || data.length === 0) return;
  
  const width = ctx.canvas.width;
  const height = ctx.canvas.height;
  
  ctx.clearRect(0, 0, width, height);
  
  if (data.length >= 2) {
    const step = width / (data.length - 1);
    
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    
    let firstPoint = true;
    for (let i = 0; i < data.length; i++) {
      const x = i * step;
      const y = height - (data[i] / maxValue) * height;
      const clampedY = Math.max(0, Math.min(height, y));
      
      if (firstPoint) {
        ctx.moveTo(x, clampedY);
        firstPoint = false;
      } else {
        ctx.lineTo(x, clampedY);
      }
    }
    ctx.stroke();
  }
}

function drawGraphs() {
  if (tempCtx) drawMinimalGraph(tempCtx, tempHistory, "#0ea5e9", 100);
  if (currentCtx) drawMinimalGraph(currentCtx, currentHistory, "#facc15", 50);
}

// CRITICAL FUNCTION: Save to localStorage
function saveToLocalStorage() {
  try {
    // Save recent history
    localStorage.setItem('breakerHistory', JSON.stringify(historyData));
    console.log(`💾 Saved ${historyData.length} logs to breakerHistory`);
  } catch(e) {
    console.error("Error saving to localStorage:", e);
  }
}

// CRITICAL FUNCTION: Add to full history
function addToFullHistory(data) {
  try {
    const fullHistory = JSON.parse(localStorage.getItem('breakerFullHistory') || '[]');
    fullHistory.push({
      timestamp: new Date().toLocaleString(),
      temperature: data.temperature,
      current: data.current,
      breakerState: data.breakerState,
      systemStatus: data.systemStatus
    });
    
    // Keep last 500 logs
    while (fullHistory.length > 500) fullHistory.shift();
    localStorage.setItem('breakerFullHistory', JSON.stringify(fullHistory));
    console.log(`📊 Full history now has ${fullHistory.length} total logs`);
  } catch(e) {
    console.error("Error saving to full history:", e);
  }
}

// Add to history
function addToHistory(data) {
  const logEntry = {
    timestamp: new Date().toLocaleString(),
    time: data.time || new Date().toLocaleTimeString(),
    date: data.date || new Date().toLocaleDateString(),
    temperature: data.temperature,
    current: data.current,
    breakerState: data.breakerState,
    systemStatus: data.systemStatus
  };
  
  // Add to recent history (keep last 10)
  historyData.unshift(logEntry);
  if (historyData.length > 10) historyData.pop();
  
  // Save to localStorage
  saveToLocalStorage();
  
  // Add to full history
  addToFullHistory(data);
  
  console.log(`📝 Log added: ${logEntry.breakerState} at ${logEntry.time}`);
  renderHistoryTable();
}

// Render history table
function renderHistoryTable() {
  const logBody = document.getElementById("log-body");
  if (!logBody) return;
  
  logBody.innerHTML = "";
  
  if (historyData.length === 0) {
    logBody.innerHTML = '<tr><td colspan="5" style="text-align:center; padding:20px;">⏳ Waiting for data...<br><small style="color:#64748b;">Logs will appear here</small></td></tr>';
    return;
  }
  
  historyData.forEach(entry => {
    const row = document.createElement("tr");
    let icon = entry.breakerState === "Normal" ? "✅" : (entry.breakerState === "Overload" ? "⚠️" : "🔥");
    row.innerHTML = `
      <td>${entry.time}</td>
      <td><strong>${entry.temperature}°C</strong></td>
      <td><strong>${entry.current}A</strong></td>
      <td>${icon} ${entry.breakerState}</td>
      <td>${entry.systemStatus}</td>
    `;
    logBody.appendChild(row);
  });
}

// Load saved history on startup
function loadSavedHistory() {
  try {
    const savedHistory = localStorage.getItem('breakerHistory');
    if (savedHistory) {
      historyData = JSON.parse(savedHistory);
      renderHistoryTable();
      console.log(`✅ Loaded ${historyData.length} recent logs from storage`);
    } else {
      console.log("📭 No saved history found, starting fresh");
    }
    
    const fullHistory = localStorage.getItem('breakerFullHistory');
    if (fullHistory) {
      const parsed = JSON.parse(fullHistory);
      console.log(`📊 Full history has ${parsed.length} total logs`);
    } else {
      console.log("📭 No full history found, will create new one");
      localStorage.setItem('breakerFullHistory', JSON.stringify([]));
    }
  } catch(e) {
    console.error("Error loading history:", e);
    historyData = [];
  }
}

// Mitigation functions
function generateMitigation(temp, current, breakerState) {
  let suggestion = "", action = "", riskLevel = "normal", riskLabel = "✅ NORMAL";
  
  if (breakerState === "Overheating" || temp > 75) {
    riskLevel = "critical";
    riskLabel = "🔥 CRITICAL";
    suggestion = `🔥 CRITICAL: ${temp}°C / ${current}A - Immediate intervention required!`;
    action = `🚨 EMERGENCY: Isolate circuit immediately!`;
  } 
  else if (breakerState === "Overload" || temp > 62 || current > 21) {
    riskLevel = "caution";
    riskLabel = "⚠️ CAUTION";
    suggestion = `⚠️ WARNING: ${temp}°C / ${current}A - Approaching limits. Reduce load now.`;
    action = `⚙️ Shed non-critical loads immediately`;
  }
  else {
    riskLevel = "normal";
    riskLabel = "🟢 NORMAL";
    suggestion = `✅ SAFE: ${temp}°C, ${current}A - Operating within normal range.`;
    action = `📊 Continue monitoring`;
  }
  
  return { suggestion, action, riskLevel, riskLabel };
}

function updateMitigationUI(temp, current, breakerState) {
  const { suggestion, action, riskLevel, riskLabel } = generateMitigation(temp, current, breakerState);
  const suggestionDiv = document.getElementById("suggestion-main");
  const actionSpan = document.getElementById("action-text");
  const riskContainer = document.getElementById("risk-badge-container");
  const mitiCard = document.getElementById("mitigationCard");
  
  if (suggestionDiv) suggestionDiv.innerHTML = `💡 ${suggestion}`;
  if (actionSpan) actionSpan.innerText = action;
  
  let badgeHtml = `<span class="risk-badge-compact ${riskLevel === 'critical' ? 'risk-high' : (riskLevel === 'caution' ? 'risk-mid' : 'risk-normal')}">${riskLabel}</span>`;
  if (riskContainer) riskContainer.innerHTML = badgeHtml;
}

function updateTimeToTrip(timeToTrip, breakerState) {
  const container = document.getElementById("time-to-trip-container");
  const valueEl = document.getElementById("time-to-trip-value");
  const noteEl = document.getElementById("time-to-trip-note");
  
  if (breakerState === "Overload" && timeToTrip && timeToTrip.formatted) {
    container.style.display = "block";
    valueEl.innerHTML = timeToTrip.formatted;
    noteEl.innerHTML = `Urgency: ${timeToTrip.urgency} - ${(timeToTrip.overload_ratio * 100).toFixed(0)}% overload`;
    container.className = `time-to-trip-container urgency-${timeToTrip.urgency.toLowerCase()}`;
  } else {
    container.style.display = "none";
  }
}

// Alert functions
let lastAlertTime = 0;
const ALERT_COOLDOWN_MS = 300000;

async function sendAlertToBackend(temperature, current, breakerState) {
  const shouldAlert = breakerState === "Overheating" || temperature > 75;
  if (!shouldAlert) return;
  
  const currentTime = Date.now();
  if (currentTime - lastAlertTime < ALERT_COOLDOWN_MS) return;
  
  try {
    const response = await fetch(`${BACKEND_URL}/api/check-alert`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ temperature, current, ambient_temp: 25 })
    });
    const result = await response.json();
    if (result.success && result.alert_sent) {
      lastAlertTime = currentTime;
      console.log("✅ Alert sent!");
      showNotification("Email Alert Sent!", result.alert_type);
    }
  } catch (error) {
    console.error("Failed to send alert:", error);
  }
}

function showNotification(message, type) {
  const notification = document.createElement('div');
  const bgColor = type === 'overheating' ? '#ef4444' : '#f97316';
  notification.innerHTML = `
    <div style="position:fixed; top:20px; right:20px; background:${bgColor}; color:white; padding:16px 24px; border-radius:12px; z-index:10000; animation:slideIn 0.3s ease-out; font-weight:600;">
      <div style="display:flex; align-items:center; gap:12px;">
        <span style="font-size:24px;">${type === 'overheating' ? '🔥' : '⚠️'}</span>
        <div>
          <div style="font-size:16px; margin-bottom:4px;">${type === 'overheating' ? 'CRITICAL ALERT!' : 'Prevention Alert'}</div>
          <div style="font-size:12px;">${message}</div>
        </div>
      </div>
    </div>
  `;
  document.body.appendChild(notification);
  setTimeout(() => notification.remove(), 5000);
}

// Main update function
function updateDashboard(data) {
  console.log(`📊 Updating: ${data.breakerState} | ${data.temperature}°C / ${data.current}A`);
  
  // Update graphs
  tempHistory.push(data.temperature);
  currentHistory.push(data.current);
  if (tempHistory.length > MAX_HISTORY) tempHistory.shift();
  if (currentHistory.length > MAX_HISTORY) currentHistory.shift();
  drawGraphs();
  
  // Update displays
  document.getElementById("temperature-value").textContent = data.temperature.toFixed(1);
  document.getElementById("current-value").textContent = data.current.toFixed(1);
  document.getElementById("temperature-note").textContent = `${data.date} ${data.time}`;
  document.getElementById("current-note").textContent = `${data.date} ${data.time}`;
  
  const breakerEl = document.getElementById("breaker-state");
  breakerEl.textContent = data.breakerState;
  breakerEl.className = `kpi__value state ${data.breakerState}`;
  
  const indicator = document.getElementById("breaker-indicator");
  if (indicator) {
    indicator.className = "breaker-status-indicator";
    if (data.breakerState === "Normal") indicator.classList.add("normal-bg");
    else if (data.breakerState === "Overload") indicator.classList.add("overload-bg");
    else indicator.classList.add("overheating-bg");
  }
  
  document.getElementById("state-note").textContent = `${data.date} ${data.time}`;
  
  const statusHeader = document.getElementById("system-status-header");
  if (statusHeader) statusHeader.textContent = data.systemStatus;
  
  updateMitigationUI(data.temperature, data.current, data.breakerState);
  updateTimeToTrip(data.time_to_trip, data.breakerState);
  
  // CRITICAL: Add to history (this saves to localStorage)
  addToHistory(data);
  
  // Send alert
  sendAlertToBackend(data.temperature, data.current, data.breakerState);
}

// Fetch data
let isFetching = false;

async function refreshDashboard() {
  if (isFetching) return;
  isFetching = true;
  
  try {
    const response = await fetch(`${BACKEND_URL}/api/simulate`);
    const data = await response.json();
    updateDashboard(data);
  } catch (error) {
    console.error("Error fetching data:", error);
  } finally {
    isFetching = false;
  }
}

// Initialize
window.addEventListener('load', () => {
  console.log("🚀 Dashboard initializing...");
  initGraphs();
  loadSavedHistory();
  refreshDashboard();
  setInterval(refreshDashboard, 3000);
});

// Manual refresh button
const refreshBtn = document.getElementById("refresh-btn");
if (refreshBtn) {
  refreshBtn.addEventListener("click", () => {
    refreshDashboard();
    refreshBtn.style.transform = "scale(0.97)";
    setTimeout(() => { refreshBtn.style.transform = ""; }, 120);
  });
}