/* ===========================
   DASHBOARD — Live Monitoring
   =========================== */

const API = '';
let tempChart = null;
let currentChart = null;
let tempData = [];
let currentData = [];
let timeLabels = [];
let dashboardInitialized = false;
let dashboardIntervals = [];
let systemInfoFetched = false;

function initDashboard() {
  if (!dashboardInitialized) {
    initCharts();
    dashboardInitialized = true;
  }

  // Fetch system info once on first init
  if (!systemInfoFetched) {
    fetchSystemInfo();
    systemInfoFetched = true;
  }

  pollLiveData();
  pollPrediction();
  pollSensorHealth();

  const liveInterval = setInterval(pollLiveData, 2000);
  const predInterval = setInterval(pollPrediction, 2000);
  const healthInterval = setInterval(pollSensorHealth, 5000);

  if (typeof activeIntervals !== 'undefined') {
    activeIntervals.push(liveInterval, predInterval, healthInterval);
  }
}

function initCharts() {
  const sharedOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 400, easing: 'easeOutQuart' },
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: '#111827',
        titleFont: { family: "'DM Sans', sans-serif", size: 12 },
        bodyFont: { family: "'Space Mono', monospace", size: 12 },
        padding: 10,
        cornerRadius: 8,
        displayColors: false
      }
    },
    scales: {
      x: {
        grid: { display: false },
        ticks: {
          font: { family: "'Space Mono', monospace", size: 10 },
          color: '#9CA3AF',
          maxRotation: 0
        },
        border: { display: false }
      },
      y: {
        grid: { color: '#F3F4F6', drawBorder: false },
        ticks: {
          font: { family: "'Space Mono', monospace", size: 11 },
          color: '#9CA3AF',
          padding: 8
        },
        border: { display: false }
      }
    },
    interaction: { intersect: false, mode: 'index' }
  };

  const tempCtx = document.getElementById('temp-chart');
  if (tempCtx) {
    tempChart = new Chart(tempCtx, {
      type: 'line',
      data: {
        labels: timeLabels,
        datasets: [{
          label: 'Temperature (°C)',
          data: tempData,
          borderColor: '#2563EB',
          backgroundColor: 'rgba(37, 99, 235, 0.08)',
          borderWidth: 2.5,
          fill: true,
          tension: 0.4,
          pointRadius: 0,
          pointHoverRadius: 5,
          pointHoverBackgroundColor: '#2563EB'
        }]
      },
      options: {
        ...sharedOptions,
        scales: {
          ...sharedOptions.scales,
          y: {
            ...sharedOptions.scales.y,
            min: 0,
            max: 60
          }
        }
      }
    });
  }

  const currentCtx = document.getElementById('current-chart');
  if (currentCtx) {
    currentChart = new Chart(currentCtx, {
      type: 'line',
      data: {
        labels: timeLabels,
        datasets: [{
          label: 'Current (A)',
          data: currentData,
          borderColor: '#D97706',
          backgroundColor: 'rgba(217, 119, 6, 0.08)',
          borderWidth: 2.5,
          fill: true,
          tension: 0.4,
          pointRadius: 0,
          pointHoverRadius: 5,
          pointHoverBackgroundColor: '#D97706'
        }]
      },
      options: {
        ...sharedOptions,
        scales: {
          ...sharedOptions.scales,
          y: {
            ...sharedOptions.scales.y,
            min: 0,
            max: 5
          }
        }
      }
    });
  }
}

async function pollLiveData() {
  try {
    const res = await fetch(API + '/api/live-data');
    if (!res.ok) throw new Error('Live data fetch failed');
    const data = await res.json();

    updateSystemStatus(true);

    if (!data.readings || data.readings.length === 0) return;

    const latest = data.readings[0];

    const tempEl = document.getElementById('metric-temp');
    const humidityEl = document.getElementById('metric-humidity');
    const currentEl = document.getElementById('metric-current');
    const vibLed = document.getElementById('led-vibration');
    const vibText = document.getElementById('metric-vibration');
    const flameLed = document.getElementById('led-flame');
    const flameText = document.getElementById('metric-flame');

    if (tempEl) tempEl.textContent = parseFloat(latest.temp).toFixed(1) + ' °C';
    if (humidityEl) humidityEl.textContent = parseFloat(latest.humidity || 0).toFixed(1) + ' %';
    if (currentEl) currentEl.textContent = parseFloat(latest.current).toFixed(2) + ' A';

    if (vibLed) {
      vibLed.className = latest.vibration ? 'led led-active-vibration' : 'led';
    }
    if (vibText) vibText.textContent = latest.vibration ? 'Active' : 'Inactive';

    if (flameLed) {
      flameLed.className = latest.flame ? 'led led-active-flame' : 'led';
    }
    if (flameText) flameText.textContent = latest.flame ? 'Detected' : 'Clear';

    const ts = latest.timestamp;
    let timeStr = '';
    try {
      const d = new Date(ts);
      timeStr = d.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
    } catch (e) {
      timeStr = ts ? ts.substring(11, 19) : '--:--:--';
    }

    if (timeLabels.length === 0 || timeLabels[timeLabels.length - 1] !== timeStr) {
      timeLabels.push(timeStr);
      tempData.push(parseFloat(latest.temp));
      currentData.push(parseFloat(latest.current));

      if (timeLabels.length > 20) {
        timeLabels.shift();
        tempData.shift();
        currentData.shift();
      }

      if (tempChart) tempChart.update('none');
      if (currentChart) currentChart.update('none');
    }

  } catch (err) {
    console.error('Dashboard pollLiveData error:', err);
    updateSystemStatus(false);
  }
}

async function pollPrediction() {
  try {
    const res = await fetch(API + '/api/prediction');
    if (!res.ok) throw new Error('Prediction fetch failed');
    const data = await res.json();

    const badge = document.getElementById('prediction-badge');
    const badgeText = document.getElementById('prediction-text');
    const confidenceEl = document.getElementById('confidence-value');
    const tsEl = document.getElementById('prediction-timestamp');
    const flameBanner = document.getElementById('flame-banner');

    if (badge) {
      badge.className = 'status-badge';
      const pred = (data.prediction || 'Initializing').toUpperCase();
      if (pred.includes('CRITICAL') || pred.includes('FLAME')) {
        badge.classList.add('badge-critical');
      } else if (pred.includes('FAULT')) {
        badge.classList.add('badge-fault');
      } else if (pred.includes('WARNING')) {
        badge.classList.add('badge-warning');
      } else {
        badge.classList.add('badge-normal');
      }
    }

    if (badgeText) {
      badgeText.textContent = (data.prediction || 'Initializing').replace(/_/g, ' ');
    }

    if (confidenceEl) {
      const conf = data.confidence != null ? (data.confidence * 100).toFixed(1) : '0';
      confidenceEl.textContent = conf + '%';
    }

    if (tsEl && data.timestamp) {
      try {
        const d = new Date(data.timestamp);
        tsEl.textContent = 'Last updated: ' + d.toLocaleTimeString();
      } catch (e) {
        tsEl.textContent = 'Last updated: ' + data.timestamp;
      }
    }

    if (flameBanner) {
      const pred = (data.prediction || '').toUpperCase();
      if (pred.includes('CRITICAL') || pred.includes('FLAME')) {
        flameBanner.classList.remove('hidden');
      } else {
        flameBanner.classList.add('hidden');
      }
    }

  } catch (err) {
    console.error('Dashboard pollPrediction error:', err);
  }
}

async function pollSensorHealth() {
  try {
    const res = await fetch(API + '/api/sensor-health');
    if (!res.ok) return;
    const health = await res.json();

    const STATUS_LABELS = { ok: 'Active', fallback: 'Fallback', failed: 'Offline' };
    const sensors = ['temp', 'humidity', 'current', 'vibration', 'flame'];

    sensors.forEach(sensor => {
      const dot = document.getElementById('health-' + sensor);
      const text = document.getElementById('health-' + sensor + '-text');
      const wrap = document.getElementById('health-' + sensor + '-wrap');
      const status = health[sensor] || 'ok';

      if (dot) dot.className = 'sensor-dot sensor-dot-' + status;
      if (text) {
        text.textContent = STATUS_LABELS[status] || 'Active';
        text.className = 'sensor-status-text sensor-text-' + status;
      }
      if (wrap) wrap.className = 'sensor-status status-' + status;
    });
  } catch (err) {
    console.error('Sensor health poll error:', err);
  }
}

async function fetchSystemInfo() {
  try {
    const res = await fetch(API + '/api/system-info');
    if (!res.ok) return;
    const info = await res.json();
    // Store data source for status display
    window._dataSource = info.data_source || 'simulator';
  } catch (err) {
    console.error('System info fetch error:', err);
    window._dataSource = 'simulator';
  }
}

function updateSystemStatus(online) {
  const dot = document.getElementById('status-dot');
  const text = document.getElementById('status-text');
  if (dot) {
    dot.className = online ? 'status-dot online' : 'status-dot offline';
  }
  if (text) {
    if (online) {
      const source = window._dataSource || 'simulator';
      const label = source === 'hardware' ? 'Hardware' : 'Simulator';
      text.textContent = 'Live — ' + label;
    } else {
      text.textContent = 'Offline';
    }
  }
}
