/* ===========================
   ANALYTICS TAB
   =========================== */

let analyticsTempChart = null;
let analyticsCurrentChart = null;
let analyticsDoughnutChart = null;
let analyticsSeverityChart = null;

async function initAnalytics() {
  try {
    const [historyRes, alertsRes] = await Promise.all([
      fetch('http://localhost:8000/api/history?limit=200'),
      fetch('http://localhost:8000/api/alerts')
    ]);

    if (!historyRes.ok || !alertsRes.ok) throw new Error('Analytics data fetch failed');

    const historyData = await historyRes.json();
    const alertsData = await alertsRes.json();

    const readings = historyData.readings || [];
    const alerts = alertsData.alerts || [];

    updateSummaryStats(readings, alerts);
    renderTempTrend(readings);
    renderCurrentTrend(readings);
    renderDistribution(readings);
    renderSeverityChart(alerts);

  } catch (err) {
    console.error('Analytics init error:', err);
  }
}

function updateSummaryStats(readings, alerts) {
  const maxTempEl = document.getElementById('stat-max-temp');
  const avgCurrentEl = document.getElementById('stat-avg-current');
  const totalReadingsEl = document.getElementById('stat-total-readings');
  const totalFaultsEl = document.getElementById('stat-total-faults');

  if (readings.length > 0) {
    const temps = readings.map(r => parseFloat(r.temp));
    const currents = readings.map(r => parseFloat(r.current));
    const maxTemp = Math.max(...temps);
    const avgCurrent = currents.reduce((a, b) => a + b, 0) / currents.length;
    const faults = readings.filter(r => parseFloat(r.temp) >= 68).length;

    if (maxTempEl) maxTempEl.textContent = maxTemp.toFixed(1) + ' °C';
    if (avgCurrentEl) avgCurrentEl.textContent = avgCurrent.toFixed(1) + ' mA';
    if (totalReadingsEl) totalReadingsEl.textContent = readings.length;
    if (totalFaultsEl) totalFaultsEl.textContent = faults;
  } else {
    if (maxTempEl) maxTempEl.textContent = '— °C';
    if (avgCurrentEl) avgCurrentEl.textContent = '— mA';
    if (totalReadingsEl) totalReadingsEl.textContent = '0';
    if (totalFaultsEl) totalFaultsEl.textContent = '0';
  }
}

function renderTempTrend(readings) {
  const ctx = document.getElementById('analytics-temp-chart');
  if (!ctx) return;

  const labels = readings.map(r => {
    try {
      const d = new Date(r.timestamp);
      return d.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
    } catch (e) { return ''; }
  });
  const data = readings.map(r => parseFloat(r.temp));

  if (analyticsTempChart) {
    analyticsTempChart.destroy();
    analyticsTempChart = null;
  }

  analyticsTempChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        label: 'Temperature (°C)',
        data: data,
        borderColor: '#2563EB',
        backgroundColor: 'rgba(37, 99, 235, 0.06)',
        borderWidth: 2,
        fill: true,
        tension: 0.4,
        pointRadius: 0,
        pointHoverRadius: 4
      }]
    },
    options: getLineChartOptions()
  });
}

function renderCurrentTrend(readings) {
  const ctx = document.getElementById('analytics-current-chart');
  if (!ctx) return;

  const labels = readings.map(r => {
    try {
      const d = new Date(r.timestamp);
      return d.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
    } catch (e) { return ''; }
  });
  const data = readings.map(r => parseFloat(r.current));

  if (analyticsCurrentChart) {
    analyticsCurrentChart.destroy();
    analyticsCurrentChart = null;
  }

  analyticsCurrentChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        label: 'Current (mA)',
        data: data,
        borderColor: '#D97706',
        backgroundColor: 'rgba(217, 119, 6, 0.06)',
        borderWidth: 2,
        fill: true,
        tension: 0.4,
        pointRadius: 0,
        pointHoverRadius: 4
      }]
    },
    options: getLineChartOptions()
  });
}

function renderDistribution(readings) {
  const ctx = document.getElementById('analytics-doughnut-chart');
  if (!ctx) return;

  let normal = 0, warning = 0, fault = 0, critical = 0;
  readings.forEach(r => {
    const temp = parseFloat(r.temp);
    if (temp < 55) normal++;
    else if (temp < 68) warning++;
    else if (temp < 80) fault++;
    else critical++;
  });

  if (analyticsDoughnutChart) {
    analyticsDoughnutChart.destroy();
    analyticsDoughnutChart = null;
  }

  analyticsDoughnutChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Normal', 'Warning', 'Fault', 'Critical Flame'],
      datasets: [{
        data: [normal, warning, fault, critical],
        backgroundColor: [
          'rgba(22, 163, 74, 0.85)',
          'rgba(217, 119, 6, 0.85)',
          'rgba(220, 38, 38, 0.85)',
          'rgba(127, 29, 29, 0.9)'
        ],
        borderColor: '#FFFFFF',
        borderWidth: 3,
        hoverOffset: 8
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '60%',
      plugins: {
        legend: {
          position: 'bottom',
          labels: {
            font: { family: "'DM Sans', sans-serif", size: 12 },
            color: '#6B7280',
            padding: 16,
            usePointStyle: true,
            pointStyleWidth: 12
          }
        },
        tooltip: {
          backgroundColor: '#111827',
          titleFont: { family: "'DM Sans', sans-serif", size: 12 },
          bodyFont: { family: "'Space Mono', monospace", size: 12 },
          padding: 10,
          cornerRadius: 8,
          callbacks: {
            label: function(ctx) {
              const total = ctx.dataset.data.reduce((a, b) => a + b, 0);
              const pct = total > 0 ? ((ctx.parsed / total) * 100).toFixed(1) : 0;
              return ctx.label + ': ' + ctx.parsed + ' (' + pct + '%)';
            }
          }
        }
      }
    }
  });
}

function renderSeverityChart(alerts) {
  const ctx = document.getElementById('analytics-severity-chart');
  if (!ctx) return;

  const severityCounts = { info: 0, warning: 0, fault: 0, critical: 0 };
  alerts.forEach(a => {
    if (severityCounts.hasOwnProperty(a.severity)) {
      severityCounts[a.severity]++;
    }
  });

  if (analyticsSeverityChart) {
    analyticsSeverityChart.destroy();
    analyticsSeverityChart = null;
  }

  analyticsSeverityChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['Info', 'Warning', 'Fault', 'Critical'],
      datasets: [{
        label: 'Count',
        data: [severityCounts.info, severityCounts.warning, severityCounts.fault, severityCounts.critical],
        backgroundColor: [
          'rgba(37, 99, 235, 0.8)',
          'rgba(217, 119, 6, 0.8)',
          'rgba(220, 38, 38, 0.8)',
          'rgba(127, 29, 29, 0.9)'
        ],
        borderRadius: 6,
        barThickness: 32
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#111827',
          titleFont: { family: "'DM Sans', sans-serif", size: 12 },
          bodyFont: { family: "'Space Mono', monospace", size: 12 },
          padding: 10,
          cornerRadius: 8
        }
      },
      scales: {
        x: {
          grid: { color: '#F3F4F6', drawBorder: false },
          ticks: {
            font: { family: "'Space Mono', monospace", size: 11 },
            color: '#9CA3AF',
            stepSize: 1
          },
          border: { display: false }
        },
        y: {
          grid: { display: false },
          ticks: {
            font: { family: "'DM Sans', sans-serif", size: 12 },
            color: '#374151'
          },
          border: { display: false }
        }
      }
    }
  });
}

function getLineChartOptions() {
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 400 },
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
          font: { family: "'Space Mono', monospace", size: 9 },
          color: '#9CA3AF',
          maxRotation: 0,
          maxTicksLimit: 10
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
}
