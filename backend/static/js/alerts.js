/* ===========================
   ALERTS TAB
   =========================== */

let allAlerts = [];
let currentFilter = 'all';

async function initAlerts() {
  setupFilterButtons();
  await fetchAlerts();

  const interval = setInterval(fetchAlerts, 5000);
  if (typeof activeIntervals !== 'undefined') {
    activeIntervals.push(interval);
  }
}

function setupFilterButtons() {
  const filterBtns = document.querySelectorAll('.filter-btn');
  filterBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      filterBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentFilter = btn.getAttribute('data-filter');
      renderAlertsTable(currentFilter);
    });
  });
}

async function fetchAlerts() {
  try {
    const res = await fetch('/api/alerts');
    if (!res.ok) throw new Error('Alerts fetch failed');
    const data = await res.json();

    allAlerts = data.alerts || [];

    const totalEl = document.getElementById('alert-total');
    const activeEl = document.getElementById('alert-active');
    const criticalEl = document.getElementById('alert-critical');

    if (totalEl) totalEl.textContent = allAlerts.length;
    if (activeEl) activeEl.textContent = allAlerts.filter(a => a.status === 'active').length;
    if (criticalEl) criticalEl.textContent = allAlerts.filter(a => a.severity === 'critical').length;

    renderAlertsTable(currentFilter);

  } catch (err) {
    console.error('Fetch alerts error:', err);
  }
}

function renderAlertsTable(filter) {
  const tbody = document.getElementById('alerts-table-body');
  const table = document.getElementById('alerts-table');
  const emptyState = document.getElementById('alerts-empty');

  if (!tbody) return;

  let filtered = allAlerts;
  if (filter === 'active') {
    filtered = allAlerts.filter(a => a.status === 'active');
  } else if (filter === 'resolved') {
    filtered = allAlerts.filter(a => a.status === 'resolved');
  }

  if (filtered.length === 0) {
    if (table) table.style.display = 'none';
    if (emptyState) emptyState.classList.remove('hidden');
    return;
  }

  if (table) table.style.display = 'table';
  if (emptyState) emptyState.classList.add('hidden');

  let html = '';
  filtered.forEach(alert => {
    const rowClass = 'row-' + (alert.severity || 'info');
    const sevClass = 'sev-' + (alert.severity || 'info');

    let timeFormatted = '—';
    try {
      const d = new Date(alert.timestamp);
      const time = d.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
      const date = d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
      timeFormatted = time + ' — ' + date;
    } catch (e) {
      timeFormatted = alert.timestamp || '—';
    }

    const statusHtml = alert.status === 'active'
      ? '<span class="status-active-text">Active</span>'
      : '<span class="status-resolved-text">Resolved</span>';

    const actionHtml = alert.status === 'active'
      ? '<button class="btn-resolve" onclick="resolveAlert(' + alert.id + ')">Resolve</button>'
      : '<span class="status-resolved-text">—</span>';

    const message = alert.message || '';
    const truncMsg = message.length > 80 ? message.substring(0, 80) + '...' : message;

    html += '<tr class="' + rowClass + '">';
    html += '<td style="font-family:\'Space Mono\',monospace;font-size:12px;white-space:nowrap;">' + timeFormatted + '</td>';
    html += '<td style="font-weight:500;">' + (alert.type || '') + '</td>';
    html += '<td><span class="sev-badge ' + sevClass + '">' + (alert.severity || '').toUpperCase() + '</span></td>';
    html += '<td title="' + message.replace(/"/g, '&quot;') + '">' + truncMsg + '</td>';
    html += '<td>' + statusHtml + '</td>';
    html += '<td>' + actionHtml + '</td>';
    html += '</tr>';
  });

  tbody.innerHTML = html;
}

async function resolveAlert(id) {
  try {
    const res = await fetch('/api/alerts/' + id + '/resolve', {
      method: 'POST'
    });
    if (!res.ok) throw new Error('Resolve failed');
    await fetchAlerts();
  } catch (err) {
    console.error('Resolve alert error:', err);
  }
}
