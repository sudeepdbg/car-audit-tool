/* ==========================================================================
   Car Audit Tool - frontend logic
   --------------------------------------------------------------------------
   Improvements vs original:
   - Tab switching via `data-tab` (no more relying on event.target which can
     be a child element).
   - Single image source: file input + drag-and-drop + camera input are all
     funneled through one preview/render pipeline per drop zone.
   - Fixed camera input handling (the original `||` chain always picked the
     file input, never the camera).
   - Batch processing now uploads actual files via multipart/form-data
     (the original sent URL.createObjectURL blob URLs that don't exist on
     the server - the whole batch flow was broken).
   - Toast notifications instead of alert().
   - Modal viewer for audit details instead of alert(JSON).
   - Health-check polls /api/health on load to show offline-ready badge.
   - Side-by-side / overlay / difference heatmap toggle.
   - PDF export button for the latest audit.
   - Dark mode toggle persisted in localStorage.
   ========================================================================== */

(function () {
  'use strict';

  // ---------- Tiny helpers ----------
  const $  = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const TOKEN_KEY = 'car_audit_access_token';
  const REFRESH_KEY = 'car_audit_refresh_token';
  const USER_KEY = 'car_audit_user';

  const state = {
    currentTab: 'single',
    golden: null,
    returned: null,
    latestResult: null,
    batchPairs: [], // [{golden, returned, vehicleId}]
    lastHeatFiles: null,
    lastAuditId: null,
    user: null,
    refreshing: null, // promise guard
  };

  // ---------- Token storage ----------
  function getAccessToken() {
    try { return localStorage.getItem(TOKEN_KEY) || sessionStorage.getItem(TOKEN_KEY); } catch (_) { return null; }
  }
  function getRefreshToken() {
    try { return localStorage.getItem(REFRESH_KEY) || sessionStorage.getItem(REFRESH_KEY); } catch (_) { return null; }
  }
  function saveTokens(access, refresh, user, persistent) {
    const store = persistent ? localStorage : sessionStorage;
    const other = persistent ? sessionStorage : localStorage;
    try {
      other.removeItem(TOKEN_KEY); other.removeItem(REFRESH_KEY); other.removeItem(USER_KEY);
      store.setItem(TOKEN_KEY, access);
      store.setItem(REFRESH_KEY, refresh);
      if (user) store.setItem(USER_KEY, JSON.stringify(user));
    } catch (_) {}
  }
  function clearTokens() {
    try {
      localStorage.removeItem(TOKEN_KEY); localStorage.removeItem(REFRESH_KEY); localStorage.removeItem(USER_KEY);
      sessionStorage.removeItem(TOKEN_KEY); sessionStorage.removeItem(REFRESH_KEY); sessionStorage.removeItem(USER_KEY);
    } catch (_) {}
  }
  function loadStoredUser() {
    try {
      const raw = localStorage.getItem(USER_KEY) || sessionStorage.getItem(USER_KEY);
      return raw ? JSON.parse(raw) : null;
    } catch (_) { return null; }
  }

  // ---------- Toasts ----------
  function toast(message, type = 'info', timeout = 3500) {
    const host = $('#toastHost');
    const el = document.createElement('div');
    el.className = 'toast';
    el.dataset.type = type;
    el.textContent = message;
    host.appendChild(el);
    setTimeout(() => {
      el.style.opacity = '0';
      el.style.transform = 'translateX(20px)';
      setTimeout(() => el.remove(), 260);
    }, timeout);
  }

  // ---------- Theme ----------
  const themeToggle = $('#themeToggle');
  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    try { localStorage.setItem('car-audit-theme', theme); } catch (_) {}
  }
  (function initTheme() {
    let saved = null;
    try { saved = localStorage.getItem('car-audit-theme'); } catch (_) {}
    if (saved) {
      applyTheme(saved);
    } else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
      applyTheme('dark');
    }
  })();
  themeToggle.addEventListener('click', () => {
    const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    applyTheme(next);
  });

  // ---------- Auth-aware fetch ----------
  async function refreshAccessToken() {
    // Single-flight: only one refresh in flight at a time
    if (state.refreshing) return state.refreshing;
    const refreshToken = getRefreshToken();
    if (!refreshToken) return null;
    state.refreshing = (async () => {
      try {
        const r = await fetch('/api/auth/refresh', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ refresh_token: refreshToken }),
        });
        if (!r.ok) return null;
        const data = await r.json();
        const user = loadStoredUser();
        saveTokens(data.access_token, data.refresh_token, user, !!localStorage.getItem(REFRESH_KEY));
        return data.access_token;
      } catch (e) {
        return null;
      } finally {
        state.refreshing = null;
      }
    })();
    return state.refreshing;
  }

  function showLoginScreen(reason) {
    clearTokens();
    state.user = null;
    $('#appShell').hidden = true;
    $('#loginScreen').hidden = false;
    if (reason) {
      $('#loginError').hidden = false;
      $('#loginError').textContent = reason;
    } else {
      $('#loginError').hidden = true;
    }
    $('#loginUsername').focus();
  }
  function showAppShell() {
    $('#loginScreen').hidden = true;
    $('#appShell').hidden = false;
    refreshUserMenu();
    checkHealth();
  }
  function refreshUserMenu() {
    const u = state.user || loadStoredUser();
    if (!u) {
      $('#userMenu').hidden = true;
      return;
    }
    $('#userMenu').hidden = false;
    $('#userName').textContent = u.username + (u.role === 'admin' ? ' (admin)' : '');
    $('#userAvatar').textContent = (u.username || '?').slice(0, 1).toUpperCase();
  }

  async function apiFetch(url, opts = {}) {
    const token = getAccessToken();
    const headers = Object.assign({}, opts.headers || {});
    if (token) headers['Authorization'] = 'Bearer ' + token;
    const init = Object.assign({}, opts, { headers });
    let res = await fetch(url, init);

    // If the access token expired, try refreshing once and replay the request.
    if (res.status === 401) {
      const body = await res.clone().json().catch(() => ({}));
      if (body.code === 'bad_token' || body.code === 'no_token') {
        const newToken = await refreshAccessToken();
        if (newToken) {
          headers['Authorization'] = 'Bearer ' + newToken;
          res = await fetch(url, Object.assign({}, opts, { headers }));
          if (res.status === 401) {
            showLoginScreen('Session expired. Please sign in again.');
          }
        } else {
          showLoginScreen('Session expired. Please sign in again.');
        }
      }
    }
    return res;
  }
  window.__apiFetch = apiFetch; // for debugging

  // Parse a response body as JSON without throwing. Proxies and size limits can
  // answer with an HTML page; surface a readable message instead of
  // "Unexpected token < in JSON".
  async function safeJson(res) {
    try {
      return await res.json();
    } catch (_) {
      return { error: 'Server returned an unexpected response (HTTP ' + res.status + ').' };
    }
  }

  // Images and heatmaps are protected (they need the bearer token), so they
  // cannot be loaded with a plain <img src>. Fetch them with auth and show a blob URL.
  async function setAuthedImage(imgEl, relPath) {
    if (!imgEl || !relPath) return;
    try {
      const res = await apiFetch('/' + String(relPath).replace(/^\/+/, ''));
      if (!res.ok) throw new Error('HTTP ' + res.status);
      const blob = await res.blob();
      if (imgEl.dataset.blobUrl) URL.revokeObjectURL(imgEl.dataset.blobUrl);
      const url = URL.createObjectURL(blob);
      imgEl.dataset.blobUrl = url;
      imgEl.src = url;
    } catch (err) {
      imgEl.alt = 'Image unavailable';
    }
  }

  async function downloadPdf(auditId) {
    try {
      const res = await apiFetch('/api/audit/' + encodeURIComponent(auditId) + '/export/pdf');
      if (!res.ok) throw new Error('HTTP ' + res.status);
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'audit_' + auditId + '.pdf';
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 10000);
    } catch (err) {
      toast('PDF export failed: ' + err.message, 'error');
    }
  }

  // ---------- Tabs ----------
  $$('.tab').forEach((btn) => {
    btn.addEventListener('click', () => {
      const tab = btn.dataset.tab;
      $$('.tab').forEach((b) => {
        b.classList.toggle('active', b === btn);
        b.setAttribute('aria-selected', b === btn ? 'true' : 'false');
      });
      $$('.panel').forEach((p) => {
        const active = p.id === tab + 'Tab';
        p.classList.toggle('active', active);
        p.hidden = !active;
      });
      state.currentTab = tab;
      if (tab === 'history') loadAudits();
      if (tab === 'stats')   loadStatistics();
    });
  });

  // ---------- Drop zones ----------
  function wireDropZone(zoneEl, side) {
    const target = $('.drop-target', zoneEl);
    const previewImg = $('.drop-preview', zoneEl);
    const fileInput = $('.drop-input', zoneEl);
    const cameraTrigger = $('.camera-trigger', zoneEl);
    const cameraInput = $('.camera-input', zoneEl);

    target.addEventListener('click', () => fileInput.click());
    target.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        fileInput.click();
      }
    });

    ['dragenter', 'dragover'].forEach((evt) =>
      target.addEventListener(evt, (e) => {
        e.preventDefault();
        target.classList.add('dragover');
      })
    );
    ['dragleave', 'drop'].forEach((evt) =>
      target.addEventListener(evt, (e) => {
        e.preventDefault();
        target.classList.remove('dragover');
      })
    );
    target.addEventListener('drop', (e) => {
      const file = e.dataTransfer.files && e.dataTransfer.files[0];
      if (file) acceptFile(file);
    });

    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) acceptFile(e.target.files[0]);
    });

    cameraTrigger.addEventListener('click', (e) => {
      e.stopPropagation();
      cameraInput.click();
    });
    cameraInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) acceptFile(e.target.files[0]);
    });

    function acceptFile(file) {
      // Accept images OR videos. The server extracts one representative frame
      // from a video and runs the same comparison pipeline.
      if (!/^image\//.test(file.type) && !/^video\//.test(file.type)) {
        toast('Please choose an image (PNG/JPG/WEBP) or video (MP4/MOV).', 'error');
        return;
      }
      state[side] = file;
      // Preview images directly. For videos, capture the first frame as a
      // poster so the auditor still sees what they uploaded.
      if (/^image\//.test(file.type)) {
        const reader = new FileReader();
        reader.onload = (ev) => {
          previewImg.src = ev.target.result;
          target.classList.add('has-image');
        };
        reader.readAsDataURL(file);
      } else {
        const url = URL.createObjectURL(file);
        const v = document.createElement('video');
        v.preload = 'metadata';
        v.muted = true;
        v.src = url;
        const drawPoster = () => {
          try {
            const canvas = document.createElement('canvas');
            canvas.width = v.videoWidth || 160;
            canvas.height = v.videoHeight || 120;
            canvas.getContext('2d').drawImage(v, 0, 0, canvas.width, canvas.height);
            previewImg.src = canvas.toDataURL('image/png');
            target.classList.add('has-image');
          } catch (e) { /* same-origin blob: is fine; ignore */ }
          URL.revokeObjectURL(url);
        };
        v.addEventListener('loadeddata', () => {
          // Some browsers have no painted frame at t=0; wait for a seek to finish.
          v.addEventListener('seeked', drawPoster, { once: true });
          try { v.currentTime = Math.min(0.1, (v.duration || 1) / 2); } catch (e) { drawPoster(); }
        }, { once: true });
        v.addEventListener('error', () => { URL.revokeObjectURL(url); }, { once: true });
      }
      refreshCompareBtn();
    }
  }

  wireDropZone($('#goldenZone'), 'golden');
  wireDropZone($('#returnedZone'), 'returned');

  function refreshCompareBtn() {
    $('#compareBtn').disabled = !(state.golden && state.returned);
  }

  // ---------- Single compare ----------
  $('#compareBtn').addEventListener('click', async () => {
    if (!state.golden || !state.returned) return;
    $('#loadingIndicator').hidden = false;
    $('#resultsSection').hidden = true;

    const fd = new FormData();
    fd.append('golden_image', state.golden);
    fd.append('returned_image', state.returned);
    const v = $('#vehicleId').value.trim();
    const c = $('#customerName').value.trim();
    const s = $('#rentalStartDate').value;
    const e = $('#rentalEndDate').value;
    if (v) fd.append('vehicle_id', v);
    if (c) fd.append('customer_name', c);
    if (s) fd.append('rental_start_date', s);
    if (e) fd.append('rental_end_date', e);

    try {
      const res = await apiFetch('/api/compare', { method: 'POST', body: fd });
      const data = await safeJson(res);
      $('#loadingIndicator').hidden = true;
      if (data.error) {
        toast(data.error, 'error');
        return;
      }
      state.latestResult = data;
      state.lastHeatFiles = data.heatmap_files || {};
      state.lastAuditId = data.audit_id;
      renderResults(data);
      $('#resultsSection').hidden = false;
      toast('Audit complete: ' + data.condition_assessment, 'success');
    } catch (err) {
      $('#loadingIndicator').hidden = true;
      toast('Request failed: ' + err.message, 'error');
    }
  });

  function renderResults(r) {
    $('#trustScoreValue').textContent = (r.trust_score || 0).toFixed(2);
    $('#conditionPill').textContent = r.condition_assessment;
    $('#conditionPill').dataset.condition = r.condition_assessment;
    $('#scoreCircle').dataset.condition = r.condition_assessment;
    $('#recommendationText').textContent = r.recommendation || '';
    $('#severityText').textContent = r.severity_level || '-';
    $('#auditIdText').textContent = r.audit_id || '-';
    $('#processingTimeText').textContent =
      (r.metadata && r.metadata.processing_time_sec != null)
        ? 'processed in ' + r.metadata.processing_time_sec + 's'
        : '';

    // Warnings (aspect ratio, missing weights, no alignment ...) and the inputs used
    const warnEl = $('#warningsList');
    warnEl.innerHTML = '';
    (r.warnings || []).forEach((w) => {
      const li = document.createElement('li');
      li.textContent = w;
      warnEl.appendChild(li);
    });
    warnEl.hidden = !(r.warnings && r.warnings.length);
    const inputsEl = $('#inputsText');
    const describe = (label, i) => {
      if (!i) return null;
      if (i.type !== 'video') return label + ': image';
      const t = (i.time_sec != null) ? ' @' + i.time_sec + 's' : '';
      return label + ': video frame' + t;
    };
    const parts = [describe('Golden', r.inputs && r.inputs.golden), describe('Returned', r.inputs && r.inputs.returned)].filter(Boolean);
    inputsEl.textContent = parts.join(' \u00b7 ');
    inputsEl.hidden = !parts.length;

    // Heatmap default view = overlay
    $$('.heatmap-toggle .seg').forEach((b) => b.classList.toggle('active', b.dataset.heatmapView === 'overlay'));
    setHeatmapView('overlay');

    // Semantic details
    const sa = r.semantic_analysis || {};
    $('#semanticDetails').innerHTML = metricRows([
      ['Cosine similarity', sa.cosine_similarity == null ? 'not used (no pretrained weights)' : sa.cosine_similarity],
      ['Feature vector length', sa.feature_vector_length || null],
    ]);

    // Localised change analysis
    const ca = r.change_analysis || {};
    const pixelExtra = [
      ['Changed area (%)', ca.changed_area_pct],
      ['Changed regions', ca.region_count],
      ['Largest region (%)', ca.largest_region_pct],
      ['Area compared (%)', ca.compared_area_pct],
    ];

    // Pixel metrics
    const pm = r.pixel_level_metrics || {};
    $('#pixelDetails').innerHTML = metricRows([
      ['SSIM', pm.ssim],
      ['MSE', pm.mse],
      ['PSNR (dB)', pm.psnr],
      ['MAD', pm.mad],
      ...pixelExtra,
    ]);

    // Metadata
    const md = r.metadata || {};
    $('#metadataDetails').innerHTML = metricRows([
      ['Model', md.model_used],
      ['Weight source', md.weight_source],
      ['Device', md.device],
      ['Method', md.comparison_method],
      ['Processing time (s)', md.processing_time_sec],
    ]);

    $('#jsonOutput').textContent = JSON.stringify(r, null, 2);
  }

  function metricRows(pairs) {
    return pairs
      .filter(([, v]) => v !== undefined && v !== null)
      .map(([label, value]) =>
        `<div class="metric-row"><span class="metric-label">${label}</span><span class="metric-value">${value}</span></div>`
      )
      .join('');
  }

  // Heatmap view toggle
  $$('.heatmap-toggle .seg').forEach((btn) =>
    btn.addEventListener('click', () => {
      $$('.heatmap-toggle .seg').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      setHeatmapView(btn.dataset.heatmapView);
    })
  );
  function setHeatmapView(view) {
    if (!state.lastHeatFiles) return;
    const map = {
      overlay: state.lastHeatFiles.overlay,
      composite: state.lastHeatFiles.composite,
      difference: state.lastHeatFiles.difference,
    };
    const rel = map[view];
    if (!rel) return;
    setAuthedImage($('#heatmapImage'), rel);
  }

  // New compare
  $('#newCompareBtn').addEventListener('click', () => {
    state.golden = null;
    state.returned = null;
    state.latestResult = null;
    state.lastHeatFiles = null;
    $$('.drop-target').forEach((t) => {
      t.classList.remove('has-image');
      $('.drop-preview', t).removeAttribute('src');
    });
    $$('.drop-input, .camera-input').forEach((i) => (i.value = ''));
    $('#resultsSection').hidden = true;
    refreshCompareBtn();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });

  // Copy JSON
  $('#copyJsonBtn').addEventListener('click', () => {
    const txt = $('#jsonOutput').textContent;
    navigator.clipboard.writeText(txt).then(
      () => toast('JSON copied to clipboard.', 'success'),
      () => toast('Could not copy - check browser permissions.', 'error')
    );
  });

  // PDF export (latest audit)
  $('#exportPdfBtn').addEventListener('click', () => {
    if (!state.lastAuditId) {
      toast('No audit to export yet.', 'warn');
      return;
    }
    downloadPdf(state.lastAuditId);
  });

  // ---------- Batch ----------
  function renderBatchPairs() {
    const container = $('#batchPairsContainer');
    container.innerHTML = '';
    state.batchPairs.forEach((pair, index) => {
      const div = document.createElement('div');
      div.className = 'batch-pair';
      div.innerHTML = `
        <div>
          <div class="pair-label">Golden image/video</div>
          <input type="file" class="batch-golden" accept="image/*,video/*" />
        </div>
        <div>
          <div class="pair-label">Returned image/video</div>
          <input type="file" class="batch-returned" accept="image/*,video/*" />
        </div>
        <div>
          <div class="pair-label">Vehicle ID</div>
          <input type="text" class="batch-vehicle-id" placeholder="MH-12-AB-1234" />
        </div>
        <button class="remove-pair-btn" type="button" title="Remove pair">&times;</button>
      `;
      $('.batch-golden', div).addEventListener('change', (e) => {
        pair.golden = e.target.files[0];
        refreshBatchBtn();
      });
      $('.batch-returned', div).addEventListener('change', (e) => {
        pair.returned = e.target.files[0];
        refreshBatchBtn();
      });
      $('.batch-vehicle-id', div).addEventListener('input', (e) => {
        pair.vehicleId = e.target.value;
      });
      $('.remove-pair-btn', div).addEventListener('click', () => {
        state.batchPairs.splice(index, 1);
        renderBatchPairs();
        refreshBatchBtn();
      });
      container.appendChild(div);
    });
  }
  function refreshBatchBtn() {
    const ready = state.batchPairs.some((p) => p.golden && p.returned);
    $('#batchRunBtn').disabled = !ready;
  }
  $('#addPairBtn').addEventListener('click', () => {
    state.batchPairs.push({ golden: null, returned: null, vehicleId: '' });
    renderBatchPairs();
  });
  // Seed one empty pair
  state.batchPairs.push({ golden: null, returned: null, vehicleId: '' });
  renderBatchPairs();
  refreshBatchBtn();

  $('#batchRunBtn').addEventListener('click', async () => {
    const readyPairs = state.batchPairs.filter((p) => p.golden && p.returned);
    if (!readyPairs.length) {
      toast('Add at least one complete image pair.', 'warn');
      return;
    }
    $('#batchLoading').hidden = false;
    $('#batchResults').hidden = true;
    $('#batchProgress').textContent = '(0 / ' + readyPairs.length + ')';

    const fd = new FormData();
    readyPairs.forEach((pair, i) => {
      fd.append('golden_image_' + i, pair.golden);
      fd.append('returned_image_' + i, pair.returned);
      if (pair.vehicleId) fd.append('vehicle_id_' + i, pair.vehicleId);
    });

    try {
      const res = await apiFetch('/api/batch-compare', { method: 'POST', body: fd });
      const data = await safeJson(res);
      $('#batchLoading').hidden = true;
      if (!res.ok || !Array.isArray(data.results)) {
        toast(data.error || 'Batch request failed.', 'error');
        return;
      }
      $('#batchResults').hidden = false;
      renderBatchResults(data);
      toast('Batch done: ' + data.successful + ' / ' + data.total_pairs + ' succeeded.', data.failed ? 'warn' : 'success');
    } catch (err) {
      $('#batchLoading').hidden = true;
      toast('Batch request failed: ' + err.message, 'error');
    }
  });

  function renderBatchResults(summary) {
    $('#batchSummary').innerHTML = `
      <div class="summary-card" data-state="neutral">
        <div class="label">Total pairs</div>
        <div class="value">${summary.total_pairs}</div>
      </div>
      <div class="summary-card" data-state="good">
        <div class="label">Successful</div>
        <div class="value">${summary.successful}</div>
      </div>
      <div class="summary-card" data-state="bad">
        <div class="label">Failed</div>
        <div class="value">${summary.failed}</div>
      </div>
      <div class="summary-card" data-state="neutral">
        <div class="label">Avg trust score</div>
        <div class="value">${(summary.avg_trust_score || 0).toFixed(4)}</div>
      </div>
    `;
    const details = $('#batchDetails');
    details.innerHTML = '';
    summary.results.forEach((r) => {
      const item = document.createElement('div');
      item.className = 'audit-item';
      if (r.error) {
        item.innerHTML = `
          <span class="ai-id">Pair #${r.pair_index}</span>
          <span class="ai-vehicle">${escapeHtml(r.vehicle_id || 'N/A')}</span>
          <span class="pill" data-condition="CRITICAL">FAILED</span>
          <span class="ai-date">${escapeHtml(r.error)}</span>
        `;
      } else {
        item.innerHTML = `
          <span class="ai-id">Pair #${r.pair_index}</span>
          <span class="ai-vehicle">${escapeHtml(r.vehicle_id || 'N/A')}</span>
          <span class="pill" data-condition="${escapeHtml(r.condition_assessment)}">${escapeHtml(r.condition_assessment)}</span>
          <span class="ai-score">${(r.trust_score || 0).toFixed(4)}</span>
        `;
        item.addEventListener('click', () => openAuditModal(r.audit_id));
      }
      details.appendChild(item);
    });
  }

  // ---------- History ----------
  $('#refreshAuditsBtn').addEventListener('click', loadAudits);
  $('#searchAudit').addEventListener('input', filterAudits);

  async function loadAudits() {
    try {
      const res = await apiFetch('/api/audits?limit=100');
      const data = await res.json();
      window.__auditsCache = data.audits || [];
      renderAudits(window.__auditsCache);
    } catch (err) {
      toast('Failed to load audits: ' + err.message, 'error');
    }
  }

  function renderAudits(audits) {
    const list = $('#auditsList');
    if (!audits.length) {
      list.innerHTML = `<div class="empty-state"><p>No audits found yet.</p></div>`;
      return;
    }
    list.innerHTML = '';
    audits.forEach((a) => {
      const item = document.createElement('div');
      item.className = 'audit-item';
      const score = (a.trust_score || 0).toFixed(4);
      const date = formatDate(a.created_at);
      item.innerHTML = `
        <span class="ai-id">${escapeHtml(a.audit_id)}</span>
        <span class="ai-vehicle">${escapeHtml(a.vehicle_id || 'N/A')}</span>
        <span class="pill" data-condition="${escapeHtml(a.condition_assessment)}">${escapeHtml(a.condition_assessment)}</span>
        <span class="ai-score">${score} <span class="ai-date"> &middot; ${date}</span></span>
      `;
      item.addEventListener('click', () => openAuditModal(a.audit_id));
      list.appendChild(item);
    });
  }

  function filterAudits() {
    const term = ($('#searchAudit').value || '').toLowerCase().trim();
    const cached = window.__auditsCache || [];
    if (!term) return renderAudits(cached);
    renderAudits(cached.filter((a) =>
      (a.audit_id || '').toLowerCase().includes(term) ||
      (a.vehicle_id || '').toLowerCase().includes(term) ||
      (a.customer_name || '').toLowerCase().includes(term)
    ));
  }

  // ---------- Modal ----------
  const modal = $('#auditModal');
  function closeAuditModal() { modal.hidden = true; };
  $$('[data-close]', modal).forEach((el) => el.addEventListener('click', closeAuditModal));
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      if (!modal.hidden) closeAuditModal();
      if (!$('#passModal').hidden) closeModal($('#passModal'));
    }
  });

  async function openAuditModal(auditId) {
    try {
      const res = await apiFetch('/api/audit/' + encodeURIComponent(auditId));
      const data = await safeJson(res);
      if (data.error) { toast(data.error, 'error'); return; }
      state.lastAuditId = auditId;
      state.lastHeatFiles = data.heatmap_files || {};
      $('#modalTitle').textContent = 'Audit ' + data.audit_id;
      const heat = data.heatmap_files || {};
      const heatImg = heat.overlay ? `<img class="modal-heatmap" id="modalHeatmapImg" alt="Heatmap" />` : '';
      $('#modalBody').innerHTML = `
        <div class="metric-list">
          <div class="metric-row"><span class="metric-label">Vehicle</span><span class="metric-value">${escapeHtml(data.vehicle_id || 'N/A')}</span></div>
          <div class="metric-row"><span class="metric-label">Customer</span><span class="metric-value">${escapeHtml(data.customer_name || 'N/A')}</span></div>
          <div class="metric-row"><span class="metric-label">Rental period</span><span class="metric-value">${escapeHtml(data.rental_start_date || '?')} &rarr; ${escapeHtml(data.rental_end_date || '?')}</span></div>
          <div class="metric-row"><span class="metric-label">Trust score</span><span class="metric-value">${(data.trust_score || 0).toFixed(4)}</span></div>
          <div class="metric-row"><span class="metric-label">Condition</span><span class="metric-value">${escapeHtml(data.condition_assessment)}</span></div>
          <div class="metric-row"><span class="metric-label">Severity</span><span class="metric-value">${data.severity_level ?? 'n/a'}</span></div>
          <div class="metric-row"><span class="metric-label">SSIM</span><span class="metric-value">${data.ssim ?? 'n/a'}</span></div>
          <div class="metric-row"><span class="metric-label">MSE</span><span class="metric-value">${data.mse ?? 'n/a'}</span></div>
          <div class="metric-row"><span class="metric-label">PSNR (dB)</span><span class="metric-value">${data.psnr ?? 'n/a'}</span></div>
          <div class="metric-row"><span class="metric-label">MAD</span><span class="metric-value">${data.mad ?? 'n/a'}</span></div>
          <div class="metric-row"><span class="metric-label">Changed area (%)</span><span class="metric-value">${data.changed_area_pct ?? 'n/a'}</span></div>
          <div class="metric-row"><span class="metric-label">Changed regions</span><span class="metric-value">${data.region_count ?? 'n/a'}</span></div>
          <div class="metric-row"><span class="metric-label">Semantic sim</span><span class="metric-value">${data.semantic_similarity ?? 'n/a (not used)'}</span></div>
          <div class="metric-row"><span class="metric-label">Created</span><span class="metric-value">${formatDate(data.created_at)}</span></div>
        </div>
        <p><strong>Recommendation:</strong> ${escapeHtml(data.recommendation || '')}</p>
        ${heatImg}
        <div>
          <label class="field-label">Auditor notes</label>
          <textarea id="modalNotes" placeholder="Add notes...">${escapeHtml(data.notes || '')}</textarea>
        </div>
      `;
      if (heat.overlay) setAuthedImage($('#modalHeatmapImg'), heat.overlay);
      $('#modalExportBtn').onclick = () => downloadPdf(auditId);
      const notesEl = $('#modalNotes');
      notesEl.addEventListener('blur', async () => {
        const notes = notesEl.value;
        try {
          const r = await apiFetch('/api/audit/' + encodeURIComponent(auditId) + '/notes', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ notes }),
          });
          if (r.ok) toast('Notes saved.', 'success');
        } catch (e) { /* ignore */ }
      });
      modal.hidden = false;
    } catch (err) {
      toast('Failed to load audit: ' + err.message, 'error');
    }
  }

  // ---------- Statistics ----------
  $('#loadStatsBtn').addEventListener('click', loadStatistics);
  async function loadStatistics() {
    try {
      const res = await apiFetch('/api/statistics');
      const stats = await res.json();
      renderStats(stats);
    } catch (err) {
      toast('Failed to load statistics: ' + err.message, 'error');
    }
  }

  function renderStats(s) {
    const host = $('#statisticsContent');
    const dist = s.condition_distribution || {};
    const total = s.total_audits || 0;
    const maxCount = Math.max(1, ...Object.values(dist));

    const distBars = Object.entries(dist).map(([cond, count]) => {
      const pct = (count / maxCount) * 100;
      return `<div class="bar-row">
        <span class="bar-label">${escapeHtml(cond)}</span>
        <span class="bar-track"><span class="bar-fill" data-pct="${pct}"></span></span>
        <span class="bar-count">${count}</span>
      </div>`;
    }).join('');

    const trend = (s.trend || []).map((t) => {
      const h = Math.max(4, Math.round((t.trust_score || 0) * 60));
      return `<div class="bar" data-h="${h}" title="${(t.trust_score || 0).toFixed(3)}"></div>`;
    }).join('');

    const worst = (s.worst_vehicles || []).map((v) =>
      `<div class="metric-row"><span class="metric-label">${escapeHtml(v.vehicle_id || 'N/A')}</span><span class="metric-value">${v.audits} audits &middot; avg ${(v.avg_trust_score || 0).toFixed(4)}</span></div>`
    ).join('') || '<div class="metric-row"><span class="metric-label">No data</span><span class="metric-value">-</span></div>';

    host.innerHTML = `
      <div class="stat-card">
        <h4>Total audits</h4>
        <div class="big">${total}</div>
        <div class="sub">${s.recent_audits_7days} in the last 7 days</div>
      </div>
      <div class="stat-card">
        <h4>Avg trust score</h4>
        <div class="big">${(s.avg_trust_score || 0).toFixed(4)}</div>
        <div class="sub">across all audits</div>
      </div>
      <div class="stat-card">
        <h4>Score trend (last 14)</h4>
        <div class="trend-spark">${trend}</div>
      </div>
      <div class="stat-card">
        <h4>Condition distribution</h4>
        ${distBars || '<div class="sub">No data yet</div>'}
      </div>
      <div class="stat-card wide">
        <h4>Worst performing vehicles</h4>
        <div class="metric-list">${worst}</div>
      </div>
    `;
    // Sizes are applied through the CSSOM: the CSP forbids inline style="" attributes.
    $$('.bar-fill', host).forEach((el) => { el.style.width = el.dataset.pct + '%'; });
    $$('.trend-spark .bar', host).forEach((el) => { el.style.height = el.dataset.h + 'px'; });
  }

  // ---------- Health check ----------
  async function checkHealth() {
    const badge = $('#modelBadge');
    const footerInfo = $('#footerModelInfo');
    try {
      const res = await apiFetch('/api/health');
      const data = await res.json();
      if (data.offline_ready) {
        badge.dataset.state = 'ok';
        badge.textContent = 'offline ready';
      } else {
        badge.dataset.state = 'offline';
        badge.textContent = 'weights not cached';
      }
      footerInfo.textContent = data.model + ' on ' + data.device;
    } catch (e) {
      badge.dataset.state = 'error';
      badge.textContent = 'server unreachable';
    }
  }

  // ---------- Helpers ----------
  function escapeHtml(s) {
    if (s === null || s === undefined) return '';
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }
  function formatDate(s) {
    if (!s) return '-';
    const d = new Date(s);
    if (isNaN(d.getTime())) return s;
    return d.toLocaleString();
  }

  // ---------- Login form ----------
  const loginForm = $('#loginForm');
  const loginError = $('#loginError');
  const togglePass = $('#togglePass');
  togglePass.addEventListener('click', () => {
    const inp = $('#loginPassword');
    inp.type = inp.type === 'password' ? 'text' : 'password';
  });

  loginForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    loginError.hidden = true;
    const username = $('#loginUsername').value.trim();
    const password = $('#loginPassword').value;
    if (!username || !password) {
      loginError.hidden = false;
      loginError.textContent = 'Username and password are required.';
      return;
    }
    const submit = $('#loginSubmit');
    submit.disabled = true;
    submit.querySelector('span').textContent = 'Signing in...';
    try {
      const res = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      });
      const data = await res.json();
      if (!res.ok) {
        loginError.hidden = false;
        loginError.textContent = data.error || 'Login failed';
        return;
      }
      state.user = data.user;
      saveTokens(data.access_token, data.refresh_token, data.user, true);
      $('#loginPassword').value = '';
      showAppShell();
      toast('Welcome, ' + data.user.username + '!', 'success');
    } catch (err) {
      loginError.hidden = false;
      loginError.textContent = 'Network error: ' + err.message;
    } finally {
      submit.disabled = false;
      submit.querySelector('span').textContent = 'Sign in';
    }
  });

  // ---------- User menu ----------
  const userBtn = $('#userBtn');
  const userDropdown = $('#userDropdown');
  userBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    const open = !userDropdown.hidden;
    userDropdown.hidden = open;
    userBtn.setAttribute('aria-expanded', String(!open));
  });
  document.addEventListener('click', (e) => {
    if (!userDropdown.hidden && !userMenuContains(e.target)) {
      userDropdown.hidden = true;
      userBtn.setAttribute('aria-expanded', 'false');
    }
  });
  function userMenuContains(target) {
    const menu = $('#userMenu');
    return menu && menu.contains(target);
  }
  $$('#userDropdown [data-menu]').forEach((btn) =>
    btn.addEventListener('click', () => {
      const action = btn.dataset.menu;
      userDropdown.hidden = true;
      if (action === 'logout') {
        // Tell the server to revoke this user's tokens (bumps token_version),
        // so the tokens are invalid even before their 8h/7d expiry. We fire and
        // forget - clearing local state below proceeds regardless.
        const token = getAccessToken();
        if (token) {
          fetch('/api/auth/logout', { method: 'POST', headers: { 'Authorization': 'Bearer ' + token } })
            .catch(() => {});
        }
        clearTokens();
        state.user = null;
        showLoginScreen();
        toast('Signed out.', 'info');
      } else if (action === 'change-pass') {
        openPassModal();
      }
    })
  );

  // ---------- Change password modal ----------
  const passModal = $('#passModal');
  function openModal(modalEl) { modalEl.hidden = false; }
  function closeModal(modalEl) { modalEl.hidden = true; }
  function openPassModal() {
    $('#passCurrent').value = '';
    $('#passNew').value = '';
    $('#passConfirm').value = '';
    $('#passError').hidden = true;
    openModal(passModal);
    $('#passCurrent').focus();
  }
  $$('[data-close]', passModal).forEach((el) => el.addEventListener('click', () => closeModal(passModal)));
  $('#passSubmit').addEventListener('click', async () => {
    const cur = $('#passCurrent').value;
    const neu = $('#passNew').value;
    const cnf = $('#passConfirm').value;
    const errEl = $('#passError');
    if (neu.length < 8) {
      errEl.hidden = false; errEl.textContent = 'New password must be at least 8 characters.'; return;
    }
    if (neu !== cnf) {
      errEl.hidden = false; errEl.textContent = 'New password and confirmation do not match.'; return;
    }
    errEl.hidden = true;
    try {
      const res = await apiFetch('/api/auth/change-password', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ current_password: cur, new_password: neu }),
      });
      const data = await safeJson(res);
      if (!res.ok) {
        errEl.hidden = false; errEl.textContent = data.error || 'Failed'; return;
      }
      // The server bumps token_version on a password change, which invalidates
      // the token we just used. It returns a fresh access+refresh pair so the
      // user stays signed in on this device.
      if (data.access_token && data.refresh_token) {
        const persistent = !!localStorage.getItem(REFRESH_KEY);
        saveTokens(data.access_token, data.refresh_token, state.user, persistent);
      }
      toast('Password changed.', 'success');
      closeModal(passModal);
    } catch (e) {
      errEl.hidden = false; errEl.textContent = 'Network error: ' + e.message;
    }
  });

  // ---------- Initial load ----------
  (async function init() {
    // Try to reuse stored tokens; if they fail, show login screen.
    const token = getAccessToken();
    if (token) {
      try {
        const res = await apiFetch('/api/auth/me');
        if (res.ok) {
          const data = await res.json();
          state.user = data.user;
          saveTokens(getAccessToken(), getRefreshToken(), data.user, !!localStorage.getItem(REFRESH_KEY));
          showAppShell();
          return;
        }
      } catch (_) { /* fall through */ }
    }
    showLoginScreen();
  })();
})();
