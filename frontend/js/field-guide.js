/**
 * PHYTOSIGNAL — Field Guide Interactive Controller
 * Powers pathology library search, multi-factor crop/condition filtering,
 * and scientific monograph drawer inspections.
 */
(function () {
  'use strict';

  let currentCropFilter = 'all';
  let currentHealthFilter = 'all';
  let currentSearchQuery = '';
  let activeMonographRecord = null;

  const elements = {
    searchInput: document.getElementById('field-guide-search'),
    searchClearBtn: document.getElementById('search-clear-btn'),
    cropFilters: document.querySelectorAll('.crop-filter-btn'),
    healthFilters: document.querySelectorAll('.health-filter-btn'),
    cardsContainer: document.getElementById('field-guide-cards-grid'),
    resultCount: document.getElementById('guide-result-count'),
    noResults: document.getElementById('guide-no-results'),
    // Drawer
    drawer: document.getElementById('monograph-drawer'),
    drawerBackdrop: document.getElementById('drawer-backdrop'),
    drawerCloseBtn: document.getElementById('drawer-close-btn'),
    drawerContent: document.getElementById('drawer-body-content'),
    drawerCropBadge: document.getElementById('drawer-crop-badge'),
    drawerHealthBadge: document.getElementById('drawer-health-badge'),
    drawerConditionTitle: document.getElementById('drawer-condition-title'),
    drawerCausalAgent: document.getElementById('drawer-causal-agent'),
  };

  function initFieldGuide() {
    if (!elements.cardsContainer) return;

    // Attach event listeners
    if (elements.searchInput) {
      elements.searchInput.addEventListener('input', (e) => {
        currentSearchQuery = e.target.value.trim().toLowerCase();
        if (elements.searchClearBtn) {
          elements.searchClearBtn.style.display = currentSearchQuery ? 'inline-flex' : 'none';
        }
        renderCatalog();
      });
    }

    if (elements.searchClearBtn) {
      elements.searchClearBtn.addEventListener('click', () => {
        if (elements.searchInput) {
          elements.searchInput.value = '';
          currentSearchQuery = '';
          elements.searchClearBtn.style.display = 'none';
          renderCatalog();
          elements.searchInput.focus();
        }
      });
    }

    elements.cropFilters.forEach((btn) => {
      btn.addEventListener('click', () => {
        elements.cropFilters.forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        currentCropFilter = btn.getAttribute('data-crop') || 'all';
        renderCatalog();
      });
    });

    elements.healthFilters.forEach((btn) => {
      btn.addEventListener('click', () => {
        elements.healthFilters.forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        currentHealthFilter = btn.getAttribute('data-health') || 'all';
        renderCatalog();
      });
    });

    if (elements.drawerCloseBtn) {
      elements.drawerCloseBtn.addEventListener('click', closeMonographDrawer);
    }
    if (elements.drawerBackdrop) {
      elements.drawerBackdrop.addEventListener('click', closeMonographDrawer);
    }

    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && elements.drawer && elements.drawer.classList.contains('open')) {
        closeMonographDrawer();
      }
    });

    // Check URL parameters for direct deep-linking e.g. ?classId=2 or ?crop=tomato
    const urlParams = new URLSearchParams(window.location.search);
    const deepCrop = urlParams.get('crop');
    const deepClass = urlParams.get('classId');

    if (deepCrop) {
      const matchBtn = Array.from(elements.cropFilters).find(
        (b) => (b.getAttribute('data-crop') || '').toLowerCase() === deepCrop.toLowerCase()
      );
      if (matchBtn) {
        elements.cropFilters.forEach((b) => b.classList.remove('active'));
        matchBtn.classList.add('active');
        currentCropFilter = deepCrop.toLowerCase();
      }
    }

    renderCatalog();

    if (deepClass !== null) {
      const idNum = parseInt(deepClass, 10);
      if (!isNaN(idNum) && idNum >= 0 && idNum <= 14) {
        setTimeout(() => openMonographDrawer(idNum), 150);
      }
    }
  }

  function getKnowledgeRecords() {
    return Array.isArray(window.KNOWLEDGE_BASE_DATA) ? window.KNOWLEDGE_BASE_DATA : [];
  }

  function renderCatalog() {
    const records = getKnowledgeRecords();
    const filtered = records.filter((rec) => {
      // 1. Crop filter
      if (currentCropFilter !== 'all') {
        const cropLower = (rec.crop || '').toLowerCase();
        if (currentCropFilter === 'bell_pepper' || currentCropFilter === 'pepper') {
          if (!cropLower.includes('pepper')) return false;
        } else if (!cropLower.includes(currentCropFilter)) {
          return false;
        }
      }

      // 2. Health filter
      if (currentHealthFilter !== 'all') {
        const healthLower = (rec.health_status || '').toLowerCase();
        if (healthLower !== currentHealthFilter) return false;
      }

      // 3. Search query
      if (currentSearchQuery) {
        const hay = [
          rec.condition || '',
          rec.crop || '',
          rec.causal_agent || '',
          rec.symptoms?.summary || '',
        ]
          .join(' ')
          .toLowerCase();

        if (!hay.includes(currentSearchQuery)) return false;
      }

      return true;
    });

    if (elements.resultCount) {
      elements.resultCount.textContent = `${filtered.length} of ${records.length} conditions`;
    }

    if (elements.cardsContainer) {
      elements.cardsContainer.innerHTML = '';

      if (filtered.length === 0) {
        if (elements.noResults) elements.noResults.classList.remove('hidden');
        return;
      }

      if (elements.noResults) elements.noResults.classList.add('hidden');

      filtered.forEach((rec) => {
        const card = createPathologyCard(rec);
        elements.cardsContainer.appendChild(card);
      });
    }
  }

  function createPathologyCard(rec) {
    const isHealthy = (rec.health_status || '').toLowerCase() === 'healthy';
    const card = document.createElement('article');
    card.className = `pathology-card ${isHealthy ? 'card-healthy' : 'card-diseased'}`;
    card.setAttribute('tabindex', '0');
    card.setAttribute('role', 'button');
    card.setAttribute('aria-label', `View details for ${rec.crop} ${rec.condition}`);

    const header = document.createElement('div');
    header.className = 'pathology-card-header';

    const badges = document.createElement('div');
    badges.className = 'pathology-badges';

    const cropBadge = document.createElement('span');
    cropBadge.className = 'badge-crop';
    cropBadge.textContent = rec.crop;
    badges.appendChild(cropBadge);

    const healthBadge = document.createElement('span');
    healthBadge.className = `badge-status ${isHealthy ? 'status-healthy' : 'status-diseased'}`;
    healthBadge.textContent = isHealthy ? '✓ Healthy Foliage' : '⚠ Pathology Detected';
    badges.appendChild(healthBadge);

    header.appendChild(badges);

    const specimenId = document.createElement('span');
    specimenId.className = 'pathology-specimen-id';
    specimenId.textContent = `#SOL-${String(rec.class_id).padStart(3, '0')}`;
    header.appendChild(specimenId);

    card.appendChild(header);

    const title = document.createElement('h3');
    title.className = 'pathology-title';
    title.textContent = rec.condition;
    card.appendChild(title);

    if (rec.causal_agent) {
      const agent = document.createElement('p');
      agent.className = 'pathology-causal-agent';
      agent.innerHTML = `<em>${escapeHtml(rec.causal_agent)}</em>`;
      card.appendChild(agent);
    } else {
      const healthyNote = document.createElement('p');
      healthyNote.className = 'pathology-causal-agent healthy-agent';
      healthyNote.textContent = 'Physiologically Normal Tissue';
      card.appendChild(healthyNote);
    }

    const summary = document.createElement('p');
    summary.className = 'pathology-summary';
    summary.textContent = rec.symptoms?.summary || 'No diagnostic summary available.';
    card.appendChild(summary);

    const footer = document.createElement('div');
    footer.className = 'pathology-card-footer';

    const urgency = document.createElement('span');
    urgency.className = 'pathology-urgency';
    urgency.textContent = isHealthy ? 'Routine Maintenance' : 'Active Management Required';
    footer.appendChild(urgency);

    const inspectBtn = document.createElement('span');
    inspectBtn.className = 'pathology-inspect-action';
    inspectBtn.innerHTML = 'Inspect Monograph &rarr;';
    footer.appendChild(inspectBtn);

    card.appendChild(footer);

    // Click & Keyboard listeners to open drawer
    card.addEventListener('click', () => openMonographDrawer(rec.class_id));
    card.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        openMonographDrawer(rec.class_id);
      }
    });

    return card;
  }

  function openMonographDrawer(classId) {
    const records = getKnowledgeRecords();
    const rec = records.find((r) => r.class_id === classId);
    if (!rec) return;

    activeMonographRecord = rec;
    const isHealthy = (rec.health_status || '').toLowerCase() === 'healthy';

    if (elements.drawerCropBadge) elements.drawerCropBadge.textContent = rec.crop;
    if (elements.drawerHealthBadge) {
      elements.drawerHealthBadge.className = `badge-status ${isHealthy ? 'status-healthy' : 'status-diseased'}`;
      elements.drawerHealthBadge.textContent = isHealthy ? '✓ Healthy' : '⚠ Diseased';
    }
    if (elements.drawerConditionTitle) elements.drawerConditionTitle.textContent = rec.condition;
    if (elements.drawerCausalAgent) {
      elements.drawerCausalAgent.innerHTML = rec.causal_agent
        ? `<strong>Causal Agent:</strong> <em>${escapeHtml(rec.causal_agent)}</em>`
        : '<strong>Health Status:</strong> Verified asymptomatic foliar tissue.';
    }

    if (elements.drawerContent) {
      elements.drawerContent.innerHTML = buildMonographHtml(rec);
    }

    if (elements.drawer) {
      elements.drawer.classList.add('open');
      document.body.style.overflow = 'hidden';
      elements.drawer.setAttribute('aria-hidden', 'false');
    }
    if (elements.drawerBackdrop) {
      elements.drawerBackdrop.classList.add('visible');
    }
  }

  function closeMonographDrawer() {
    if (elements.drawer) {
      elements.drawer.classList.remove('open');
      elements.drawer.setAttribute('aria-hidden', 'true');
      document.body.style.overflow = '';
    }
    if (elements.drawerBackdrop) {
      elements.drawerBackdrop.classList.remove('visible');
    }
    activeMonographRecord = null;
  }

  function buildMonographHtml(rec) {
    const isHealthy = (rec.health_status || '').toLowerCase() === 'healthy';
    let html = '';

    // Symptoms section
    if (rec.symptoms) {
      html += `<div class="monograph-section">
        <h4 class="monograph-heading">1. Clinical Presentation & Symptoms</h4>
        <p class="monograph-lead">${escapeHtml(rec.symptoms.summary || '')}</p>`;

      if (rec.symptoms.foliar_symptoms && rec.symptoms.foliar_symptoms.length > 0) {
        html += `<h5 class="monograph-subheading">Foliar Symptoms</h5>
          <ul class="monograph-list">
            ${rec.symptoms.foliar_symptoms.map((s) => `<li>${escapeHtml(s)}</li>`).join('')}
          </ul>`;
      }

      if (rec.symptoms.stem_and_fruit_symptoms && rec.symptoms.stem_and_fruit_symptoms.length > 0) {
        html += `<h5 class="monograph-subheading">Stem & Fruit Manifestations</h5>
          <ul class="monograph-list">
            ${rec.symptoms.stem_and_fruit_symptoms.map((s) => `<li>${escapeHtml(s)}</li>`).join('')}
          </ul>`;
      }

      if (rec.symptoms.look_alike_conditions && rec.symptoms.look_alike_conditions.length > 0) {
        html += `<h5 class="monograph-subheading">Differential Diagnosis (Look-Alikes)</h5>
          <ul class="monograph-list look-alikes">
            ${rec.symptoms.look_alike_conditions.map((s) => `<li>${escapeHtml(s)}</li>`).join('')}
          </ul>`;
      }

      html += `</div>`;
    }

    // Favorable conditions
    if (rec.favorable_conditions && !isHealthy) {
      html += `<div class="monograph-section">
        <h4 class="monograph-heading">2. Environmental Drivers & Transmission</h4>
        <p><strong>Favorable Climate:</strong> ${escapeHtml(rec.favorable_conditions.environmental_factors || 'N/A')}</p>`;

      if (rec.favorable_conditions.transmission_routes && rec.favorable_conditions.transmission_routes.length > 0) {
        html += `<h5 class="monograph-subheading">Vectors & Dispersal Routes</h5>
          <ul class="monograph-list">
            ${rec.favorable_conditions.transmission_routes.map((r) => `<li>${escapeHtml(r)}</li>`).join('')}
          </ul>`;
      }
      html += `</div>`;
    }

    // Prevention & Cultural
    if (rec.prevention_and_cultural) {
      html += `<div class="monograph-section">
        <h4 class="monograph-heading">${isHealthy ? '2. Optimal Foliar Maintenance' : '3. Prevention & Cultural Sanitation'}</h4>`;

      if (rec.prevention_and_cultural.sanitation && rec.prevention_and_cultural.sanitation.length > 0) {
        html += `<h5 class="monograph-subheading">Sanitation Protocols</h5>
          <ul class="monograph-list">
            ${rec.prevention_and_cultural.sanitation.map((s) => `<li>${escapeHtml(s)}</li>`).join('')}
          </ul>`;
      }

      if (rec.prevention_and_cultural.crop_rotation) {
        html += `<p><strong>Crop Rotation:</strong> ${escapeHtml(rec.prevention_and_cultural.crop_rotation)}</p>`;
      }

      if (rec.prevention_and_cultural.irrigation_and_airflow && rec.prevention_and_cultural.irrigation_and_airflow.length > 0) {
        html += `<h5 class="monograph-subheading">Irrigation & Airflow</h5>
          <ul class="monograph-list">
            ${rec.prevention_and_cultural.irrigation_and_airflow.map((i) => `<li>${escapeHtml(i)}</li>`).join('')}
          </ul>`;
      }
      html += `</div>`;
    }

    // Treatment & IPM principles
    if (rec.treatment) {
      html += `<div class="monograph-section">
        <h4 class="monograph-heading">${isHealthy ? '3. Health Preservation Advisory' : '4. Integrated Pest Management (IPM) Protocols'}</h4>
        <div class="urgency-callout"><span class="badge-urgency">Guidance Level</span> ${escapeHtml(rec.treatment.urgency || '')}</div>`;

      if (rec.treatment.non_chemical_controls && rec.treatment.non_chemical_controls.length > 0) {
        html += `<h5 class="monograph-subheading">Biological & Non-Chemical Controls</h5>
          <ul class="monograph-list">
            ${rec.treatment.non_chemical_controls.map((c) => `<li>${escapeHtml(c)}</li>`).join('')}
          </ul>`;
      }

      if (rec.treatment.ipm_chemical_principles && rec.treatment.ipm_chemical_principles.length > 0) {
        html += `<h5 class="monograph-subheading">Chemical Application Principles</h5>
          <ul class="monograph-list">
            ${rec.treatment.ipm_chemical_principles.map((p) => `<li>${escapeHtml(p)}</li>`).join('')}
          </ul>`;
      }

      html += `</div>`;
    }

    // References
    if (rec.sources && rec.sources.length > 0) {
      html += `<div class="monograph-section monograph-sources">
        <h4 class="monograph-heading">Authoritative Sources & Extension Services</h4>
        <ul class="sources-list">
          ${rec.sources.map((s) => `<li>${escapeHtml(s)}</li>`).join('')}
        </ul>
      </div>`;
    }

    return html;
  }

  function escapeHtml(str) {
    if (typeof str !== 'string') return '';
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initFieldGuide);
  } else {
    initFieldGuide();
  }
})();
