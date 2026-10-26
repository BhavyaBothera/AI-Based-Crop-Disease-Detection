/**
 * AI-Based Crop Disease Detection & Treatment Recommendation
 * Frontend UI Controller
 *
 * Manages DOM interactions, drag-and-drop file upload, image preview,
 * pre-flight validation, loading states, API dispatch, diagnostic rendering,
 * and agronomic disease knowledge integration.
 */

(function () {
  'use strict';

  // ==========================================================================
  // State Management
  // ==========================================================================
  const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024; // 10 MB (Matches backend configuration)
  const MAX_PIXEL_COUNT = 20 * 1024 * 1024; // 20 MP limit
  const ALLOWED_MIME_TYPES = new Set(['image/jpeg', 'image/png', 'image/webp']);

  let currentFile = null;
  let currentObjectUrl = null;
  let isAnalyzing = false;
  let isModelReady = false;

  // Prediction & Knowledge Base State
  let lastPrediction = null;
  let activeCandidateClassId = null;
  let activeCandidateMeta = null;
  let currentKnowledgeRequestId = 0;
  let isKnowledgeLoading = false;

  // ==========================================================================
  // DOM Elements
  // ==========================================================================
  const elements = {
    // Header & Readiness
    connectionStatus: document.getElementById('connection-status'),
    connectionStatusText: document.getElementById('connection-status-text'),

    // Alert Banner
    alertBanner: document.getElementById('alert-banner'),
    alertTitle: document.getElementById('alert-title'),
    alertMessage: document.getElementById('alert-message'),
    alertDismissBtn: document.getElementById('alert-dismiss-btn'),

    // Upload & Dropzone
    dropZone: document.getElementById('drop-zone'),
    fileInput: document.getElementById('file-input'),
    browseBtn: document.getElementById('browse-btn'),

    // Preview
    previewContainer: document.getElementById('preview-container'),
    imagePreview: document.getElementById('image-preview'),
    metaFilename: document.getElementById('meta-filename'),
    metaDimensions: document.getElementById('meta-dimensions'),
    metaFilesize: document.getElementById('meta-filesize'),
    analyzeBtn: document.getElementById('analyze-btn'),
    removeBtn: document.getElementById('remove-btn'),

    // Loading State
    loadingContainer: document.getElementById('loading-container'),

    // Results Section
    resultsCard: document.getElementById('results-card'),
    resultsEmpty: document.getElementById('results-empty'),
    resultsContent: document.getElementById('results-content'),
    newScanBtn: document.getElementById('new-scan-btn'),

    // Primary Diagnosis
    resultCrop: document.getElementById('result-crop'),
    resultCondition: document.getElementById('result-condition'),
    resultHealthBadge: document.getElementById('result-health-badge'),
    resultScore: document.getElementById('result-score'),
    candidatesList: document.getElementById('candidates-list'),
    resultDisclaimer: document.getElementById('result-disclaimer'),

    // Knowledge Section Elements
    knowledgeSection: document.getElementById('knowledge-section'),
    knowledgeContextBadge: document.getElementById('knowledge-context-badge'),
    candidateExplorationNotice: document.getElementById('candidate-exploration-notice'),
    candidateExplorationDesc: document.getElementById('candidate-exploration-desc'),
    knowledgeLoading: document.getElementById('knowledge-loading'),
    knowledgeError: document.getElementById('knowledge-error'),
    knowledgeErrorMessage: document.getElementById('knowledge-error-message'),
    knowledgeRetryBtn: document.getElementById('knowledge-retry-btn'),
    knowledgeDetails: document.getElementById('knowledge-details'),
  };

  // ==========================================================================
  // Helper Utilities
  // ==========================================================================

  /**
   * Formats raw bytes into human-readable strings (KB or MB).
   */
  function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    const kb = bytes / 1024;
    if (kb < 1024) return kb.toFixed(1) + ' KB';
    return (kb / 1024).toFixed(2) + ' MB';
  }

  /**
   * Safely escapes HTML strings to prevent XSS during dynamic DOM rendering.
   */
  function escapeHtml(str) {
    if (typeof str !== 'string') return '';
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  /**
   * Validates whether a given string is a safe HTTP/HTTPS URL.
   */
  function isValidHttpUrl(urlString) {
    if (typeof urlString !== 'string' || !urlString.trim()) return false;
    try {
      const parsed = new URL(urlString);
      return parsed.protocol === 'http:' || parsed.protocol === 'https:';
    } catch {
      return false;
    }
  }

  /**
   * Displays an alert notification banner to the user.
   */
  function showAlert(message, title = 'Notice', type = 'error') {
    elements.alertTitle.textContent = title;
    elements.alertMessage.textContent = message;

    elements.alertBanner.className = `alert alert-${type}`;
    elements.alertBanner.classList.remove('hidden');
  }

  /**
   * Dismisses and hides the alert banner.
   */
  function hideAlert() {
    elements.alertBanner.classList.add('hidden');
  }

  /**
   * Updates the header backend readiness indicator pill.
   */
  function updateReadinessUI(ready, statusText) {
    isModelReady = ready;
    elements.connectionStatus.className = 'status-pill ' + (ready ? 'status-ready' : 'status-offline');
    elements.connectionStatusText.textContent = statusText;
  }

  /**
   * Revokes the existing Object URL to prevent browser memory leaks.
   */
  function revokeActiveObjectUrl() {
    if (currentObjectUrl) {
      URL.revokeObjectURL(currentObjectUrl);
      currentObjectUrl = null;
    }
  }

  // ==========================================================================
  // File Validation & Handling
  // ==========================================================================

  /**
   * Validates selected file for MIME type, size, and dimensions.
   */
  function handleSelectedFile(file) {
    if (!file) return;

    hideAlert();

    // 1. MIME type validation
    const fileType = (file.type || '').toLowerCase();
    if (!ALLOWED_MIME_TYPES.has(fileType)) {
      elements.fileInput.value = '';
      showAlert(
        `Unsupported file type "${file.type || 'unknown'}". Please select a JPEG, PNG, or WEBP image.`,
        'Invalid Image Format',
        'error'
      );
      return;
    }

    // 2. File size validation (10 MB hard limit)
    if (file.size > MAX_FILE_SIZE_BYTES) {
      elements.fileInput.value = '';
      showAlert(
        `The selected file is too large (${formatFileSize(file.size)}). Uploads must not exceed 10 MB.`,
        'File Size Exceeded',
        'error'
      );
      return;
    }

    if (file.size === 0) {
      elements.fileInput.value = '';
      showAlert('The selected file is empty (0 bytes). Please choose a valid image.', 'Empty File', 'error');
      return;
    }

    // 3. Store file and generate object URL
    revokeActiveObjectUrl();
    currentFile = file;
    currentObjectUrl = URL.createObjectURL(file);

    // 4. Update preview UI
    elements.imagePreview.src = currentObjectUrl;
    elements.metaFilename.textContent = file.name;
    elements.metaFilesize.textContent = formatFileSize(file.size);
    elements.metaDimensions.textContent = 'Calculating...';

    // 5. Inspect image dimensions
    const imgInspector = new Image();
    imgInspector.onload = function () {
      const width = imgInspector.naturalWidth;
      const height = imgInspector.naturalHeight;
      elements.metaDimensions.textContent = `${width} \u00D7 ${height} px`;

      const numPixels = width * height;
      if (numPixels > MAX_PIXEL_COUNT) {
        showAlert(
          `Image resolution (${width} \u00D7 ${height} = ${(numPixels / 1000000).toFixed(1)} MP) ` +
            'exceeds the 20 Megapixel ceiling. The server will reject this image to prevent memory exhaustion.',
          'High Resolution Warning',
          'warning'
        );
      }
    };
    imgInspector.onerror = function () {
      elements.metaDimensions.textContent = 'Unknown (failed to decode)';
    };
    imgInspector.src = currentObjectUrl;

    // 6. Transition view states
    elements.dropZone.classList.add('hidden');
    elements.previewContainer.classList.remove('hidden');
    elements.loadingContainer.classList.add('hidden');
  }

  /**
   * Resets the entire analysis workflow to the initial state.
   */
  function resetWorkflow() {
    revokeActiveObjectUrl();
    currentFile = null;
    elements.fileInput.value = '';
    elements.imagePreview.src = '';
    elements.metaFilename.textContent = '\u2014';
    elements.metaDimensions.textContent = '\u2014';
    elements.metaFilesize.textContent = '\u2014';

    // Clear prediction & knowledge state
    lastPrediction = null;
    activeCandidateClassId = null;
    activeCandidateMeta = null;
    currentKnowledgeRequestId++;
    isKnowledgeLoading = false;

    // Reset views
    elements.dropZone.classList.remove('hidden');
    elements.previewContainer.classList.add('hidden');
    elements.loadingContainer.classList.add('hidden');

    elements.resultsContent.classList.add('hidden');
    elements.resultsEmpty.classList.remove('hidden');
    elements.newScanBtn.classList.add('hidden');

    // Reset knowledge section
    elements.knowledgeSection.classList.add('hidden');
    elements.candidateExplorationNotice.classList.add('hidden');
    elements.knowledgeLoading.classList.add('hidden');
    elements.knowledgeError.classList.add('hidden');
    elements.knowledgeDetails.classList.add('hidden');
    elements.knowledgeDetails.textContent = '';

    hideAlert();
  }

  // ==========================================================================
  // Inference & API Dispatch
  // ==========================================================================

  /**
   * Executes leaf disease prediction via the API.
   */
  async function performAnalysis() {
    if (!currentFile || isAnalyzing) return;

    if (!isModelReady) {
      // Re-probe readiness once in case model finished loading
      const readyCheck = await window.CropDiseaseApi.checkReady();
      if (!readyCheck.ready) {
        updateReadinessUI(
          false,
          readyCheck.status === 'not_ready' ? 'Model Initializing' : 'Backend Offline'
        );
        showAlert(
          'The prediction backend is currently unavailable or initializing. ' +
            'Please ensure the FastAPI service is running at http://127.0.0.1:8000.',
          'Service Unavailable',
          'error'
        );
        return;
      }
      updateReadinessUI(true, 'Model Ready');
    }

    hideAlert();
    isAnalyzing = true;

    // Concurrency guard: Invalidate any pending knowledge requests
    currentKnowledgeRequestId++;
    isKnowledgeLoading = false;
    activeCandidateClassId = null;
    activeCandidateMeta = null;

    // UI state transition: Loading active and clear stale results
    elements.analyzeBtn.disabled = true;
    elements.removeBtn.disabled = true;
    elements.previewContainer.classList.add('hidden');
    elements.loadingContainer.classList.remove('hidden');
    elements.resultsContent.classList.add('hidden');
    elements.resultsEmpty.classList.remove('hidden');
    elements.newScanBtn.classList.add('hidden');

    // Hide and clear previous knowledge display
    elements.knowledgeSection.classList.add('hidden');
    elements.knowledgeDetails.textContent = '';

    try {
      const response = await window.CropDiseaseApi.predictCropDisease(currentFile);
      lastPrediction = response;

      // Set active candidate to the top predicted class
      activeCandidateClassId = response.predicted_class_id;
      activeCandidateMeta = {
        isPrimary: true,
        crop: response.crop,
        condition: response.condition,
        rank: 1,
        score: response.score,
      };

      // 1. Render prediction results immediately (never blocked by knowledge retrieval)
      renderResults(response);

      // 2. Fetch and display agronomic knowledge asynchronously
      fetchAndRenderKnowledge(response.predicted_class_id, activeCandidateMeta);
    } catch (err) {
      showAlert(err.message, 'Analysis Failed', 'error');
      // Restore preview view on failure (preserves selected image and allows retry)
      elements.previewContainer.classList.remove('hidden');
      elements.loadingContainer.classList.add('hidden');

      // Update readiness indicator if backend is unreachable or model is unready
      const status = typeof err?.status === 'number' ? err.status : null;
      if (status === 0 || status === 408) {
        updateReadinessUI(false, 'Backend Offline');
      } else if (status === 503) {
        updateReadinessUI(false, 'Model Unavailable');
      }
    } finally {
      isAnalyzing = false;
      elements.analyzeBtn.disabled = false;
      elements.removeBtn.disabled = false;
    }
  }

  /**
   * Renders the structured prediction response in the results card.
   */
  function renderResults(data) {
    // 1. Primary Diagnosis Details
    elements.resultCrop.textContent = data.crop || 'Unknown Crop';
    elements.resultCondition.textContent = data.condition || 'Unknown Condition';

    // Health badge
    const isHealthy = (data.health || '').toLowerCase() === 'healthy';
    elements.resultHealthBadge.textContent = isHealthy ? '\u2713 Healthy' : '\u26A0 Diseased';
    elements.resultHealthBadge.className = 'badge ' + (isHealthy ? 'badge-healthy' : 'badge-diseased');

    // Score formatting
    const scorePercent = (data.score * 100).toFixed(2);
    elements.resultScore.textContent = `${scorePercent}%`;

    // 2. Render Top 3 Candidate Predictions
    renderCandidatesList();

    // 3. Disclaimer text
    elements.resultDisclaimer.textContent = data.disclaimer || '';

    // 4. View Transitions
    elements.loadingContainer.classList.add('hidden');
    elements.previewContainer.classList.remove('hidden');
    elements.resultsEmpty.classList.add('hidden');
    elements.resultsContent.classList.remove('hidden');
    elements.newScanBtn.classList.remove('hidden');

    // Smooth scroll to results on mobile devices
    if (window.innerWidth < 900) {
      elements.resultsContent.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }

  /**
   * Renders the Top 3 Candidate list using safe DOM creation methods.
   * Completely avoids innerHTML for data values to prevent XSS.
   */
  function renderCandidatesList() {
    if (!lastPrediction || !Array.isArray(lastPrediction.top_predictions)) return;

    elements.candidatesList.textContent = ''; // Safe clear

    lastPrediction.top_predictions.forEach((cand, idx) => {
      // Validate candidate object and class_id
      if (!cand || typeof cand !== 'object') return;
      const isValidClassId = Number.isInteger(cand.class_id) && cand.class_id >= 0 && cand.class_id <= 14;
      const rank = Number.isInteger(cand.rank) ? cand.rank : idx + 1;
      const scoreVal = typeof cand.score === 'number' && !isNaN(cand.score) ? cand.score : 0;
      const candPercent = (scoreVal * 100).toFixed(2);
      const isCandHealthy = (cand.health || '').toLowerCase() === 'healthy';
      const isPrimaryRank = rank === 1 || cand.class_id === lastPrediction.predicted_class_id;
      const isActiveSelection = isValidClassId && cand.class_id === activeCandidateClassId;

      // Item container
      const itemEl = document.createElement('div');
      itemEl.className = 'candidate-item' + (isActiveSelection ? ' candidate-active' : '');

      // Header row
      const headerEl = document.createElement('div');
      headerEl.className = 'candidate-header';

      const leftEl = document.createElement('div');
      leftEl.className = 'candidate-left';

      const rankBadge = document.createElement('span');
      rankBadge.className = 'rank-badge';
      rankBadge.setAttribute('aria-label', `Rank ${rank}`);
      rankBadge.textContent = `#${rank}`;
      leftEl.appendChild(rankBadge);

      const nameEl = document.createElement('span');
      nameEl.className = 'candidate-name';
      nameEl.textContent = `${cand.crop || 'Unknown'} \u2014 ${cand.condition || 'Unknown'}`;
      leftEl.appendChild(nameEl);

      const healthBadge = document.createElement('span');
      healthBadge.className = `badge ${isCandHealthy ? 'badge-healthy' : 'badge-diseased'}`;
      healthBadge.style.fontSize = '0.7rem';
      healthBadge.style.padding = '0.15rem 0.5rem';
      healthBadge.textContent = isCandHealthy ? '\u2713 Healthy' : '\u26A0 Diseased';
      leftEl.appendChild(healthBadge);

      if (isPrimaryRank) {
        const primaryBadge = document.createElement('span');
        primaryBadge.className = 'badge badge-primary-candidate';
        primaryBadge.style.fontSize = '0.7rem';
        primaryBadge.style.padding = '0.15rem 0.5rem';
        primaryBadge.textContent = 'Primary Prediction';
        leftEl.appendChild(primaryBadge);
      }

      headerEl.appendChild(leftEl);

      const scoreEl = document.createElement('span');
      scoreEl.className = 'candidate-score';
      scoreEl.textContent = `${candPercent}%`;
      headerEl.appendChild(scoreEl);

      itemEl.appendChild(headerEl);

      // Progress bar
      const trackEl = document.createElement('div');
      trackEl.className = 'progress-track';
      trackEl.setAttribute('role', 'progressbar');
      trackEl.setAttribute('aria-valuenow', candPercent);
      trackEl.setAttribute('aria-valuemin', '0');
      trackEl.setAttribute('aria-valuemax', '100');
      trackEl.setAttribute('aria-label', `${cand.condition || 'Condition'} score`);

      const fillEl = document.createElement('div');
      fillEl.className = rank === 1 ? 'progress-fill' : 'progress-fill progress-fill-secondary';
      fillEl.style.width = `${Math.min(100, Math.max(0, scoreVal * 100))}%`;
      trackEl.appendChild(fillEl);

      itemEl.appendChild(trackEl);

      // Interactive Candidate Exploration Action Row (Only if valid class ID)
      if (isValidClassId) {
        const actionsEl = document.createElement('div');
        actionsEl.className = 'candidate-actions';

        const noteEl = document.createElement('span');
        noteEl.className = 'candidate-status-note';
        noteEl.textContent = isActiveSelection
          ? 'Viewing reference guidance below'
          : (isPrimaryRank ? 'Explore primary guidance' : 'Explore alternative guidance');
        actionsEl.appendChild(noteEl);

        const btnEl = document.createElement('button');
        btnEl.type = 'button';
        btnEl.className = 'btn btn-sm ' + (isActiveSelection ? 'btn-candidate-active' : 'btn-candidate-select');
        btnEl.textContent = isActiveSelection ? 'Active Guidance' : 'Explore Guidance';
        btnEl.setAttribute(
          'aria-label',
          isActiveSelection
            ? `Currently viewing guidance for ${cand.crop} ${cand.condition}`
            : `Explore agronomic reference guidance for candidate #${rank}: ${cand.crop} ${cand.condition}`
        );

        if (isActiveSelection) {
          btnEl.disabled = true;
          btnEl.setAttribute('aria-current', 'true');
        } else {
          btnEl.addEventListener('click', (e) => {
            e.stopPropagation();
            selectCandidateForExploration(cand.class_id, {
              isPrimary: cand.class_id === lastPrediction.predicted_class_id,
              crop: cand.crop,
              condition: cand.condition,
              rank: rank,
              score: scoreVal,
            });
          });
        }

        actionsEl.appendChild(btnEl);
        itemEl.appendChild(actionsEl);
      }

      elements.candidatesList.appendChild(itemEl);
    });
  }

  /**
   * User interaction handler to select a candidate for reference information exploration.
   */
  function selectCandidateForExploration(classId, meta) {
    if (!Number.isInteger(classId) || classId < 0 || classId > 14) {
      showAlert('Invalid candidate class identifier.', 'Selection Error', 'error');
      return;
    }

    // Prevent duplicate simultaneous requests for the same active candidate selection
    if (classId === activeCandidateClassId && isKnowledgeLoading) {
      return;
    }

    activeCandidateClassId = classId;
    activeCandidateMeta = meta;

    // Refresh candidate list UI to reflect current selection
    renderCandidatesList();

    // Fetch and display candidate knowledge
    fetchAndRenderKnowledge(classId, meta);
  }

  // ==========================================================================
  // Knowledge Base Retrieval & Rendering
  // ==========================================================================

  /**
   * Fetches and renders agronomic disease reference information.
   * Employs sequence token guarding to avoid stale response race conditions.
   */
  async function fetchAndRenderKnowledge(classId, meta) {
    if (!Number.isInteger(classId) || classId < 0 || classId > 14) {
      return;
    }

    // Sequence token to handle race conditions / out-of-order candidate switching
    const reqId = ++currentKnowledgeRequestId;
    isKnowledgeLoading = true;

    // Ensure knowledge section is visible
    elements.knowledgeSection.classList.remove('hidden');

    // Transition loading states
    elements.knowledgeLoading.classList.remove('hidden');
    elements.knowledgeError.classList.add('hidden');
    elements.knowledgeDetails.classList.add('hidden');

    // Update Context Badge & Notice
    if (meta.isPrimary) {
      elements.knowledgeContextBadge.textContent = 'Primary Diagnosis Reference';
      elements.knowledgeContextBadge.className = 'badge badge-primary-subtle';
      elements.candidateExplorationNotice.classList.add('hidden');
    } else {
      elements.knowledgeContextBadge.textContent = `Candidate #${meta.rank} Reference (Not Diagnosis)`;
      elements.knowledgeContextBadge.className = 'badge badge-warning';
      elements.candidateExplorationNotice.classList.remove('hidden');
      elements.candidateExplorationDesc.textContent =
        `You are exploring reference information for alternative candidate #${meta.rank}: ` +
        `${meta.crop} \u2014 ${meta.condition} (model score ${(meta.score * 100).toFixed(2)}%). ` +
        `The primary model prediction above remains unchanged.`;
    }

    try {
      const knowledgeData = await window.CropDiseaseApi.getDiseaseKnowledge(classId);

      // Stale response guard: If another request was dispatched while this was pending, drop it
      if (reqId !== currentKnowledgeRequestId) {
        return;
      }

      // Render knowledge content safely into DOM
      renderKnowledgeDetails(knowledgeData, meta);

      elements.knowledgeLoading.classList.add('hidden');
      elements.knowledgeDetails.classList.remove('hidden');
    } catch (err) {
      // Stale error guard
      if (reqId !== currentKnowledgeRequestId) {
        return;
      }

      elements.knowledgeLoading.classList.add('hidden');
      elements.knowledgeDetails.classList.add('hidden');
      elements.knowledgeError.classList.remove('hidden');

      // Keep prediction results visible; only inform about reference unavailability
      elements.knowledgeErrorMessage.textContent =
        'Agronomic reference details could not be loaded at this time. ' +
        'The model prediction results above remain valid.';
    } finally {
      if (reqId === currentKnowledgeRequestId) {
        isKnowledgeLoading = false;
      }
    }
  }

  // ==========================================================================
  // Safe DOM Builders for Knowledge Section
  // ==========================================================================

  function createKnowledgeBlock(title, iconText) {
    const block = document.createElement('div');
    block.className = 'knowledge-block';

    const header = document.createElement('div');
    header.className = 'knowledge-block-header';

    const h5 = document.createElement('h5');
    h5.className = 'knowledge-block-title';

    if (iconText) {
      const iconSpan = document.createElement('span');
      iconSpan.setAttribute('aria-hidden', 'true');
      iconSpan.textContent = iconText;
      h5.appendChild(iconSpan);
    }

    const titleText = document.createTextNode(title);
    h5.appendChild(titleText);

    header.appendChild(h5);
    block.appendChild(header);
    return { block, header };
  }

  function createSubtitle(text) {
    const p = document.createElement('p');
    p.className = 'knowledge-block-subtitle';
    p.textContent = text;
    return p;
  }

  function createText(text) {
    const p = document.createElement('p');
    p.className = 'knowledge-text';
    p.textContent = text;
    return p;
  }

  function createList(items) {
    const ul = document.createElement('ul');
    ul.className = 'knowledge-list';
    if (Array.isArray(items)) {
      items.forEach((item) => {
        if (typeof item === 'string' && item.trim()) {
          const li = document.createElement('li');
          li.textContent = item;
          ul.appendChild(li);
        }
      });
    }
    return ul;
  }

  function createCallout(text, type = 'info') {
    const div = document.createElement('div');
    div.className = `knowledge-callout knowledge-callout-${type}`;
    div.textContent = text;
    return div;
  }

  /**
   * Safely constructs the entire knowledge section in the DOM without innerHTML.
   */
  function renderKnowledgeDetails(data, meta) {
    const container = elements.knowledgeDetails;
    container.textContent = ''; // Safe clear

    if (!data || typeof data !== 'object') return;

    const isHealthy = (data.health_status || '').toLowerCase() === 'healthy';

    // ------------------------------------------------------------------------
    // 1. Overview & Pathogen Identification
    // ------------------------------------------------------------------------
    const overview = createKnowledgeBlock('Agronomic Profile Overview', '📋');

    const metaSummaryRow = document.createElement('div');
    metaSummaryRow.style.display = 'flex';
    metaSummaryRow.style.alignItems = 'center';
    metaSummaryRow.style.justifyContent = 'space-between';
    metaSummaryRow.style.flexWrap = 'wrap';
    metaSummaryRow.style.gap = '0.5rem';

    const metaTitle = document.createElement('strong');
    metaTitle.style.fontSize = '1.05rem';
    metaTitle.style.color = 'var(--color-text-main)';
    metaTitle.textContent = `${data.crop || 'Crop'} \u2014 ${data.condition || 'Condition'}`;
    metaSummaryRow.appendChild(metaTitle);

    const healthBadge = document.createElement('span');
    healthBadge.className = `badge ${isHealthy ? 'badge-healthy' : 'badge-diseased'}`;
    healthBadge.textContent = isHealthy ? '\u2713 Healthy Crop' : '\u26A0 Diseased Crop';
    metaSummaryRow.appendChild(healthBadge);

    overview.block.appendChild(metaSummaryRow);

    const causalAgentP = document.createElement('p');
    causalAgentP.className = 'knowledge-text';
    const causalLabel = document.createElement('strong');
    causalLabel.textContent = 'Causal Agent: ';
    causalAgentP.appendChild(causalLabel);

    if (data.causal_agent) {
      const agentSpan = document.createElement('span');
      agentSpan.style.fontStyle = 'italic';
      agentSpan.textContent = data.causal_agent;
      causalAgentP.appendChild(agentSpan);
    } else {
      const noneSpan = document.createElement('span');
      noneSpan.textContent = 'None (Healthy Physiological State)';
      causalAgentP.appendChild(noneSpan);
    }
    overview.block.appendChild(causalAgentP);

    container.appendChild(overview.block);

    // ------------------------------------------------------------------------
    // 2. Diagnostic Visual Symptoms
    // ------------------------------------------------------------------------
    if (data.symptoms) {
      const symBlock = createKnowledgeBlock('Diagnostic Visual Symptoms', '🔍');

      if (data.symptoms.summary) {
        symBlock.block.appendChild(createText(data.symptoms.summary));
      }

      if (Array.isArray(data.symptoms.foliar_symptoms) && data.symptoms.foliar_symptoms.length > 0) {
        symBlock.block.appendChild(createSubtitle('Foliar Signs & Leaf Lesions'));
        symBlock.block.appendChild(createList(data.symptoms.foliar_symptoms));
      }

      if (Array.isArray(data.symptoms.stem_and_fruit_symptoms) && data.symptoms.stem_and_fruit_symptoms.length > 0) {
        symBlock.block.appendChild(createSubtitle('Stem & Fruit Symptoms'));
        symBlock.block.appendChild(createList(data.symptoms.stem_and_fruit_symptoms));
      }

      if (Array.isArray(data.symptoms.look_alike_conditions) && data.symptoms.look_alike_conditions.length > 0) {
        symBlock.block.appendChild(createSubtitle('Differential Diagnosis & Look-Alikes'));
        symBlock.block.appendChild(createList(data.symptoms.look_alike_conditions));
      }

      container.appendChild(symBlock.block);
    }

    // ------------------------------------------------------------------------
    // 3. Favorable Environmental Conditions & Transmission
    // ------------------------------------------------------------------------
    if (data.favorable_conditions) {
      const envBlock = createKnowledgeBlock('Favorable Environmental Conditions & Transmission', '🌦️');

      if (data.favorable_conditions.environmental_factors) {
        envBlock.block.appendChild(createSubtitle('Environmental Drivers'));
        envBlock.block.appendChild(createText(data.favorable_conditions.environmental_factors));
      }

      if (
        Array.isArray(data.favorable_conditions.transmission_routes) &&
        data.favorable_conditions.transmission_routes.length > 0
      ) {
        envBlock.block.appendChild(createSubtitle('Vectors & Transmission Pathways'));
        envBlock.block.appendChild(createList(data.favorable_conditions.transmission_routes));
      }

      container.appendChild(envBlock.block);
    }

    // ------------------------------------------------------------------------
    // 4. Cultural Prevention & Field Hygiene
    // ------------------------------------------------------------------------
    if (data.prevention_and_cultural) {
      const prevBlock = createKnowledgeBlock('Cultural Prevention & Agronomic Practices', '🛡️');

      if (Array.isArray(data.prevention_and_cultural.sanitation) && data.prevention_and_cultural.sanitation.length > 0) {
        prevBlock.block.appendChild(createSubtitle('Field Sanitation & Hygiene'));
        prevBlock.block.appendChild(createList(data.prevention_and_cultural.sanitation));
      }

      if (data.prevention_and_cultural.crop_rotation) {
        prevBlock.block.appendChild(createSubtitle('Crop Rotation Schedule'));
        prevBlock.block.appendChild(createText(data.prevention_and_cultural.crop_rotation));
      }

      if (
        Array.isArray(data.prevention_and_cultural.irrigation_and_airflow) &&
        data.prevention_and_cultural.irrigation_and_airflow.length > 0
      ) {
        prevBlock.block.appendChild(createSubtitle('Canopy Airflow & Irrigation Management'));
        prevBlock.block.appendChild(createList(data.prevention_and_cultural.irrigation_and_airflow));
      }

      if (
        Array.isArray(data.prevention_and_cultural.resistant_cultivars) &&
        data.prevention_and_cultural.resistant_cultivars.length > 0
      ) {
        prevBlock.block.appendChild(createSubtitle('Genetic Resistance & Variety Selection'));
        prevBlock.block.appendChild(createList(data.prevention_and_cultural.resistant_cultivars));
      }

      container.appendChild(prevBlock.block);
    }

    // ------------------------------------------------------------------------
    // 5. Treatment / Integrated Pest Management (Strict Healthy vs Diseased)
    // ------------------------------------------------------------------------
    if (data.treatment) {
      const treatmentTitle = isHealthy
        ? 'Crop Health Maintenance (No Treatment Required)'
        : 'Integrated Pest & Disease Management (IPM)';
      const treatmentIcon = isHealthy ? '🌱' : '💊';

      const treatBlock = createKnowledgeBlock(treatmentTitle, treatmentIcon);

      if (data.treatment.urgency) {
        const calloutType = isHealthy ? 'healthy' : 'warning';
        treatBlock.block.appendChild(createCallout(data.treatment.urgency, calloutType));
      }

      if (Array.isArray(data.treatment.non_chemical_controls) && data.treatment.non_chemical_controls.length > 0) {
        treatBlock.block.appendChild(
          createSubtitle(isHealthy ? 'Routine Non-Chemical Maintenance' : 'Non-Chemical & Cultural Controls')
        );
        treatBlock.block.appendChild(createList(data.treatment.non_chemical_controls));
      }

      if (Array.isArray(data.treatment.ipm_chemical_principles) && data.treatment.ipm_chemical_principles.length > 0) {
        treatBlock.block.appendChild(
          createSubtitle(isHealthy ? 'Chemical Application Guidance' : 'IPM Chemical Management Principles')
        );
        treatBlock.block.appendChild(createList(data.treatment.ipm_chemical_principles));
      }

      if (
        Array.isArray(data.treatment.safety_and_regulatory_constraints) &&
        data.treatment.safety_and_regulatory_constraints.length > 0
      ) {
        treatBlock.block.appendChild(createSubtitle('Safety & Regulatory Constraints'));
        treatBlock.block.appendChild(createList(data.treatment.safety_and_regulatory_constraints));
      }

      container.appendChild(treatBlock.block);
    }

    // ------------------------------------------------------------------------
    // 6. Agricultural Extension Consultation
    // ------------------------------------------------------------------------
    if (data.extension_consultation) {
      const extBlock = createKnowledgeBlock('Agricultural Extension & Specialist Referral', '📞');

      if (data.extension_consultation.local_specialist) {
        extBlock.block.appendChild(createSubtitle('Recommended Extension Authority'));
        extBlock.block.appendChild(createText(data.extension_consultation.local_specialist));
      }

      if (
        Array.isArray(data.extension_consultation.escalation_triggers) &&
        data.extension_consultation.escalation_triggers.length > 0
      ) {
        extBlock.block.appendChild(createSubtitle('Field Escalation Triggers (Inspection Required)'));
        extBlock.block.appendChild(createList(data.extension_consultation.escalation_triggers));
      }

      container.appendChild(extBlock.block);
    }

    // ------------------------------------------------------------------------
    // 7. Diagnostic Uncertainty & Field Verification Notice
    // ------------------------------------------------------------------------
    if (data.safety_and_uncertainty_notes) {
      const noticeDiv = document.createElement('div');
      noticeDiv.className = 'knowledge-callout knowledge-callout-warning';
      noticeDiv.setAttribute('role', 'note');

      const strongEl = document.createElement('strong');
      strongEl.textContent = 'Diagnostic Uncertainty & Field Verification: ';
      noticeDiv.appendChild(strongEl);

      const noteText = document.createTextNode(data.safety_and_uncertainty_notes);
      noticeDiv.appendChild(noteText);

      container.appendChild(noticeDiv);
    }

    // ------------------------------------------------------------------------
    // 8. Scientific & Extension References (Untrusted Data & Safe URLs)
    // ------------------------------------------------------------------------
    if (Array.isArray(data.sources) && data.sources.length > 0) {
      const sourcesBlock = createKnowledgeBlock('Scientific & Extension Source References', '📚');

      const sourceList = document.createElement('div');
      sourceList.className = 'source-list';

      data.sources.forEach((src) => {
        if (!src || typeof src !== 'object') return;

        const srcItem = document.createElement('div');
        srcItem.className = 'source-item';

        const srcOrg = document.createElement('span');
        srcOrg.className = 'source-org';
        srcOrg.textContent = src.organization || 'Institutional Agricultural Source';
        srcItem.appendChild(srcOrg);

        const srcTitle = document.createElement('h6');
        srcTitle.className = 'source-title';
        srcTitle.textContent = src.title || 'Agronomic Reference Publication';
        srcItem.appendChild(srcTitle);

        if (src.claims_supported) {
          const srcClaims = document.createElement('p');
          srcClaims.className = 'source-claims';
          const claimsLabel = document.createElement('strong');
          claimsLabel.textContent = 'Claims Substantiated: ';
          srcClaims.appendChild(claimsLabel);
          srcClaims.appendChild(document.createTextNode(src.claims_supported));
          srcItem.appendChild(srcClaims);
        }

        // Validate source URL before making it an active hyperlink
        if (isValidHttpUrl(src.url)) {
          const srcLink = document.createElement('a');
          srcLink.className = 'source-link';
          srcLink.href = src.url;
          srcLink.target = '_blank';
          srcLink.rel = 'noopener noreferrer';
          srcLink.setAttribute('target', '_blank');
          srcLink.setAttribute('rel', 'noopener noreferrer');
          srcLink.textContent = 'View Official Reference \u2197';
          srcLink.setAttribute(
            'aria-label',
            `View official reference: ${src.title || 'Publication'} from ${src.organization || 'Publisher'} (opens in a new tab)`
          );
          srcItem.appendChild(srcLink);
        } else {
          const srcDisabled = document.createElement('span');
          srcDisabled.className = 'source-link-disabled';
          srcDisabled.textContent =
            'Publication reference: ' +
            (src.reference_url_or_publication || 'Official Institutional Extension Archives');
          srcItem.appendChild(srcDisabled);
        }

        sourceList.appendChild(srcItem);
      });

      sourcesBlock.block.appendChild(sourceList);
      container.appendChild(sourcesBlock.block);
    }
  }

  // ==========================================================================
  // Event Listeners Setup
  // ==========================================================================
  function setupEventListeners() {
    // Dropzone Click & Keyboard activation
    elements.dropZone.addEventListener('click', () => elements.fileInput.click());
    elements.browseBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      elements.fileInput.click();
    });
    elements.dropZone.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        elements.fileInput.click();
      }
    });

    // File Input change
    elements.fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleSelectedFile(e.target.files[0]);
      }
    });

    // Drag and Drop handlers
    ['dragenter', 'dragover'].forEach((eventName) => {
      elements.dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        elements.dropZone.classList.add('drag-over');
      });
    });

    ['dragleave', 'drop'].forEach((eventName) => {
      elements.dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        elements.dropZone.classList.remove('drag-over');
      });
    });

    elements.dropZone.addEventListener('drop', (e) => {
      const dt = e.dataTransfer;
      if (dt && dt.files && dt.files.length > 0) {
        handleSelectedFile(dt.files[0]);
      }
    });

    // Preview actions
    elements.analyzeBtn.addEventListener('click', performAnalysis);
    elements.removeBtn.addEventListener('click', resetWorkflow);
    elements.newScanBtn.addEventListener('click', resetWorkflow);

    // Knowledge Retry action
    elements.knowledgeRetryBtn.addEventListener('click', () => {
      if (activeCandidateClassId !== null && activeCandidateMeta) {
        fetchAndRenderKnowledge(activeCandidateClassId, activeCandidateMeta);
      }
    });

    // Alert dismiss
    elements.alertDismissBtn.addEventListener('click', hideAlert);

    // Prevent page drop defaults
    window.addEventListener('dragover', (e) => e.preventDefault());
    window.addEventListener('drop', (e) => e.preventDefault());
  }

  // ==========================================================================
  // Application Bootstrap & Readiness Probe
  // ==========================================================================
  async function init() {
    setupEventListeners();

    // Initial readiness probe
    try {
      const readyStatus = await window.CropDiseaseApi.checkReady();
      if (readyStatus.ready) {
        updateReadinessUI(true, 'Model Ready');
      } else if (readyStatus.status === 'not_ready') {
        updateReadinessUI(false, 'Model Initializing');
      } else {
        updateReadinessUI(false, 'Backend Offline');
      }
    } catch {
      updateReadinessUI(false, 'Backend Offline');
    }
  }

  // Run on DOM load
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
