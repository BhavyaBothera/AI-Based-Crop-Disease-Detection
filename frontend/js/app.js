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
  let lastReadyState = null;
  let lastPrediction = null;
  let activeCandidateClassId = null;
  let activeCandidateMeta = null;
  let currentKnowledgeRequestId = 0;
  let isKnowledgeLoading = false;
  let currentActiveKnowledgeData = null;
  let toastTimeout = null;

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

    // Unsupported / abstained input state
    resultsRejected: document.getElementById('results-rejected'),
    rejectionBadge: document.getElementById('rejection-badge'),
    rejectionHeadline: document.getElementById('rejection-headline'),
    rejectionDetail: document.getElementById('rejection-detail'),
    rejectionActionBtn: document.getElementById('rejection-action-btn'),

    // Primary Diagnosis
    resultCrop: document.getElementById('result-crop'),
    resultCondition: document.getElementById('result-condition'),
    resultHealthBadge: document.getElementById('result-health-badge'),
    resultScore: document.getElementById('result-score'),
    confidenceBandBadge: document.getElementById('confidence-band-badge'),
    confidenceGuidanceText: document.getElementById('confidence-guidance-text'),
    diagnosisTelemetry: document.getElementById('diagnosis-telemetry'),
    telemetryTotal: document.getElementById('telemetry-total'),
    telemetryBreakdown: document.getElementById('telemetry-breakdown'),
    telemetryModel: document.getElementById('telemetry-model'),
    candidatesList: document.getElementById('candidates-list'),
    resultDisclaimer: document.getElementById('result-disclaimer'),

    // Recent Scans
    recentScansSection: document.getElementById('recent-scans-section'),
    recentScansList: document.getElementById('recent-scans-list'),
    clearHistoryBtn: document.getElementById('clear-history-btn'),

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
    cameraBtn: document.getElementById('camera-btn'),
    cameraInput: document.getElementById('camera-input'),
    printAdvisoryBtn: document.getElementById('print-advisory-btn'),
    copyGuidanceBtn: document.getElementById('copy-guidance-btn'),
    printReportContainer: document.getElementById('print-report-container'),

    // Toast Notification Elements
    toast: document.getElementById('toast'),
    toastMessage: document.getElementById('toast-message'),
    toastIcon: document.getElementById('toast-icon'),
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

  /**
   * Clears every diagnostic artifact. Called whenever a new analysis starts, a file is
   * replaced, or a request fails, so a stale diagnosis can never survive a rejection.
   */
  function clearPredictionState() {
    lastPrediction = null;
    activeCandidateClassId = null;
    activeCandidateMeta = null;
    currentKnowledgeRequestId++;
    isKnowledgeLoading = false;
    currentActiveKnowledgeData = null;

    // Remove rendered diagnosis content from the DOM as well, so stale candidates or a
    // previous condition cannot linger behind a rejection, in print output or in the
    // accessibility tree.
    if (elements.candidatesList) elements.candidatesList.textContent = '';
    if (elements.resultCrop) elements.resultCrop.textContent = '\u2014';
    if (elements.resultCondition) elements.resultCondition.textContent = '\u2014';
    if (elements.resultScore) elements.resultScore.textContent = '0.0%';
    if (elements.resultDisclaimer) elements.resultDisclaimer.textContent = '';
    if (elements.resultHealthBadge) {
      elements.resultHealthBadge.textContent = '\u2014';
      elements.resultHealthBadge.className = 'badge';
    }
    if (elements.confidenceBandBadge) {
      elements.confidenceBandBadge.textContent = 'Score \u2014';
      elements.confidenceBandBadge.className = 'badge';
    }
    if (elements.confidenceGuidanceText) {
      elements.confidenceGuidanceText.textContent = 'Relative softmax score across 15 trained classes';
    }
    if (elements.diagnosisTelemetry) {
      elements.diagnosisTelemetry.classList.add('hidden');
    }
    const thumb = document.getElementById('result-specimen-thumb');
    if (thumb) {
      thumb.src = '';
      const wrap = thumb.closest('.diagnosis-thumb-wrapper');
      if (wrap) wrap.classList.add('hidden');
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

    // 7. Selecting a different image invalidates the previous outcome entirely
    clearPredictionState();
    elements.resultsContent.classList.add('hidden');
    elements.resultsRejected.classList.add('hidden');
    elements.resultsEmpty.classList.remove('hidden');
    elements.newScanBtn.classList.add('hidden');
    elements.knowledgeSection.classList.add('hidden');
    elements.knowledgeDetails.textContent = '';
    if (elements.printAdvisoryBtn) elements.printAdvisoryBtn.classList.add('hidden');
    if (elements.copyGuidanceBtn) elements.copyGuidanceBtn.classList.add('hidden');
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
    clearPredictionState();

    // Reset views
    elements.dropZone.classList.remove('hidden');
    elements.previewContainer.classList.add('hidden');
    elements.loadingContainer.classList.add('hidden');

    elements.resultsContent.classList.add('hidden');
    elements.resultsRejected.classList.add('hidden');
    elements.resultsEmpty.classList.remove('hidden');
    elements.newScanBtn.classList.add('hidden');
    elements.rejectionBadge.textContent = 'Image not verified';
    elements.rejectionDetail.textContent = '';
    if (elements.printAdvisoryBtn) elements.printAdvisoryBtn.classList.add('hidden');
    if (elements.copyGuidanceBtn) elements.copyGuidanceBtn.classList.add('hidden');

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

    // Concurrency guard: invalidate pending knowledge requests and clear all diagnostic
    // state before the request, so nothing stale can be shown if the input is rejected.
    clearPredictionState();

    // UI state transition: Loading active and clear stale results
    elements.analyzeBtn.disabled = true;
    elements.removeBtn.disabled = true;
    elements.previewContainer.classList.add('hidden');
    elements.loadingContainer.classList.remove('hidden');
    elements.resultsContent.classList.add('hidden');
    elements.resultsRejected.classList.add('hidden');
    elements.resultsEmpty.classList.remove('hidden');
    elements.newScanBtn.classList.add('hidden');
    if (elements.printAdvisoryBtn) elements.printAdvisoryBtn.classList.add('hidden');
    if (elements.copyGuidanceBtn) elements.copyGuidanceBtn.classList.add('hidden');

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
      // A failed or declined analysis must never leave a previous diagnosis on screen.
      clearPredictionState();

      const isInputRejection = Boolean(err) && err.name === 'InputRejectedError';

      if (isInputRejection) {
        // Polished PHYTOSIGNAL rejection state: no diagnosis, candidates or guidance.
        renderInputRejection(err);
      } else {
        elements.resultsContent.classList.add('hidden');
        elements.resultsRejected.classList.add('hidden');
        elements.resultsEmpty.classList.remove('hidden');
        if (elements.printAdvisoryBtn) elements.printAdvisoryBtn.classList.add('hidden');
        if (elements.copyGuidanceBtn) elements.copyGuidanceBtn.classList.add('hidden');
        showAlert(err.message, 'Analysis Failed', 'error');
      }

      elements.knowledgeSection.classList.add('hidden');
      elements.knowledgeDetails.textContent = '';

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

    // Relative Softmax Evaluation (Softmax scores are not calibrated diagnostic probabilities)
    const scoreVal = typeof data.score === 'number' ? data.score : 0;
    if (elements.confidenceBandBadge) {
      if (scoreVal >= 0.90) {
        elements.confidenceBandBadge.textContent = 'Top Candidate';
        elements.confidenceBandBadge.className = 'badge badge-confidence-high';
        if (elements.confidenceGuidanceText) {
          elements.confidenceGuidanceText.textContent = 'Dominant relative softmax score across 15 trained classes';
        }
      } else if (scoreVal >= 0.70) {
        elements.confidenceBandBadge.textContent = 'Primary Candidate';
        elements.confidenceBandBadge.className = 'badge badge-confidence-moderate';
        if (elements.confidenceGuidanceText) {
          elements.confidenceGuidanceText.textContent = 'Review differential candidate classes below before action';
        }
      } else {
        elements.confidenceBandBadge.textContent = 'Diffuse Score Distribution';
        elements.confidenceBandBadge.className = 'badge badge-confidence-low';
        if (elements.confidenceGuidanceText) {
          elements.confidenceGuidanceText.textContent = 'Scores distributed across candidates; inspect differential monographs';
        }
      }
    }

    // Telemetry & Latency Rendering
    if (data.timing && elements.diagnosisTelemetry) {
      const t = data.timing;
      if (elements.telemetryTotal) elements.telemetryTotal.textContent = `⚡ Latency: ${t.total_ms}ms`;
      if (elements.telemetryBreakdown) elements.telemetryBreakdown.textContent = `🛡️ Gate: ${t.gate_ms}ms · 🧠 Neural: ${t.inference_ms}ms`;
      if (elements.telemetryModel) {
        const meta = data.model_metadata || {};
        const arch = meta.model_architecture || 'MobileNetV2';
        const f1 = meta.val_macro_f1 ? ` (val-F1: ${meta.val_macro_f1})` : '';
        elements.telemetryModel.textContent = `🔬 ${arch}${f1}`;
      }
      elements.diagnosisTelemetry.classList.remove('hidden');
    }

    // Save successful scan to local history
    saveToRecentScans(data, elements.imagePreview ? elements.imagePreview.src : null);

    // Specimen Thumbnail in diagnosis header
    const thumb = document.getElementById('result-specimen-thumb');
    if (thumb && elements.imagePreview && elements.imagePreview.src) {
      thumb.src = elements.imagePreview.src;
      const wrap = thumb.closest('.diagnosis-thumb-wrapper');
      if (wrap) wrap.classList.remove('hidden');
    }

    // 2. Render Top 3 Candidate Predictions
    renderCandidatesList();

    // 3. Disclaimer text
    elements.resultDisclaimer.textContent = data.disclaimer || '';

    // 4. View Transitions
    elements.loadingContainer.classList.add('hidden');
    elements.previewContainer.classList.remove('hidden');
    elements.resultsRejected.classList.add('hidden');
    elements.resultsEmpty.classList.add('hidden');
    elements.resultsContent.classList.remove('hidden');
    elements.newScanBtn.classList.remove('hidden');
    if (elements.printAdvisoryBtn) elements.printAdvisoryBtn.classList.remove('hidden');
    if (elements.copyGuidanceBtn) elements.copyGuidanceBtn.classList.remove('hidden');

    // Smooth scroll to results on mobile and tablet devices
    if (window.innerWidth <= 1024) {
      elements.resultsContent.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }

  const REJECTION_BADGES = {
    non_leaf_image: 'No crop leaf detected',
    unsupported_species: 'Unsupported plant species',
    ambiguous_image: 'Image could not be verified',
    low_quality_image: 'Image quality too low',
  };

  /**
   * Renders the non-diagnostic rejection/abstention state for an input that the backend
   * declined to classify. This state exposes no class, candidate, score or treatment.
   */
  function renderInputRejection(err) {
    const category = err && err.category ? err.category : 'ambiguous_image';
    elements.rejectionBadge.textContent = REJECTION_BADGES[category] || 'Image not verified';
    elements.rejectionHeadline.textContent =
      (err && err.message) || "We couldn't verify that this image contains a supported crop leaf.";
    elements.rejectionDetail.textContent =
      (err && err.detailMessage) ||
      'Please upload a close-up photograph of a single tomato, potato or bell pepper leaf.';

    elements.resultsRejected.classList.remove('hidden');
    elements.resultsContent.classList.add('hidden');
    elements.resultsEmpty.classList.add('hidden');
    elements.knowledgeSection.classList.add('hidden');
    elements.knowledgeDetails.textContent = '';
    elements.newScanBtn.classList.remove('hidden');
    if (elements.printAdvisoryBtn) elements.printAdvisoryBtn.classList.add('hidden');
    if (elements.copyGuidanceBtn) elements.copyGuidanceBtn.classList.add('hidden');

    if (window.innerWidth < 900) {
      elements.resultsRejected.scrollIntoView({ behavior: 'smooth', block: 'start' });
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
      const candPercent = formatModelScore(scoreVal, false);
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
      scoreEl.textContent = candPercent;
      headerEl.appendChild(scoreEl);

      itemEl.appendChild(headerEl);

      // Progress bar
      const trackEl = document.createElement('div');
      trackEl.className = 'progress-track';
      trackEl.setAttribute('role', 'progressbar');
      trackEl.setAttribute('aria-valuenow', (scoreVal * 100).toFixed(2));
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

      // Cache active knowledge data for IPM Advisory report generation
      currentActiveKnowledgeData = knowledgeData;

      // Render knowledge content safely into DOM
      renderKnowledgeDetails(knowledgeData, meta, elements.knowledgeDetails);

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

  function createKnowledgeBlock(title, iconText, badgeText, defaultOpen = false) {
    const details = document.createElement('details');
    details.className = 'monograph-accordion';
    if (defaultOpen) {
      details.setAttribute('open', '');
    }

    const summary = document.createElement('summary');
    summary.className = 'monograph-accordion-header';
    summary.setAttribute('tabindex', '0');

    const left = document.createElement('div');
    left.className = 'accordion-left';

    if (iconText) {
      const iconSpan = document.createElement('span');
      iconSpan.className = 'accordion-icon';
      iconSpan.setAttribute('aria-hidden', 'true');
      if (typeof iconText === 'string' && iconText.startsWith('<svg')) {
        iconSpan.innerHTML = iconText;
      } else {
        iconSpan.textContent = iconText;
      }
      left.appendChild(iconSpan);
    }

    const titleEl = document.createElement('h5');
    titleEl.className = 'monograph-accordion-title';
    titleEl.textContent = title;
    left.appendChild(titleEl);
    summary.appendChild(left);

    const right = document.createElement('div');
    right.className = 'accordion-right';

    if (badgeText) {
      const badge = document.createElement('span');
      badge.className = 'badge badge-accordion';
      badge.textContent = badgeText;
      right.appendChild(badge);
    }

    const chevron = document.createElement('span');
    chevron.className = 'accordion-chevron';
    chevron.setAttribute('aria-hidden', 'true');
    chevron.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="6 9 12 15 18 9"/></svg>';
    right.appendChild(chevron);

    summary.appendChild(right);
    details.appendChild(summary);

    const body = document.createElement('div');
    body.className = 'monograph-accordion-body';
    details.appendChild(body);

    const block = details;
    block.body = body;
    const origAppend = block.appendChild.bind(block);
    block.appendChild = function (child) {
      if (child === summary || child === body) {
        return origAppend(child);
      }
      return body.appendChild(child);
    };

    return { details, block, header: summary, body };
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
   * Safely constructs the entire knowledge section in the DOM with progressive disclosure accordions.
   */
  function renderKnowledgeDetails(data, meta, targetContainer = elements.knowledgeDetails) {
    const container = targetContainer || elements.knowledgeDetails;
    if (!container) return;
    container.textContent = ''; // Safe clear

    if (!data || typeof data !== 'object') return;

    const isHealthy = (data.health_status || '').toLowerCase() === 'healthy';

    // ------------------------------------------------------------------------
    // Monograph Controls & Context Bar
    // ------------------------------------------------------------------------
    const controlsBar = document.createElement('div');
    controlsBar.className = 'monograph-controls-bar';

    const statusPill = document.createElement('div');
    statusPill.className = 'monograph-context-pill';
    statusPill.textContent = meta?.isPrimary
      ? `Primary Diagnostic Monograph (${data.crop || 'Crop'} \u2014 ${data.condition || 'Condition'})`
      : `Differential Candidate Monograph (${data.crop || 'Crop'} \u2014 ${data.condition || 'Condition'})`;
    controlsBar.appendChild(statusPill);

    const toggleGroup = document.createElement('div');
    toggleGroup.className = 'monograph-toggle-group';

    const expandAllBtn = document.createElement('button');
    expandAllBtn.type = 'button';
    expandAllBtn.className = 'btn btn-secondary btn-xs';
    expandAllBtn.textContent = 'Expand All';
    expandAllBtn.addEventListener('click', () => {
      container.querySelectorAll('details.monograph-accordion').forEach((d) => d.setAttribute('open', ''));
    });
    toggleGroup.appendChild(expandAllBtn);

    const collapseAllBtn = document.createElement('button');
    collapseAllBtn.type = 'button';
    collapseAllBtn.className = 'btn btn-secondary btn-xs';
    collapseAllBtn.textContent = 'Collapse All';
    collapseAllBtn.addEventListener('click', () => {
      container.querySelectorAll('details.monograph-accordion').forEach((d) => d.removeAttribute('open'));
    });
    toggleGroup.appendChild(collapseAllBtn);

    controlsBar.appendChild(toggleGroup);
    container.appendChild(controlsBar);

    // ------------------------------------------------------------------------
    // 1. Overview & Pathogen Identification
    // ------------------------------------------------------------------------
    const svgOverview = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>';
    const overview = createKnowledgeBlock('Agronomic Overview & Pathogen Biology', svgOverview, 'Verified Taxonomy', true);

    const metaSummaryRow = document.createElement('div');
    metaSummaryRow.style.display = 'flex';
    metaSummaryRow.style.alignItems = 'center';
    metaSummaryRow.style.justifyContent = 'space-between';
    metaSummaryRow.style.flexWrap = 'wrap';
    metaSummaryRow.style.gap = '0.5rem';
    metaSummaryRow.style.marginBottom = '0.75rem';

    const metaTitle = document.createElement('strong');
    metaTitle.className = 'knowledge-overview-title';
    metaTitle.textContent = `${data.crop || 'Crop'} \u2014 ${data.condition || 'Condition'}`;
    metaSummaryRow.appendChild(metaTitle);

    const healthBadge = document.createElement('span');
    healthBadge.className = `badge ${isHealthy ? 'badge-healthy' : 'badge-diseased'}`;
    healthBadge.textContent = isHealthy ? 'Healthy Crop' : 'Diseased Crop';
    metaSummaryRow.appendChild(healthBadge);

    overview.body.appendChild(metaSummaryRow);

    const causalAgentP = document.createElement('p');
    causalAgentP.className = 'knowledge-text';
    const causalLabel = document.createElement('strong');
    causalLabel.textContent = 'Causal Pathogen / State: ';
    causalAgentP.appendChild(causalLabel);

    if (data.causal_agent) {
      const agentSpan = document.createElement('span');
      agentSpan.style.fontStyle = 'italic';
      agentSpan.style.color = 'var(--mineral-ice)';
      agentSpan.textContent = data.causal_agent;
      causalAgentP.appendChild(agentSpan);
    } else {
      const noneSpan = document.createElement('span');
      noneSpan.textContent = 'None (Physiologically Normal Tissue)';
      causalAgentP.appendChild(noneSpan);
    }
    overview.body.appendChild(causalAgentP);

    container.appendChild(overview.details);

    // ------------------------------------------------------------------------
    // 2. Diagnostic Visual Symptoms
    // ------------------------------------------------------------------------
    if (data.symptoms) {
      const svgSymptoms = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>';
      const symBlock = createKnowledgeBlock('Diagnostic Visual Symptoms & Signs', svgSymptoms, 'Clinical Signs', true);

      if (data.symptoms.summary) {
        symBlock.body.appendChild(createText(data.symptoms.summary));
      }

      if (Array.isArray(data.symptoms.foliar_symptoms) && data.symptoms.foliar_symptoms.length > 0) {
        symBlock.body.appendChild(createSubtitle('Foliar Signs & Leaf Lesions'));
        symBlock.body.appendChild(createList(data.symptoms.foliar_symptoms));
      }

      if (Array.isArray(data.symptoms.stem_and_fruit_symptoms) && data.symptoms.stem_and_fruit_symptoms.length > 0) {
        symBlock.body.appendChild(createSubtitle('Stem & Fruit Symptoms'));
        symBlock.body.appendChild(createList(data.symptoms.stem_and_fruit_symptoms));
      }

      if (Array.isArray(data.symptoms.look_alike_conditions) && data.symptoms.look_alike_conditions.length > 0) {
        symBlock.body.appendChild(createSubtitle('Differential Diagnosis & Look-Alikes'));
        symBlock.body.appendChild(createList(data.symptoms.look_alike_conditions));
      }

      container.appendChild(symBlock.details);
    }

    // ------------------------------------------------------------------------
    // 3. Favorable Environmental Conditions & Transmission
    // ------------------------------------------------------------------------
    if (data.favorable_conditions) {
      const svgEnv = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17.5 19H9a7 7 0 1 1 6.71-9h1.79a4.5 4.5 0 1 1 0 9Z"/></svg>';
      const envBlock = createKnowledgeBlock('Favorable Environmental Conditions & Transmission', svgEnv, 'Disease Cycle', false);

      if (data.favorable_conditions.environmental_factors) {
        envBlock.body.appendChild(createSubtitle('Environmental Drivers'));
        envBlock.body.appendChild(createText(data.favorable_conditions.environmental_factors));
      }

      if (
        Array.isArray(data.favorable_conditions.transmission_routes) &&
        data.favorable_conditions.transmission_routes.length > 0
      ) {
        envBlock.body.appendChild(createSubtitle('Vectors & Transmission Pathways'));
        envBlock.body.appendChild(createList(data.favorable_conditions.transmission_routes));
      }

      container.appendChild(envBlock.details);
    }

    // ------------------------------------------------------------------------
    // 4. Cultural Prevention & Field Hygiene
    // ------------------------------------------------------------------------
    if (data.prevention_and_cultural) {
      const svgPrev = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>';
      const prevBlock = createKnowledgeBlock('Cultural Prevention & Agronomic Practices', svgPrev, 'Field Hygiene', false);

      if (Array.isArray(data.prevention_and_cultural.sanitation) && data.prevention_and_cultural.sanitation.length > 0) {
        prevBlock.body.appendChild(createSubtitle('Field Sanitation & Hygiene'));
        prevBlock.body.appendChild(createList(data.prevention_and_cultural.sanitation));
      }

      if (data.prevention_and_cultural.crop_rotation) {
        prevBlock.body.appendChild(createSubtitle('Crop Rotation Schedule'));
        prevBlock.body.appendChild(createText(data.prevention_and_cultural.crop_rotation));
      }

      if (
        Array.isArray(data.prevention_and_cultural.irrigation_and_airflow) &&
        data.prevention_and_cultural.irrigation_and_airflow.length > 0
      ) {
        prevBlock.body.appendChild(createSubtitle('Canopy Airflow & Irrigation Management'));
        prevBlock.body.appendChild(createList(data.prevention_and_cultural.irrigation_and_airflow));
      }

      if (
        Array.isArray(data.prevention_and_cultural.resistant_cultivars) &&
        data.prevention_and_cultural.resistant_cultivars.length > 0
      ) {
        prevBlock.body.appendChild(createSubtitle('Genetic Resistance & Variety Selection'));
        prevBlock.body.appendChild(createList(data.prevention_and_cultural.resistant_cultivars));
      }

      container.appendChild(prevBlock.details);
    }

    // ------------------------------------------------------------------------
    // 5. Treatment / Integrated Pest Management (Strict Healthy vs Diseased)
    // ------------------------------------------------------------------------
    if (data.treatment) {
      const treatmentTitle = isHealthy
        ? 'Crop Health Maintenance (No Treatment Required)'
        : 'Integrated Pest & Disease Management (IPM)';
      const treatmentIcon = isHealthy
        ? '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a10 10 0 0 1 10 10c0 5.5-4.5 10-10 10S2 17.5 2 12A10 10 0 0 1 12 2z"/><path d="M12 6v12"/></svg>'
        : '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m10.5 20.5 10-10a4.95 4.95 0 1 0-7-7l-10 10a4.95 4.95 0 1 0 7 7Z"/><path d="m8.5 8.5 7 7"/></svg>';
      const treatmentBadge = isHealthy ? 'Routine Care' : 'Action Protocol';

      const treatBlock = createKnowledgeBlock(treatmentTitle, treatmentIcon, treatmentBadge, false);

      if (data.treatment.urgency) {
        const calloutType = isHealthy ? 'healthy' : 'warning';
        treatBlock.body.appendChild(createCallout(data.treatment.urgency, calloutType));
      }

      if (Array.isArray(data.treatment.non_chemical_controls) && data.treatment.non_chemical_controls.length > 0) {
        treatBlock.body.appendChild(
          createSubtitle(isHealthy ? 'Routine Non-Chemical Maintenance' : 'Non-Chemical & Cultural Controls')
        );
        treatBlock.body.appendChild(createList(data.treatment.non_chemical_controls));
      }

      if (Array.isArray(data.treatment.ipm_chemical_principles) && data.treatment.ipm_chemical_principles.length > 0) {
        treatBlock.body.appendChild(
          createSubtitle(isHealthy ? 'Chemical Application Guidance' : 'IPM Chemical Management Principles')
        );
        treatBlock.body.appendChild(createList(data.treatment.ipm_chemical_principles));
      }

      container.appendChild(treatBlock.details);

      // ----------------------------------------------------------------------
      // 6. Safety & Regulatory Constraints (Separated into its own section)
      // ----------------------------------------------------------------------
      if (
        Array.isArray(data.treatment.safety_and_regulatory_constraints) &&
        data.treatment.safety_and_regulatory_constraints.length > 0
      ) {
        const svgSafety = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>';
        const safetyBlock = createKnowledgeBlock(
          'Safety, PHI & Regulatory Constraints',
          svgSafety,
          'Mandatory Safeguards',
          false
        );
        safetyBlock.body.appendChild(
          createSubtitle('Pesticide Application, PHI & Environmental Safeguards')
        );
        safetyBlock.body.appendChild(createList(data.treatment.safety_and_regulatory_constraints));
        container.appendChild(safetyBlock.details);
      }
    }

    // ------------------------------------------------------------------------
    // 7. Agricultural Extension Consultation
    // ------------------------------------------------------------------------
    if (data.extension_consultation) {
      const svgExt = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"/></svg>';
      const extBlock = createKnowledgeBlock('Agricultural Extension & Specialist Referral', svgExt, 'Extension Referral', false);

      if (data.extension_consultation.local_specialist) {
        extBlock.body.appendChild(createSubtitle('Recommended Extension Authority'));
        extBlock.body.appendChild(createText(data.extension_consultation.local_specialist));
      }

      if (
        Array.isArray(data.extension_consultation.escalation_triggers) &&
        data.extension_consultation.escalation_triggers.length > 0
      ) {
        extBlock.body.appendChild(createSubtitle('Field Escalation Triggers (Inspection Required)'));
        extBlock.body.appendChild(createList(data.extension_consultation.escalation_triggers));
      }

      container.appendChild(extBlock.details);
    }

    // ------------------------------------------------------------------------
    // 8. Scientific & Extension References (Untrusted Data & Safe URLs)
    // ------------------------------------------------------------------------
    if (Array.isArray(data.sources) && data.sources.length > 0) {
      const svgSources = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>';
      const sourcesBlock = createKnowledgeBlock(
        'Scientific Sources & Extension Citations',
        svgSources,
        `${data.sources.length} Verified Sources`,
        false
      );

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

      sourcesBlock.body.appendChild(sourceList);
      container.appendChild(sourcesBlock.details);
    }

    // ------------------------------------------------------------------------
    // 9. Diagnostic Uncertainty & Field Verification Notice
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
  }

  // ==========================================================================
  // Dedicated Laboratory Diagnostic Print Report Builder
  // ==========================================================================

  function formatModelScore(score, isHtml = true) {
    if (typeof score !== 'number' || isNaN(score)) return 'Not recorded';
    const pct = score * 100;
    if (pct === 0) return '0.00%';
    if (pct > 0 && pct < 0.01) {
      return isHtml ? '&lt; 0.01%' : '< 0.01%';
    }
    return `${pct.toFixed(2)}%`;
  }

  function buildPrintReport(prediction, knowledgeData, candidateMeta) {
    const container = elements.printReportContainer || document.getElementById('print-report-container');
    if (!container || !prediction) {
      console.warn('PhytoReport buildPrintReport skipped:', { container: !!container, prediction: !!prediction });
      return;
    }

    const now = new Date();
    const dateFormatted = now.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' });
    const timeFormatted = now.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });
    const isoDateStr = now.toISOString().slice(0, 10).replace(/-/g, '');
    const randSuffix = Math.random().toString(36).substring(2, 6).toUpperCase();
    const reportRef = `PHY-${isoDateStr}-${randSuffix}`;

    // Diagnostic target evaluation (Candidate exploration vs primary prediction)
    const isDiffExploration = candidateMeta && !candidateMeta.isPrimary;
    const targetCrop = isDiffExploration ? candidateMeta.crop : prediction.crop;
    const targetCondition = isDiffExploration ? candidateMeta.condition : prediction.condition;
    const targetScore = isDiffExploration ? candidateMeta.score : prediction.score;
    const targetRank = isDiffExploration ? candidateMeta.rank : 1;
    const primaryCrop = prediction.crop;
    const primaryCondition = prediction.condition;
    const primaryScore = prediction.score;

    // Health state
    const kd = knowledgeData;
    const targetHealth = (kd && kd.health_status) ? kd.health_status : (prediction.health || 'Diseased');
    const isHealthy = targetHealth.toLowerCase() === 'healthy';

    // Specimen image and metadata (actual uploaded specimen values)
    const specimenSrc = (elements.imagePreview && elements.imagePreview.src) ? elements.imagePreview.src : '';
    const filename = (currentFile && currentFile.name)
      ? currentFile.name
      : ((elements.metaFilename && elements.metaFilename.textContent !== '\u2014')
          ? elements.metaFilename.textContent
          : 'Not recorded');
    const dimensions = (elements.metaDimensions && elements.metaDimensions.textContent !== '\u2014' && elements.metaDimensions.textContent !== 'Calculating...' && elements.metaDimensions.textContent !== 'Unknown (failed to decode)')
      ? elements.metaDimensions.textContent
      : 'Not recorded';
    const filesize = (currentFile && currentFile.size)
      ? formatFileSize(currentFile.size)
      : ((elements.metaFilesize && elements.metaFilesize.textContent !== '\u2014')
          ? elements.metaFilesize.textContent
          : 'Not recorded');

    // Runtime Gate Verification & Pipeline Latency
    // Layer A strictly reflects backend physical integrity verification metadata
    let layerAStatus = 'Verification status unavailable';
    if (prediction.timing && typeof prediction.timing.decode_ms === 'number') {
      layerAStatus = `Passed (${prediction.timing.decode_ms.toFixed(1)} ms)`;
    }

    const clientPrecheck = (currentFile && currentFile.size > 0)
      ? 'Passed (Client decode & format validated)'
      : 'Not recorded';

    let layerBStatus = 'Verification status unavailable';
    if (prediction.timing && typeof prediction.timing.gate_ms === 'number') {
      if (prediction.timing.gate_ms > 0) {
        if (lastReadyState && lastReadyState.gateLoaded === true) {
          layerBStatus = `Learned Gate Passed (${prediction.timing.gate_ms.toFixed(1)} ms)`;
        } else if (lastReadyState && lastReadyState.gateLoaded === false) {
          layerBStatus = `Heuristic Screen Passed (${prediction.timing.gate_ms.toFixed(1)} ms)`;
        } else {
          layerBStatus = `Passed (${prediction.timing.gate_ms.toFixed(1)} ms)`;
        }
      } else {
        layerBStatus = 'Disabled / Bypassed';
      }
    }

    let latencyMs = 'Not recorded';
    if (prediction.timing && typeof prediction.timing.total_ms === 'number') {
      latencyMs = `${prediction.timing.total_ms.toFixed(1)} ms`;
    }

    // Top-3 differential candidate table rows
    let candidatesTableRows = '';
    if (Array.isArray(prediction.top_predictions)) {
      prediction.top_predictions.forEach((cand) => {
        const isCurrentActive = (isDiffExploration && cand.rank === targetRank) || (!isDiffExploration && cand.rank === 1);
        const candPercent = formatModelScore(cand.score);
        const candBarWidth = Math.min(100, Math.max(0, cand.score * 100)).toFixed(1);
        const rowClass = isCurrentActive ? 'active-row' : '';
        const activeLabel = isCurrentActive ? ' <span style="font-size:7pt; color:#14382A; font-weight:700;">[SELECTED]</span>' : '';

        candidatesTableRows += `
          <tr class="${rowClass}">
            <td style="text-align:center; font-family:'JetBrains Mono', monospace; font-weight:700;">#${cand.rank}</td>
            <td><strong>${escapeHtml(cand.crop)}</strong> &mdash; ${escapeHtml(cand.condition)}${activeLabel}</td>
            <td style="text-align:center;"><span style="font-size:7.5pt; font-weight:600; color:${cand.health === 'Healthy' ? '#15803D' : '#B91C1C'};">${escapeHtml(cand.health)}</span></td>
            <td style="text-align:right; font-family:'JetBrains Mono', monospace; font-weight:700;">${candPercent}</td>
            <td style="width:25%;">
              <div class="print-bar-track">
                <div class="print-bar-fill ${cand.rank === 1 ? '' : 'secondary'}" style="width:${candBarWidth}%;"></div>
              </div>
            </td>
          </tr>
        `;
      });
    }

    // Cultural Controls List
    let culturalListHtml = '';
    const culturalItems = (kd && kd.treatment && Array.isArray(kd.treatment.non_chemical_controls) && kd.treatment.non_chemical_controls.length > 0)
      ? kd.treatment.non_chemical_controls
      : (kd && kd.prevention_and_cultural && Array.isArray(kd.prevention_and_cultural.sanitation) ? kd.prevention_and_cultural.sanitation : []);
    if (culturalItems.length > 0) {
      culturalListHtml = `<ul class="print-bullet-list">${culturalItems.map(item => `<li>${escapeHtml(item)}</li>`).join('')}</ul>`;
    } else if (isHealthy) {
      culturalListHtml = `<p class="print-prose">Maintain balanced irrigation, soil nutrition, and routine crop surveillance. (Specific cultural guidelines not available in the current knowledge base.)</p>`;
    } else {
      culturalListHtml = `<p class="print-prose">Not available in the current knowledge base.</p>`;
    }

    // Chemical IPM Principles List
    let chemicalListHtml = '';
    const chemItems = (kd && kd.treatment && Array.isArray(kd.treatment.ipm_chemical_principles) && kd.treatment.ipm_chemical_principles.length > 0)
      ? kd.treatment.ipm_chemical_principles
      : [];
    if (chemItems.length > 0) {
      chemicalListHtml = `<ul class="print-bullet-list">${chemItems.map(item => `<li>${escapeHtml(item)}</li>`).join('')}</ul>`;
    } else if (isHealthy) {
      chemicalListHtml = `<p class="print-prose">Zero synthetic pesticide or bactericide application indicated for healthy foliage. Prophylactic spraying violates IPM stewardship.</p>`;
    } else {
      chemicalListHtml = `<p class="print-prose">Not available in the current knowledge base.</p>`;
    }

    // Safety & Regulatory List (clearly distinguished as General Safety Guidance if not specific in KB)
    let safetyListHtml = '';
    const safetyItems = (kd && kd.treatment && Array.isArray(kd.treatment.safety_and_regulatory_constraints) && kd.treatment.safety_and_regulatory_constraints.length > 0)
      ? kd.treatment.safety_and_regulatory_constraints
      : [];
    if (safetyItems.length > 0) {
      safetyListHtml = `<ul class="print-bullet-list">${safetyItems.map(item => `<li>${escapeHtml(item)}</li>`).join('')}</ul>`;
    } else {
      safetyListHtml = `
        <p class="print-prose" style="font-size:7.5pt; color:#4B5563;">
          <strong>General Safety Guidance:</strong> Specific regulatory constraints not available in the current knowledge base. Strictly observe official product labels, worker Re-Entry Intervals (REI), Pre-Harvest Intervals (PHI), and pollinator safeguards in accordance with local pesticide regulatory authorities.
        </p>`;
    }

    // Symptoms summary & list
    let symptomsSummary = (kd && kd.symptoms && kd.symptoms.summary)
      ? kd.symptoms.summary
      : 'Not available in the current knowledge base.';
    let foliarSymptomsHtml = '';
    if (kd && kd.symptoms && Array.isArray(kd.symptoms.foliar_symptoms) && kd.symptoms.foliar_symptoms.length > 0) {
      foliarSymptomsHtml = `<ul class="print-bullet-list">${kd.symptoms.foliar_symptoms.map(s => `<li>${escapeHtml(s)}</li>`).join('')}</ul>`;
    }

    // Favorable conditions & transmission
    let envFactors = (kd && kd.favorable_conditions && kd.favorable_conditions.environmental_factors)
      ? kd.favorable_conditions.environmental_factors
      : 'Not available in the current knowledge base.';
    let transmissionHtml = '';
    if (kd && kd.favorable_conditions && Array.isArray(kd.favorable_conditions.transmission_routes) && kd.favorable_conditions.transmission_routes.length > 0) {
      transmissionHtml = `<ul class="print-bullet-list">${kd.favorable_conditions.transmission_routes.map(r => `<li>${escapeHtml(r)}</li>`).join('')}</ul>`;
    }

    // Extension authority & triggers
    let extensionSpecialist = (kd && kd.extension_consultation && kd.extension_consultation.local_specialist)
      ? kd.extension_consultation.local_specialist
      : 'Not available in the current knowledge base.';
    let triggersHtml = '';
    if (kd && kd.extension_consultation && Array.isArray(kd.extension_consultation.escalation_triggers) && kd.extension_consultation.escalation_triggers.length > 0) {
      triggersHtml = `<ul class="print-bullet-list">${kd.extension_consultation.escalation_triggers.map(t => `<li>${escapeHtml(t)}</li>`).join('')}</ul>`;
    }

    // Causal agent text
    const causalAgentText = (kd && kd.causal_agent)
      ? kd.causal_agent
      : (isHealthy ? 'None (Physiologically Normal Tissue)' : 'Not available in the current knowledge base.');

    // Urgency text
    const urgencyText = (kd && kd.treatment && kd.treatment.urgency)
      ? kd.treatment.urgency
      : (isHealthy
          ? 'ROUTINE SURVEILLANCE: Normal crop health maintained. No therapeutic intervention required.'
          : 'Not available in the current knowledge base.');

    // Build complete laboratory report HTML
    container.innerHTML = `
      <!-- ================================================================= -->
      <!-- PAGE 1: DIAGNOSTIC SUMMARY & FOLIAR PATHOLOGY CHARACTERIZATION     -->
      <!-- ================================================================= -->
      <article class="print-page">
        <div>
          <!-- Header -->
          <header class="print-header">
            <div class="print-header-top">
              <div class="print-brand">
                <div class="print-brand-symbol">
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M12 2a10 10 0 0 1 10 10c0 5.5-4.5 10-10 10S2 17.5 2 12A10 10 0 0 1 12 2z"/>
                    <path d="M12 6v12"/>
                    <path d="M8 10l4-4 4 4"/>
                  </svg>
                </div>
                <div>
                  <div class="print-brand-title"><span class="phyto">PHYTO</span><span class="signal">SIGNAL</span></div>
                  <div class="print-brand-sub">Clinical Foliar Pathology Laboratory // Diagnostic Intelligence</div>
                </div>
              </div>
              <div class="print-meta-box">
                <div class="print-ref-pill">REF: ${reportRef}</div>
                <div>Date & Time: <strong>${dateFormatted}, ${timeFormatted}</strong></div>
                <div>Model Backbone: <strong>MobileNetV2-Balanced v1.0.0</strong></div>
              </div>
            </div>
          </header>

          <!-- Report Title Banner -->
          <div class="print-report-title-banner">
            <h1 class="print-report-h1">${isDiffExploration ? 'Differential Monograph &amp; Look-Alike Reference' : 'Foliar Pathology Diagnostic Report'}</h1>
            <span class="print-report-sub">${isDiffExploration ? 'Exploratory Botanical Reference &bull; Model Primary Diagnosis Unaltered' : 'Solanaceae Screening Protocol &bull; Closed-World Diagnostic Distribution'}</span>
          </div>

          <!-- Specimen & Primary Diagnosis 2-Column Grid -->
          <div class="print-grid-2col">
            <!-- Left: Specimen Ingestion -->
            <div class="print-card">
              <div class="print-card-title">Verified Specimen Ingestion</div>
              <div class="print-specimen-frame">
                ${specimenSrc ? `<img id="print-specimen-img" class="print-specimen-img" src="${specimenSrc}" alt="Verified Specimen Photo">` : '<span style="font-size:8pt; color:#9CA3AF;">No Specimen Photo Loaded</span>'}
              </div>
              <table class="print-meta-table">
                <tr><td>Specimen File:</td><td>${escapeHtml(filename)}</td></tr>
                <tr><td>Spatial Resolution:</td><td>${escapeHtml(dimensions)}</td></tr>
                <tr><td>Payload Size:</td><td>${escapeHtml(filesize)}</td></tr>
                <tr><td>Client Upload Pre-Check:</td><td>${escapeHtml(clientPrecheck)}</td></tr>
                <tr><td>Physical Integrity (Layer A):</td><td>${escapeHtml(layerAStatus)}</td></tr>
                <tr><td>Leaf Verification (Layer B):</td><td>${escapeHtml(layerBStatus)}</td></tr>
              </table>
            </div>

            <!-- Right: Diagnostic Evaluation -->
            <div class="print-card">
              <div class="print-card-title">Diagnostic Evaluation & Assessment</div>
              <div class="print-diagnosis-box">
                <div class="print-crop-kicker">Botanical Host: ${escapeHtml(targetCrop)}</div>
                <h2 class="print-condition-title">${escapeHtml(targetCondition)}</h2>
                <div class="print-metrics-row">
                  <div class="print-metric-chip">
                    <div class="print-metric-label">Model Score</div>
                    <div class="print-metric-val">${formatModelScore(targetScore)}</div>
                  </div>
                  <div class="print-metric-chip">
                    <div class="print-metric-label">Health State</div>
                    <div class="print-metric-val ${isHealthy ? 'healthy' : 'diseased'}">${escapeHtml(targetHealth.toUpperCase())}</div>
                  </div>
                  <div class="print-metric-chip">
                    <div class="print-metric-label">Evaluation Mode</div>
                    <div class="print-metric-val" style="font-size:7pt;">${isDiffExploration ? `EXPLORATORY (#${targetRank})` : 'PRIMARY (#1)'}</div>
                  </div>
                </div>

                ${isDiffExploration ? `
                  <div class="print-diff-notice">
                    <div style="font-weight:700; margin-bottom:2pt; text-transform:uppercase; letter-spacing:0.04em;">
                      ⚠ Exploratory Reference Review &mdash; Not Confirmed Diagnosis
                    </div>
                    <div>
                      The model's confirmed primary prediction for this specimen remains <strong>${escapeHtml(primaryCrop)} &mdash; ${escapeHtml(primaryCondition)} (${formatModelScore(primaryScore)})</strong>.
                      Selecting alternative candidate #${targetRank} (${escapeHtml(targetCrop)} &mdash; ${escapeHtml(targetCondition)}) does <em>not</em> alter or override the primary prediction.
                      ${(targetScore * 100 < 0.01) ? `
                        The relative model score for this candidate is <strong>${formatModelScore(targetScore)}</strong>, which represents negligible neural alignment and provides <strong>no diagnostic support</strong>. This monograph is presented solely for educational comparison and look-alike reference.
                      ` : `
                        This monograph is presented solely for educational comparison and differential look-alike reference.
                      `}
                    </div>
                  </div>
                ` : `
                  <div style="font-size:7pt; color:#4B5563; margin-top:6pt;">
                    Closed-world relative softmax score across 15 trained Solanaceae categories. Softmax scores represent relative model output distributions, not calibrated disease probabilities or diagnostic certainty.
                  </div>
                `}
              </div>

              <!-- Ingestion Gate Summary -->
              <div class="print-gate-strip">
                <span class="print-gate-item"><strong>Layer A (Physical):</strong> ${escapeHtml(layerAStatus)}</span>
                <span class="print-gate-item"><strong>Layer B (Leaf Gate):</strong> ${escapeHtml(layerBStatus)}</span>
                <span class="print-gate-item"><strong>Pipeline Latency:</strong> ${escapeHtml(latencyMs)}</span>
              </div>
            </div>
          </div>

          <!-- Top-3 Differential Candidates Table -->
          <div class="print-table-container">
            <div class="print-card-title">Top-3 Candidate Output Classes (Relative Softmax Score Distribution)</div>
            <table class="print-candidates-table">
              <thead>
                <tr>
                  <th style="width:7%; text-align:center;">Rank</th>
                  <th style="width:40%;">Taxon & Condition</th>
                  <th style="width:16%; text-align:center;">Health State</th>
                  <th style="width:17%; text-align:right;">Model Score</th>
                  <th style="width:20%;">Relative Softmax Distribution</th>
                </tr>
              </thead>
              <tbody>
                ${candidatesTableRows}
              </tbody>
            </table>
          </div>
        </div>

        <!-- Page 1 Footer -->
        <footer class="print-footer">
          <span class="print-footer-brand">PHYTOSIGNAL Clinical Foliar Intelligence</span>
          <span>Confidential Agronomic Diagnostic Document &bull; Ref: ${reportRef}</span>
          <span>Page 1 &bull; Diagnostic Summary</span>
        </footer>
      </article>

      <!-- ================================================================= -->
      <!-- PAGE 2: AGRONOMIC CLINICAL ADVISORY & IPM MANAGEMENT PROTOCOLS     -->
      <!-- ================================================================= -->
      <article class="print-page print-page-break">
        <div>
          <!-- Header -->
          <header class="print-header">
            <div class="print-header-top">
              <div class="print-brand">
                <div class="print-brand-symbol">
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M12 2a10 10 0 0 1 10 10c0 5.5-4.5 10-10 10S2 17.5 2 12A10 10 0 0 1 12 2z"/>
                    <path d="M12 6v12"/>
                    <path d="M8 10l4-4 4 4"/>
                  </svg>
                </div>
                <div>
                  <div class="print-brand-title"><span class="phyto">PHYTO</span><span class="signal">SIGNAL</span></div>
                  <div class="print-brand-sub">Clinical Agronomic Advisory // Treatment & Management Protocols</div>
                </div>
              </div>
              <div class="print-meta-box">
                <div class="print-ref-pill">REF: ${reportRef}</div>
                <div>Target: <strong>${escapeHtml(targetCrop)} &mdash; ${escapeHtml(targetCondition)}</strong></div>
                <div>Scope: <strong>Integrated Pest Management (IPM)</strong></div>
              </div>
            </div>
          </header>

          <!-- Target Diagnostic Banner -->
          <div class="print-target-banner">
            <h3>${isDiffExploration ? 'Exploratory Target Reference:' : 'Target Pathology:'} ${escapeHtml(targetCrop)} &mdash; ${escapeHtml(targetCondition)}</h3>
            <p><strong>Causal Pathogen / Biotic Factor:</strong> <em>${escapeHtml(causalAgentText)}</em></p>
          </div>

          ${isDiffExploration ? `
            <div class="print-diff-notice" style="margin-bottom:8pt;">
              <strong>Exploratory Reference Monograph (Not Confirmed Diagnosis):</strong> Protocols below correspond to alternative candidate #${targetRank} (${escapeHtml(targetCrop)} ${escapeHtml(targetCondition)}). The primary model diagnostic output remains <strong>${escapeHtml(primaryCrop)} &mdash; ${escapeHtml(primaryCondition)} (Model Score: ${formatModelScore(primaryScore)})</strong> and is unaltered. ${(targetScore * 100 < 0.01) ? 'The near-zero model score (' + formatModelScore(targetScore) + ') indicates negligible statistical probability and provides no diagnostic support.' : ''}
            </div>
          ` : ''}

          <!-- Section 1: Pathogen Profile & Foliar Morphology -->
          <section class="print-section">
            <h4 class="print-section-h4">
              <span>1. Pathogen Biology & Diagnostic Foliar Signs</span>
              <span style="font-size:7pt; color:#6B7280; text-transform:none;">Morphological Identification</span>
            </h4>
            <p class="print-prose">${escapeHtml(symptomsSummary)}</p>
            ${foliarSymptomsHtml}
            <p class="print-prose" style="margin-top:4pt;"><strong>Favorable Environmental Conditions:</strong> ${escapeHtml(envFactors)}</p>
            ${transmissionHtml}
          </section>

          <!-- Section 2: Integrated Pest Management (IPM) Protocols -->
          <section class="print-section">
            <h4 class="print-section-h4">
              <span>2. Integrated Pest Management (IPM) & Crop Care</span>
              <span style="font-size:7pt; color:#6B7280; text-transform:none;">${isHealthy ? 'Routine Maintenance' : 'Intervention Protocol'}</span>
            </h4>
            <div class="print-urgency-box ${isHealthy ? 'healthy' : ''}">
              ${escapeHtml(urgencyText)}
            </div>
            <p class="print-prose"><strong>Cultural Controls & Field Sanitation:</strong></p>
            ${culturalListHtml}
            <p class="print-prose" style="margin-top:4pt;"><strong>Chemical Application Principles & Active Ingredient Rotation:</strong></p>
            ${chemicalListHtml}
            <p class="print-prose" style="margin-top:4pt;"><strong>Mandatory Safety, PHI & Environmental Safeguards:</strong></p>
            ${safetyListHtml}
          </section>

          <!-- Section 3: Extension Consultation & Escalation -->
          <section class="print-section">
            <h4 class="print-section-h4">
              <span>3. Agricultural Extension Authority & Verification Triggers</span>
              <span style="font-size:7pt; color:#6B7280; text-transform:none;">Specialist Escalation</span>
            </h4>
            <p class="print-prose"><strong>Recommended Extension Authority:</strong> ${escapeHtml(extensionSpecialist)}</p>
            ${triggersHtml ? `<p class="print-prose" style="margin-top:3pt;"><strong>Field Escalation Triggers:</strong></p>${triggersHtml}` : ''}
          </section>

          <!-- Section 4: Responsible AI & Regulatory Disclaimer -->
          <div class="print-disclaimer-box">
            <strong>Responsible AI & Regulatory Compliance Disclaimer:</strong>
            ${escapeHtml(prediction.disclaimer || 'This automated advisory report is generated by a closed-world deep learning computer vision model (MobileNetV2). Classification outputs represent relative probability distributions across 15 trained classes and do not substitute for official on-site laboratory culture assays or professional extension verification. Always consult certified agronomic extension officers before purchasing or applying regulated chemical crop protection products.')}
          </div>
        </div>

        <!-- Page 2 Footer -->
        <footer class="print-footer">
          <span class="print-footer-brand">PHYTOSIGNAL Clinical Foliar Intelligence</span>
          <span>Confidential Agronomic Diagnostic Document &bull; Ref: ${reportRef}</span>
          <span>Page 2 &bull; End of Monograph</span>
        </footer>
      </article>
    `;
  }

  /**
   * Prepares and triggers the dedicated laboratory print report.
   * Ensures the specimen photograph is decoded before opening the print dialog.
   */
  async function prepareAndTriggerPrint() {
    if (!lastPrediction) {
      showAlert('No diagnostic results available to print. Please run an analysis first.', 'Notice', 'info');
      return;
    }

    try {
      // 1. Build dedicated report DOM
      buildPrintReport(lastPrediction, currentActiveKnowledgeData, activeCandidateMeta);

      // 2. Wait for specimen image to decode
      const printImg = document.getElementById('print-specimen-img');
      if (printImg && printImg.src && !printImg.complete) {
        await Promise.race([
          printImg.decode(),
          new Promise((resolve) => setTimeout(resolve, 800)),
        ]);
      }

      // 3. Small timeout for layout pass before triggering print
      setTimeout(() => {
        window.print();
      }, 100);
    } catch (err) {
      // Fallback: trigger print even if decode times out
      window.print();
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
    if (elements.rejectionActionBtn) {
      elements.rejectionActionBtn.addEventListener('click', resetWorkflow);
    }

    // Knowledge Retry action
    elements.knowledgeRetryBtn.addEventListener('click', () => {
      if (activeCandidateClassId !== null && activeCandidateMeta) {
        fetchAndRenderKnowledge(activeCandidateClassId, activeCandidateMeta);
      }
    });

    // Alert dismiss
    elements.alertDismissBtn.addEventListener('click', hideAlert);

    // Camera capture handler
    if (elements.cameraBtn && elements.cameraInput) {
      elements.cameraBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        elements.cameraInput.click();
      });
      elements.cameraInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
          handleSelectedFile(e.target.files[0]);
        }
      });
    }

    // Print advisory handler (Dedicated Professional Laboratory Report)
    if (elements.printAdvisoryBtn) {
      elements.printAdvisoryBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        prepareAndTriggerPrint();
      });
    }

    // Clear history handler
    if (elements.clearHistoryBtn) {
      elements.clearHistoryBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        clearRecentScans();
      });
    }

    // Copy IPM Advisory handler
    if (elements.copyGuidanceBtn) {
      elements.copyGuidanceBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        copyAdvisoryToClipboard();
      });
    }

    // Quick sample leaf specimen buttons
    const sampleButtons = document.querySelectorAll('.btn-sample');
    sampleButtons.forEach((btn) => {
      btn.addEventListener('click', async (e) => {
        e.stopPropagation();
        const sampleFileName = btn.getAttribute('data-sample');
        if (!sampleFileName) return;

        try {
          const resp = await fetch('samples/' + sampleFileName);
          if (!resp.ok) {
            throw new Error(`HTTP ${resp.status}: Could not load ${sampleFileName}`);
          }
          const blob = await resp.blob();
          const file = new File([blob], sampleFileName, { type: 'image/jpeg' });
          handleSelectedFile(file);
        } catch (err) {
          showAlert('Failed to load sample image: ' + err.message, 'Sample Load Error', 'error');
        }
      });
    });

    // Prevent page drop defaults
    window.addEventListener('dragover', (e) => e.preventDefault());
    window.addEventListener('drop', (e) => e.preventDefault());
  }

  // ==========================================================================
  // Interactive Toast & Advisory Clipboard Operations
  // ==========================================================================

  function showToast(message, icon = '✓') {
    if (!elements.toast) return;
    if (elements.toastMessage) elements.toastMessage.textContent = message;
    if (elements.toastIcon) elements.toastIcon.textContent = icon;
    elements.toast.classList.remove('hidden');

    clearTimeout(toastTimeout);
    toastTimeout = setTimeout(() => {
      elements.toast.classList.add('hidden');
    }, 3200);
  }

  function copyAdvisoryToClipboard() {
    if (!lastPrediction) {
      showAlert('No diagnostic results available to copy.', 'Notice', 'info');
      return;
    }

    const lines = [
      '=================================================================',
      'AI-BASED CROP DISEASE DIAGNOSTIC & IPM ADVISORY REPORT',
      '=================================================================',
      `Generated: ${new Date().toLocaleString()}`,
      `Host Crop: ${lastPrediction.crop || 'Unknown Crop'}`,
      `Predicted Condition: ${lastPrediction.condition || 'Unknown Condition'}`,
      `Health Status: ${lastPrediction.health || 'Unknown'}`,
      `Classification Score: ${formatModelScore(lastPrediction.score)}`,
      '',
      'TOP CANDIDATES DISTRIBUTION:',
    ];

    if (Array.isArray(lastPrediction.top_predictions)) {
      lastPrediction.top_predictions.forEach((cand, idx) => {
        const scorePercent = formatModelScore(cand.score);
        lines.push(`  ${idx + 1}. [${cand.crop}] ${cand.condition}: ${scorePercent} (${cand.health})`);
      });
    }

    if (currentActiveKnowledgeData) {
      const kd = currentActiveKnowledgeData;
      lines.push('');
      lines.push('-----------------------------------------------------------------');
      lines.push('AGRONOMIC MONOGRAPH & INTEGRATED PEST MANAGEMENT (IPM)');
      lines.push('-----------------------------------------------------------------');

      if (kd.causal_agent) {
        lines.push(`Causal Pathogen: ${kd.causal_agent}`);
      }
      if (kd.symptoms) {
        if (kd.symptoms.summary) lines.push(`Symptom Overview: ${kd.symptoms.summary}`);
        if (Array.isArray(kd.symptoms.foliar_symptoms) && kd.symptoms.foliar_symptoms.length > 0) {
          lines.push('Foliar Signs:');
          kd.symptoms.foliar_symptoms.forEach((s) => lines.push(`  - ${s}`));
        }
      }
      if (kd.favorable_conditions && kd.favorable_conditions.environmental_factors) {
        lines.push(`Environmental Transmission Factors: ${kd.favorable_conditions.environmental_factors}`);
      }
      if (kd.treatment) {
        if (kd.treatment.urgency) lines.push(`Treatment Urgency: ${kd.treatment.urgency}`);
        if (Array.isArray(kd.treatment.non_chemical_controls) && kd.treatment.non_chemical_controls.length > 0) {
          lines.push('Non-Chemical & Cultural Controls:');
          kd.treatment.non_chemical_controls.forEach((c) => lines.push(`  - ${c}`));
        }
        if (Array.isArray(kd.treatment.ipm_chemical_principles) && kd.treatment.ipm_chemical_principles.length > 0) {
          lines.push('IPM Chemical Application Principles:');
          kd.treatment.ipm_chemical_principles.forEach((p) => lines.push(`  - ${p}`));
        }
      }
      if (kd.extension_consultation && kd.extension_consultation.local_specialist) {
        lines.push(`Recommended Extension Authority: ${kd.extension_consultation.local_specialist}`);
      }
    }

    lines.push('');
    lines.push('ADVISORY DISCLAIMER:');
    lines.push(
      lastPrediction.disclaimer ||
        'Closed-world neural classification screening. Confirm foliar diagnoses with official agricultural extension personnel before executing pesticide treatments.'
    );
    lines.push('=================================================================');

    const fullReport = lines.join('\n');

    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard
        .writeText(fullReport)
        .then(() => {
          showToast('IPM Advisory copied to clipboard!', '📋');
        })
        .catch(() => {
          fallbackCopyText(fullReport);
        });
    } else {
      fallbackCopyText(fullReport);
    }
  }

  function fallbackCopyText(text) {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.focus();
    ta.select();
    try {
      document.execCommand('copy');
      showToast('IPM Advisory copied to clipboard!', '📋');
    } catch {
      showAlert('Could not access clipboard automatically. Please use Print to export.', 'Notice', 'info');
    }
    document.body.removeChild(ta);
  }

  // ==========================================================================
  // Recent Scan History (localStorage Persistence)
  // ==========================================================================
  const RECENT_SCANS_STORAGE_KEY = 'phytosignal_recent_scans';
  const MAX_RECENT_SCANS = 6;

  function getRecentScans() {
    try {
      const raw = localStorage.getItem(RECENT_SCANS_STORAGE_KEY);
      if (!raw) return [];
      const parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  }

  /**
   * Generates a compact base64 JPEG thumbnail (~1.5 KB) from an HTMLImageElement
   * using an offscreen canvas to prevent localStorage quota exhaustion.
   */
  function generateThumbnailDataUrl(imgElement, size = 64) {
    try {
      if (!imgElement || !imgElement.naturalWidth || !imgElement.naturalHeight) {
        return null;
      }
      const canvas = document.createElement('canvas');
      canvas.width = size;
      canvas.height = size;
      const ctx = canvas.getContext('2d');
      if (!ctx) return null;

      const nw = imgElement.naturalWidth;
      const nh = imgElement.naturalHeight;
      const minDim = Math.min(nw, nh);
      const sx = (nw - minDim) / 2;
      const sy = (nh - minDim) / 2;

      ctx.drawImage(imgElement, sx, sy, minDim, minDim, 0, 0, size, size);
      return canvas.toDataURL('image/jpeg', 0.6);
    } catch {
      return null;
    }
  }

  function saveToRecentScans(data, imageSrc) {
    if (!data || typeof data.predicted_class_id !== 'number') return;
    try {
      const scans = getRecentScans();

      // Generate compact persistent thumbnail via canvas, avoiding broken blob: URLs
      let thumbDataUrl = null;
      if (elements.imagePreview && elements.imagePreview.naturalWidth > 0) {
        thumbDataUrl = generateThumbnailDataUrl(elements.imagePreview, 64);
      } else if (
        imageSrc &&
        typeof imageSrc === 'string' &&
        (imageSrc.startsWith('data:image/') || imageSrc.startsWith('http://') || imageSrc.startsWith('https://')) &&
        !imageSrc.startsWith('blob:') &&
        imageSrc.length < 50000
      ) {
        thumbDataUrl = imageSrc;
      }

      const newScan = {
        id: Date.now(),
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        crop: data.crop || 'Crop',
        condition: data.condition || 'Condition',
        health: data.health || 'Healthy',
        score: data.score,
        class_id: data.predicted_class_id,
        thumb: thumbDataUrl,
        top_predictions: Array.isArray(data.top_predictions) ? data.top_predictions : [],
      };

      const filtered = scans.filter((s) => !(s.crop === newScan.crop && s.condition === newScan.condition));
      const updated = [newScan, ...filtered].slice(0, MAX_RECENT_SCANS);
      localStorage.setItem(RECENT_SCANS_STORAGE_KEY, JSON.stringify(updated));
      renderRecentScans();
    } catch {
      // Graceful fallback if storage quota exceeded or unavailable
    }
  }

  function renderRecentScans() {
    if (!elements.recentScansList || !elements.recentScansSection) return;
    const scans = getRecentScans();
    if (scans.length === 0) {
      elements.recentScansSection.classList.add('hidden');
      elements.recentScansList.innerHTML = '';
      return;
    }

    elements.recentScansSection.classList.remove('hidden');
    elements.recentScansList.innerHTML = '';

    scans.forEach((scan) => {
      const card = document.createElement('div');
      card.className = 'recent-scan-item';
      card.setAttribute('role', 'button');
      card.setAttribute('tabindex', '0');
      card.setAttribute('aria-label', `Review recent scan: ${scan.crop} ${scan.condition}`);

      const hasValidThumb =
        scan.thumb &&
        typeof scan.thumb === 'string' &&
        (scan.thumb.startsWith('data:image/') || scan.thumb.startsWith('http://') || scan.thumb.startsWith('https://')) &&
        !scan.thumb.startsWith('blob:');

      let thumb;
      if (hasValidThumb) {
        thumb = document.createElement('img');
        thumb.className = 'recent-scan-thumb';
        thumb.alt = scan.condition;
        thumb.src = scan.thumb;
      } else {
        thumb = document.createElement('div');
        thumb.className = 'recent-scan-thumb recent-scan-thumb-placeholder';
        thumb.setAttribute('aria-hidden', 'true');
        thumb.innerHTML = `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M12 22C12 22 20 18 20 10C20 5 16 2 12 2C8 2 4 5 4 10C4 18 12 22 12 22Z"/><path d="M12 22V8"/><path d="M12 13L16 10"/><path d="M12 16L8 13"/></svg>`;
      }

      const title = document.createElement('span');
      title.className = 'recent-scan-title';
      title.textContent = `${scan.crop} — ${scan.condition}`;

      const meta = document.createElement('div');
      meta.className = 'recent-scan-meta';
      const scoreStr = typeof scan.score === 'number' ? `${(scan.score * 100).toFixed(1)}%` : '';
      const scoreSpan = document.createElement('span');
      scoreSpan.textContent = scoreStr;
      const timeSpan = document.createElement('span');
      timeSpan.textContent = scan.timestamp || '';
      meta.appendChild(scoreSpan);
      meta.appendChild(timeSpan);

      card.appendChild(thumb);
      card.appendChild(title);
      card.appendChild(meta);

      const handleScanClick = () => {
        // 1. Restore Primary Diagnosis Details
        elements.resultCrop.textContent = scan.crop || 'Unknown Crop';
        elements.resultCondition.textContent = scan.condition || 'Unknown Condition';

        const isHealthy = (scan.health || '').toLowerCase() === 'healthy';
        elements.resultHealthBadge.textContent = isHealthy ? '\u2713 Healthy' : '\u26A0 Diseased';
        elements.resultHealthBadge.className = 'badge ' + (isHealthy ? 'badge-healthy' : 'badge-diseased');

        const scorePercent = typeof scan.score === 'number' ? (scan.score * 100).toFixed(2) : '--';
        elements.resultScore.textContent = `${scorePercent}%`;

        // 2. Restore Relative Score Evaluation
        const scoreVal = typeof scan.score === 'number' ? scan.score : 0;
        if (elements.confidenceBandBadge) {
          if (scoreVal >= 0.90) {
            elements.confidenceBandBadge.textContent = 'Top Candidate';
            elements.confidenceBandBadge.className = 'badge badge-confidence-high';
            if (elements.confidenceGuidanceText) {
              elements.confidenceGuidanceText.textContent = 'Dominant relative softmax score across 15 trained classes';
            }
          } else if (scoreVal >= 0.70) {
            elements.confidenceBandBadge.textContent = 'Primary Candidate';
            elements.confidenceBandBadge.className = 'badge badge-confidence-moderate';
            if (elements.confidenceGuidanceText) {
              elements.confidenceGuidanceText.textContent = 'Review differential candidate classes below before action';
            }
          } else {
            elements.confidenceBandBadge.textContent = 'Diffuse Score Distribution';
            elements.confidenceBandBadge.className = 'badge badge-confidence-low';
            if (elements.confidenceGuidanceText) {
              elements.confidenceGuidanceText.textContent = 'Scores distributed across candidates; inspect differential monographs';
            }
          }
        }

        // 3. Restore Specimen Thumbnail in Header
        const resultThumb = document.getElementById('result-specimen-thumb');
        const wrap = resultThumb ? resultThumb.closest('.diagnosis-thumb-wrapper') : null;
        if (resultThumb && hasValidThumb) {
          resultThumb.src = scan.thumb;
          if (wrap) wrap.classList.remove('hidden');
        } else if (wrap) {
          wrap.classList.add('hidden');
        }

        // 4. Restore Candidates List
        if (Array.isArray(scan.top_predictions) && scan.top_predictions.length > 0) {
          currentCandidates = scan.top_predictions;
        } else {
          currentCandidates = [{
            class_id: scan.class_id,
            crop: scan.crop,
            condition: scan.condition,
            score: scan.score,
            health: scan.health,
            rank: 1,
          }];
        }
        renderCandidatesList();

        // 5. Restore View State
        elements.loadingContainer.classList.add('hidden');
        elements.resultsRejected.classList.add('hidden');
        elements.resultsEmpty.classList.add('hidden');
        elements.resultsContent.classList.remove('hidden');
        elements.newScanBtn.classList.remove('hidden');
        if (elements.printAdvisoryBtn) elements.printAdvisoryBtn.classList.remove('hidden');
        if (elements.copyGuidanceBtn) elements.copyGuidanceBtn.classList.remove('hidden');

        // 6. Fetch Monograph and notify
        activeCandidateClassId = scan.class_id;
        activeCandidateMeta = {
          isPrimary: true,
          crop: scan.crop,
          condition: scan.condition,
          rank: 1,
          score: scan.score,
        };
        fetchAndRenderKnowledge(scan.class_id, activeCandidateMeta);
        showToast(`Loaded recent scan: ${scan.crop} — ${scan.condition}`, '📖');

        // 7. Smooth Scroll on Mobile/Tablet
        if (window.innerWidth <= 1024) {
          elements.resultsContent.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      };

      card.addEventListener('click', handleScanClick);
      card.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          handleScanClick();
        }
      });

      elements.recentScansList.appendChild(card);
    });
  }

  function clearRecentScans() {
    try {
      localStorage.removeItem(RECENT_SCANS_STORAGE_KEY);
      renderRecentScans();
      showToast('Scan history cleared', '🧹');
    } catch {}
  }

  // ==========================================================================
  // Application Bootstrap & Readiness Probe
  // ==========================================================================
  async function init() {
    setupEventListeners();
    renderRecentScans();

    // Initial readiness probe
    try {
      const readyStatus = await window.CropDiseaseApi.checkReady();
      lastReadyState = readyStatus;
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

  // Expose print report hooks for testing and headless verification
  window.PhytoReport = {
    buildPrintReport: (p, k, c) => buildPrintReport(p || lastPrediction, k || currentActiveKnowledgeData, c || activeCandidateMeta),
    prepareAndTriggerPrint: prepareAndTriggerPrint,
    getLastPrediction: () => lastPrediction,
    getActiveKnowledge: () => currentActiveKnowledgeData,
    getActiveCandidate: () => activeCandidateMeta,
    getReadyState: () => lastReadyState,
    setReadyState: (s) => { lastReadyState = s; },
    handleSelectedFile: handleSelectedFile,
  };

  // Run on DOM load
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
