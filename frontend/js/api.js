/**
 * Centralized API Client for Crop Disease Detection Backend
 *
 * Connects the vanilla frontend to the FastAPI backend service.
 * Standardizes endpoints, request formatting, timeout handling, and error mapping.
 */

// Central API Base URL configuration.
// Can be overridden globally via window.API_BASE_URL if needed.
const API_BASE_URL = (typeof window !== 'undefined' && window.API_BASE_URL) || 'http://127.0.0.1:8000';

/**
 * Custom error class for API failures with user-friendly messages.
 */
class ApiError extends Error {
  constructor(message, status = 0, detail = null) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

/**
 * Performs a fetch request with timeout protection via AbortController.
 *
 * @param {string} endpoint Relative or absolute URL
 * @param {RequestInit} options Fetch options
 * @param {number} timeoutMs Request timeout in milliseconds
 * @returns {Promise<Response>}
 */
async function fetchWithTimeout(url, options = {}, timeoutMs = 25000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    return response;
  } catch (err) {
    if (err.name === 'AbortError') {
      throw new ApiError(
        'Request timed out. The server is taking longer than expected to respond. Please try again.',
        408
      );
    }
    // Network or connection failure
    throw new ApiError(
      'Unable to connect to the prediction server. Please verify that the backend is running at ' +
        API_BASE_URL +
        '.',
      0
    );
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Safe error parser to extract user-friendly messages from FastAPI responses.
 * Never exposes raw tracebacks, internal filepaths, or sensitive server details.
 *
 * @param {Response} response Fetch Response object
 * @returns {Promise<string>} Clean error description
 */
async function parseErrorResponse(response) {
  let detailMessage = '';
  try {
    const contentType = response?.headers?.get('content-type') || '';
    if (contentType.includes('application/json')) {
      const data = await response.json();
      if (data && typeof data.detail === 'string' && data.detail.trim()) {
        detailMessage = data.detail.trim();
      } else if (data && typeof data.message === 'string' && data.message.trim()) {
        detailMessage = data.message.trim();
      }
    }
  } catch {
    // If parsing fails, detailMessage stays empty and falls back to safe status mapping
  }

  if (detailMessage) {
    return detailMessage;
  }

  switch (response?.status) {
    case 400:
      return 'The submitted file is invalid, corrupt, or unsupported. Please check the image format and resolution.';
    case 404:
      return 'The requested API endpoint was not found on the server.';
    case 408:
      return 'Request timed out. The server took too long to respond. Please try again.';
    case 413:
      return 'The selected file is too large. Uploads must not exceed 10 MB.';
    case 502:
      return 'Bad gateway. The upstream inference engine is unavailable. Please verify the backend service.';
    case 503:
      return 'The model inference service is currently initializing or unavailable. Please wait a moment and try again.';
    case 504:
      return 'Gateway timeout. The server took too long to complete processing.';
    case 500:
      return 'An unexpected server error occurred during inference. Please try again or select another image.';
    default:
      return response?.status
        ? `Server returned an error (HTTP ${response.status}). Please try again.`
        : 'An unexpected communication error occurred. Please try again.';
  }
}

/**
 * Health check probe: Verifies FastAPI process is running.
 *
 * @returns {Promise<{ status: string, app: string, version: string }>}
 */
async function checkHealth() {
  const url = `${API_BASE_URL}/health`;
  const response = await fetchWithTimeout(url, { method: 'GET' }, 5000);

  if (!response.ok) {
    const message = await parseErrorResponse(response);
    throw new ApiError(message, response.status);
  }

  return response.json();
}

/**
 * Readiness probe: Verifies ML model is loaded in memory and ready for inference.
 *
 * @returns {Promise<{ ready: boolean, status: string, modelPath?: string, detail?: string }>}
 */
async function checkReady() {
  const url = `${API_BASE_URL}/ready`;

  try {
    const response = await fetchWithTimeout(url, { method: 'GET' }, 5000);
    const data = await response.json().catch(() => ({}));

    if (response.ok && data.status === 'ready' && data.model_loaded === true) {
      return {
        ready: true,
        status: data.status,
        modelPath: data.model_path || '',
      };
    }

    return {
      ready: false,
      status: data.status || 'not_ready',
      detail: data.detail || 'Model is not yet ready for inference.',
    };
  } catch (err) {
    return {
      ready: false,
      status: 'offline',
      detail: err.message || 'Cannot reach server.',
    };
  }
}

/**
 * Sends a leaf image to POST /api/predict for disease classification.
 *
 * Uses multipart/form-data with the authoritative field name 'file'.
 *
 * @param {File|Blob} file Image file to classify
 * @returns {Promise<PredictionResponse>} Standardized prediction response object
 */
async function predictCropDisease(file) {
  if (!file) {
    throw new ApiError('No image selected for prediction.', 400);
  }

  const url = `${API_BASE_URL}/api/predict`;
  const formData = new FormData();
  formData.append('file', file, file.name || 'leaf_image.jpg');

  const response = await fetchWithTimeout(
    url,
    {
      method: 'POST',
      body: formData,
    },
    30000 // 30-second bounded timeout for inference
  );

  if (!response.ok) {
    const message = await parseErrorResponse(response);
    throw new ApiError(message, response.status);
  }

  const result = await response.json();

  // Validate response contract integrity
  if (
    typeof result.predicted_class_id !== 'number' ||
    !Array.isArray(result.top_predictions) ||
    result.top_predictions.length !== 3 ||
    typeof result.disclaimer !== 'string'
  ) {
    throw new ApiError(
      'Received an unexpected or incomplete response format from the prediction service.',
      502
    );
  }

  return result;
}

/**
 * Retrieves validated disease knowledge, symptoms, prevention, IPM treatment
 * guidelines, and source references for a specific class ID (0-14).
 *
 * @param {number} classId Authoritative integer class ID (0 through 14)
 * @returns {Promise<Object>} Validated disease knowledge record object
 */
async function getDiseaseKnowledge(classId) {
  // Validate class ID before initiating any network request
  if (!Number.isInteger(classId) || classId < 0 || classId > 14) {
    throw new ApiError(
      `Invalid class ID '${classId}'. Expected an integer between 0 and 14.`,
      400
    );
  }

  const url = `${API_BASE_URL}/api/knowledge/${classId}`;
  const response = await fetchWithTimeout(url, { method: 'GET' }, 10000);

  if (!response.ok) {
    const message = await parseErrorResponse(response);
    throw new ApiError(message, response.status);
  }

  return response.json();
}

// Expose on global window object for vanilla JavaScript script inclusion
if (typeof window !== 'undefined') {
  window.CropDiseaseApi = {
    API_BASE_URL,
    ApiError,
    checkHealth,
    checkReady,
    predictCropDisease,
    getDiseaseKnowledge,
  };
}

// Support CommonJS export for Node.js test runners without external bundlers
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    API_BASE_URL,
    ApiError,
    fetchWithTimeout,
    parseErrorResponse,
    checkHealth,
    checkReady,
    predictCropDisease,
    getDiseaseKnowledge,
  };
}
