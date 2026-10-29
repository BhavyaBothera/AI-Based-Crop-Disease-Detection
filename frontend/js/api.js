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
 * Raised when the backend declines to classify an upload because the image could not
 * be verified as a supported crop leaf (status 'rejected') or was too ambiguous or
 * low-quality to assess (status 'abstained').
 *
 * This is deliberately distinct from a transport, upload or service error: callers must
 * render a rejection state and must never treat it as a diagnosis.
 */
class InputRejectedError extends ApiError {
  constructor(payload = {}, status = 422) {
    const message =
      typeof payload.message === 'string' && payload.message.trim()
        ? payload.message.trim()
        : "We couldn't verify that this image contains a supported crop leaf.";
    super(message, status, payload);
    this.name = 'InputRejectedError';
    this.statusText = payload.status === 'abstained' ? 'abstained' : 'rejected';
    this.code = typeof payload.code === 'string' ? payload.code : 'INPUT_NOT_VERIFIED';
    this.category = typeof payload.category === 'string' ? payload.category : 'ambiguous_image';
    this.detailMessage = typeof payload.detail === 'string' ? payload.detail : '';
    this.signals = payload.signals && typeof payload.signals === 'object' ? payload.signals : {};
  }
}

/**
 * Reads a JSON error body once. Returns null when the body is absent or not JSON.
 */
async function readJsonPayload(response) {
  try {
    const contentType = response?.headers?.get('content-type') || '';
    if (!contentType.includes('application/json')) return null;
    return await response.json();
  } catch {
    return null;
  }
}

/**
 * Maps an HTTP status to the fallback user-facing message used when the server did not
 * provide a usable detail string.
 */
function statusFallbackMessage(status) {
  switch (status) {
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
      return status
        ? `Server returned an error (HTTP ${status}). Please try again.`
        : 'An unexpected communication error occurred. Please try again.';
  }
}

/**
 * Extracts the best available user-facing message from a parsed error payload.
 */
function messageFromPayload(data, status) {
  if (data && typeof data.detail === 'string' && data.detail.trim()) {
    return data.detail.trim();
  }
  if (
    data &&
    data.detail &&
    typeof data.detail === 'object' &&
    typeof data.detail.message === 'string' &&
    data.detail.message.trim()
  ) {
    return data.detail.message.trim();
  }
  if (data && typeof data.message === 'string' && data.message.trim()) {
    return data.message.trim();
  }
  return statusFallbackMessage(status);
}

/**
 * Converts a structured content-gate response into an InputRejectedError, or null when
 * the response is an ordinary upload/service error.
 */
function buildInputRejectionError(data, status) {
  if (!data || typeof data !== 'object') return null;
  const detail = data.detail;
  if (!detail || typeof detail !== 'object' || Array.isArray(detail)) return null;
  if (detail.status !== 'rejected' && detail.status !== 'abstained') return null;
  if (typeof detail.code !== 'string') return null;
  return new InputRejectedError(detail, status);
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
  const data = await readJsonPayload(response);
  return messageFromPayload(data, response?.status);
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
    const data = await readJsonPayload(response);
    const rejection = buildInputRejectionError(data, response.status);
    if (rejection) throw rejection;
    throw new ApiError(messageFromPayload(data, response.status), response.status, data?.detail ?? null);
  }

  const result = await response.json();

  // Validate response contract integrity. A response is only treated as a diagnosis when
  // it carries a complete, well-formed prediction contract.
  if (
    typeof result.predicted_class_id !== 'number' ||
    !Number.isInteger(result.predicted_class_id) ||
    result.predicted_class_id < 0 ||
    result.predicted_class_id > 14 ||
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
    InputRejectedError,
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
    InputRejectedError,
    fetchWithTimeout,
    parseErrorResponse,
    checkHealth,
    checkReady,
    predictCropDisease,
    getDiseaseKnowledge,
  };
}
