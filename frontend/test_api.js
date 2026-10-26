/**
 * Automated test suite for frontend/js/api.js using Node.js built-in test runner.
 * No external dependencies required.
 *
 * Run with: node --test frontend/test_api.js
 */
const { test, describe, beforeEach, afterEach } = require('node:test');
const assert = require('node:assert/strict');

// Set up browser-like global environment before requiring api.js
global.window = global;
const api = require('./js/api.js');

describe('CropDiseaseApi - Export Verification', () => {
  test('CropDiseaseApi exports all expected methods and properties', () => {
    assert.ok(api.API_BASE_URL, 'API_BASE_URL should be exported');
    assert.strictEqual(typeof api.ApiError, 'function', 'ApiError should be a constructor');
    assert.strictEqual(typeof api.checkHealth, 'function', 'checkHealth should be a function');
    assert.strictEqual(typeof api.checkReady, 'function', 'checkReady should be a function');
    assert.strictEqual(typeof api.predictCropDisease, 'function', 'predictCropDisease should be a function');
    assert.strictEqual(typeof api.getDiseaseKnowledge, 'function', 'getDiseaseKnowledge should be a function');

    // Also verify window.CropDiseaseApi
    assert.strictEqual(window.CropDiseaseApi.getDiseaseKnowledge, api.getDiseaseKnowledge);
  });
});

describe('CropDiseaseApi.getDiseaseKnowledge - Client-Side Validation', () => {
  const originalFetch = global.fetch;
  let fetchCallCount = 0;

  beforeEach(() => {
    fetchCallCount = 0;
    global.fetch = async () => {
      fetchCallCount++;
      throw new Error('fetch should not be called for invalid inputs');
    };
  });

  afterEach(() => {
    global.fetch = originalFetch;
  });

  const invalidTestCases = [
    { value: -1, desc: 'negative integer -1' },
    { value: 15, desc: 'out-of-range integer 15' },
    { value: 99, desc: 'large integer 99' },
    { value: 2.5, desc: 'floating point number 2.5' },
    { value: NaN, desc: 'NaN' },
    { value: Infinity, desc: 'Infinity' },
    { value: -Infinity, desc: '-Infinity' },
    { value: '2', desc: 'string "2"' },
    { value: '0', desc: 'string "0"' },
    { value: null, desc: 'null' },
    { value: undefined, desc: 'undefined' },
    { value: {}, desc: 'empty object {}' },
    { value: [], desc: 'empty array []' },
    { value: true, desc: 'boolean true' },
    { value: false, desc: 'boolean false' },
  ];

  for (const { value, desc } of invalidTestCases) {
    test(`Rejects ${desc} before initiating a network request`, async () => {
      await assert.rejects(
        async () => {
          await api.getDiseaseKnowledge(value);
        },
        (err) => {
          assert.ok(err instanceof api.ApiError, 'Error should be an instance of ApiError');
          assert.strictEqual(err.status, 400, 'Error status should be 400');
          assert.match(err.message, /Invalid class ID/, 'Error message should explain invalid class ID');
          return true;
        }
      );
      assert.strictEqual(fetchCallCount, 0, `Network request was unexpectedly initiated for ${desc}`);
    });
  }
});

describe('CropDiseaseApi.getDiseaseKnowledge - Mocked Network & Error Handling', () => {
  const originalFetch = global.fetch;

  afterEach(() => {
    global.fetch = originalFetch;
  });

  test('Successful request parses JSON and uses GET with 10s timeout', async () => {
    let capturedUrl = '';
    let capturedOptions = {};

    global.fetch = async (url, options) => {
      capturedUrl = url;
      capturedOptions = options;
      return {
        ok: true,
        status: 200,
        headers: new Headers({ 'content-type': 'application/json' }),
        json: async () => ({
          class_id: 2,
          crop: 'Potato',
          condition: 'Early Blight',
          health_status: 'Diseased',
        }),
      };
    };

    const result = await api.getDiseaseKnowledge(2);
    assert.strictEqual(capturedUrl, `${api.API_BASE_URL}/api/knowledge/2`);
    assert.strictEqual(capturedOptions.method, 'GET');
    assert.ok(capturedOptions.signal, 'AbortSignal should be attached');
    assert.strictEqual(result.class_id, 2);
    assert.strictEqual(result.crop, 'Potato');
  });

  test('HTTP 404 response extracts server detail message', async () => {
    global.fetch = async () => ({
      ok: false,
      status: 404,
      headers: new Headers({ 'content-type': 'application/json' }),
      json: async () => ({
        detail: 'Disease knowledge record not found for class_id 15.',
      }),
    });

    await assert.rejects(
      async () => {
        // Bypass validation just to test 404 parsing logic
        const url = `${api.API_BASE_URL}/api/knowledge/15`;
        const res = await api.fetchWithTimeout(url);
        if (!res.ok) {
          const msg = await api.parseErrorResponse(res);
          throw new api.ApiError(msg, res.status);
        }
      },
      (err) => {
        assert.ok(err instanceof api.ApiError);
        assert.strictEqual(err.status, 404);
        assert.strictEqual(err.message, 'Disease knowledge record not found for class_id 15.');
        return true;
      }
    );
  });

  test('HTTP 503 Service Unavailable extracts server detail message', async () => {
    global.fetch = async () => ({
      ok: false,
      status: 503,
      headers: new Headers({ 'content-type': 'application/json' }),
      json: async () => ({
        detail: 'Knowledge base service is currently unavailable.',
      }),
    });

    await assert.rejects(
      async () => {
        await api.getDiseaseKnowledge(0);
      },
      (err) => {
        assert.ok(err instanceof api.ApiError);
        assert.strictEqual(err.status, 503);
        assert.strictEqual(err.message, 'Knowledge base service is currently unavailable.');
        return true;
      }
    );
  });

  test('HTTP 422 Unprocessable Entity falls back to standard message', async () => {
    global.fetch = async () => ({
      ok: false,
      status: 422,
      headers: new Headers({ 'content-type': 'application/json' }),
      json: async () => ({
        detail: [{ loc: ['path', 'class_id'], msg: 'Input should be a valid integer' }],
      }),
    });

    await assert.rejects(
      async () => {
        await api.getDiseaseKnowledge(1);
      },
      (err) => {
        assert.ok(err instanceof api.ApiError);
        assert.strictEqual(err.status, 422);
        assert.match(err.message, /HTTP 422/);
        return true;
      }
    );
  });

  test('Network connection failure raises ApiError with status 0', async () => {
    global.fetch = async () => {
      throw new TypeError('Failed to fetch');
    };

    await assert.rejects(
      async () => {
        await api.getDiseaseKnowledge(0);
      },
      (err) => {
        assert.ok(err instanceof api.ApiError);
        assert.strictEqual(err.status, 0);
        assert.match(err.message, /Unable to connect/);
        return true;
      }
    );
  });

  test('Request timeout raises ApiError with status 408', async () => {
    global.fetch = async () => {
      const abortError = new Error('The operation was aborted');
      abortError.name = 'AbortError';
      throw abortError;
    };

    await assert.rejects(
      async () => {
        await api.getDiseaseKnowledge(0);
      },
      (err) => {
        assert.ok(err instanceof api.ApiError);
        assert.strictEqual(err.status, 408);
        assert.match(err.message, /timed out/);
        return true;
      }
    );
  });
});

describe('CropDiseaseApi - Regression Tests for Existing Endpoints', () => {
  const originalFetch = global.fetch;

  afterEach(() => {
    global.fetch = originalFetch;
  });

  test('checkHealth sends GET to /health', async () => {
    let capturedUrl = '';
    global.fetch = async (url) => {
      capturedUrl = url;
      return {
        ok: true,
        status: 200,
        headers: new Headers({ 'content-type': 'application/json' }),
        json: async () => ({ status: 'ok', app: 'API', version: '1.0.0' }),
      };
    };

    const res = await api.checkHealth();
    assert.strictEqual(capturedUrl, `${api.API_BASE_URL}/health`);
    assert.strictEqual(res.status, 'ok');
  });

  test('checkReady returns ready: true when model loaded', async () => {
    global.fetch = async () => ({
      ok: true,
      status: 200,
      headers: new Headers({ 'content-type': 'application/json' }),
      json: async () => ({ status: 'ready', model_loaded: true, model_path: 'model.keras' }),
    });

    const res = await api.checkReady();
    assert.strictEqual(res.ready, true);
    assert.strictEqual(res.status, 'ready');
  });

  test('predictCropDisease rejects null file input with 400', async () => {
    await assert.rejects(
      async () => {
        await api.predictCropDisease(null);
      },
      (err) => {
        assert.ok(err instanceof api.ApiError);
        assert.strictEqual(err.status, 400);
        assert.match(err.message, /No image selected/);
        return true;
      }
    );
  });
});

describe('CropDiseaseApi.getDiseaseKnowledge - Live Integration with Backend', () => {
  // Test live integration if backend is running at http://127.0.0.1:8000
  test('Live query for class_id 0 (Bell Pepper Bacterial Spot)', async (t) => {
    try {
      const res = await api.getDiseaseKnowledge(0);
      assert.strictEqual(res.class_id, 0);
      assert.strictEqual(res.crop, 'Bell Pepper');
      assert.strictEqual(res.condition, 'Bacterial Spot');
      assert.strictEqual(res.health_status, 'Diseased');
      assert.ok(res.symptoms, 'Symptoms should be present');
      assert.ok(res.treatment, 'Treatment should be present');
      assert.ok(Array.isArray(res.sources), 'Sources should be an array');
    } catch (err) {
      if (err.status === 0) {
        // Backend offline: report a real skip rather than a silent pass
        t.skip(`Backend not reachable at ${api.API_BASE_URL}`);
        return;
      }
      throw err;
    }
  });

  test('Live query for class_id 2 (Potato Early Blight)', async (t) => {
    try {
      const res = await api.getDiseaseKnowledge(2);
      assert.strictEqual(res.class_id, 2);
      assert.strictEqual(res.crop, 'Potato');
      assert.strictEqual(res.condition, 'Early Blight');
      assert.strictEqual(res.health_status, 'Diseased');
    } catch (err) {
      if (err.status === 0) {
        t.skip(`Backend not reachable at ${api.API_BASE_URL}`);
        return;
      }
      throw err;
    }
  });

  test('Live query for class_id 14 (Tomato Healthy)', async (t) => {
    try {
      const res = await api.getDiseaseKnowledge(14);
      assert.strictEqual(res.class_id, 14);
      assert.strictEqual(res.crop, 'Tomato');
      assert.strictEqual(res.condition, 'Healthy');
      assert.strictEqual(res.health_status, 'Healthy');
      assert.strictEqual(res.causal_agent, null);
    } catch (err) {
      if (err.status === 0) {
        t.skip(`Backend not reachable at ${api.API_BASE_URL}`);
        return;
      }
      throw err;
    }
  });
});
