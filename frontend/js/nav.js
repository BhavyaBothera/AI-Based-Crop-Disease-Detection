/**
 * PHYTOSIGNAL — Shared Navigation & Readiness Controller
 * Manages active navigation states, mobile menu toggle, and backend readiness polling.
 */
(function () {
  'use strict';

  function initNavigation() {
    const currentPath = window.location.pathname.toLowerCase();
    const navLinks = document.querySelectorAll('.nav-link');

    navLinks.forEach((link) => {
      const href = (link.getAttribute('href') || '').toLowerCase();
      const route = (link.getAttribute('data-route') || '').toLowerCase();

      let isActive = false;
      if (route === '/analyze' || href.includes('analyze.html')) {
        isActive = currentPath.includes('analyze');
      } else if (route === '/field-guide' || href.includes('field-guide.html')) {
        isActive = currentPath.includes('field-guide');
      } else if (route === '/about' || href.includes('about.html')) {
        isActive = currentPath.includes('about');
      } else if (route === '/' || href.endsWith('index.html') || href === '/') {
        isActive =
          !currentPath.includes('analyze') &&
          !currentPath.includes('field-guide') &&
          !currentPath.includes('about');
      }

      if (isActive) {
        link.classList.add('active');
        link.setAttribute('aria-current', 'page');
      } else {
        link.classList.remove('active');
        link.removeAttribute('aria-current');
      }
    });

    // Mobile Navigation Toggle
    const mobileToggle = document.getElementById('mobile-menu-toggle');
    const mobileMenu = document.getElementById('mobile-menu');
    const mobileBackdrop = document.getElementById('mobile-menu-backdrop');
    const mobileClose = document.getElementById('mobile-menu-close');
    const mobileLinks = document.querySelectorAll('.mobile-nav-links a');

    function handleMobileMenuKeyDown(e) {
      if (e.key === 'Escape') {
        e.preventDefault();
        closeMobileMenu();
        return;
      }
      if (e.key === 'Tab') {
        if (!mobileMenu) return;
        const focusables = mobileMenu.querySelectorAll(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );
        if (!focusables || focusables.length === 0) return;
        const first = focusables[0];
        const last = focusables[focusables.length - 1];
        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      }
    }

    function openMobileMenu() {
      if (mobileToggle) {
        mobileToggle.setAttribute('aria-expanded', 'true');
      }
      if (mobileMenu) {
        mobileMenu.classList.add('open');
        mobileMenu.setAttribute('aria-hidden', 'false');
        document.body.style.overflow = 'hidden';
      }
      if (mobileBackdrop) {
        mobileBackdrop.classList.add('visible');
      }
      document.addEventListener('keydown', handleMobileMenuKeyDown);
      if (mobileClose) {
        setTimeout(() => mobileClose.focus(), 60);
      }
    }

    function closeMobileMenu() {
      if (mobileToggle) {
        mobileToggle.setAttribute('aria-expanded', 'false');
      }
      if (mobileMenu) {
        mobileMenu.classList.remove('open');
        mobileMenu.setAttribute('aria-hidden', 'true');
        document.body.style.overflow = '';
      }
      if (mobileBackdrop) {
        mobileBackdrop.classList.remove('visible');
      }
      document.removeEventListener('keydown', handleMobileMenuKeyDown);
      if (mobileToggle && typeof mobileToggle.focus === 'function') {
        mobileToggle.focus();
      }
    }

    if (mobileToggle) {
      mobileToggle.addEventListener('click', openMobileMenu);
    }
    if (mobileClose) {
      mobileClose.addEventListener('click', closeMobileMenu);
    }
    if (mobileBackdrop) {
      mobileBackdrop.addEventListener('click', closeMobileMenu);
    }
    mobileLinks.forEach((link) => {
      link.addEventListener('click', closeMobileMenu);
    });

    // Polling System Readiness Probe
    pollReadiness();
  }

  let pollIntervalMs = 4000;
  let consecutiveFailures = 0;

  async function pollReadiness() {
    const statusTextEl = document.getElementById('connection-status-text');
    const statusIndicatorEl = document.getElementById('connection-status');

    if (!statusTextEl || !window.CropDiseaseApi) return;

    try {
      const readyState = await window.CropDiseaseApi.checkReady();
      if (readyState && readyState.ready) {
        statusTextEl.textContent = 'Model Ready';
        if (statusIndicatorEl) {
          statusIndicatorEl.className = 'status-indicator status-ready';
          statusIndicatorEl.setAttribute('title', 'Neural inference engine & leaf gate operational');
        }
        consecutiveFailures = 0;
        pollIntervalMs = 30000; // Relax polling once healthy
      } else {
        statusTextEl.textContent = 'Initializing...';
        if (statusIndicatorEl) {
          statusIndicatorEl.className = 'status-indicator status-warning';
          statusIndicatorEl.setAttribute('title', 'FastAPI daemon initializing neural weights');
        }
        pollIntervalMs = 4000;
      }
    } catch {
      consecutiveFailures++;
      statusTextEl.textContent = 'Backend Offline';
      if (statusIndicatorEl) {
        statusIndicatorEl.className = 'status-indicator status-offline';
        statusIndicatorEl.setAttribute('title', 'FastAPI daemon offline. Verify port 8000.');
      }
      // Exponential backoff up to 16 seconds when offline
      pollIntervalMs = Math.min(4000 * Math.pow(1.4, Math.min(consecutiveFailures, 4)), 16000);
    }

    // Schedule next probe with adaptive interval
    setTimeout(pollReadiness, pollIntervalMs);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initNavigation);
  } else {
    initNavigation();
  }
})();
