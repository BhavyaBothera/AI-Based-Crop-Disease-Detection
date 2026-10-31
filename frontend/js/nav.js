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

    function openMobileMenu() {
      if (mobileMenu) {
        mobileMenu.classList.add('open');
        document.body.style.overflow = 'hidden';
      }
      if (mobileBackdrop) mobileBackdrop.classList.add('visible');
    }

    function closeMobileMenu() {
      if (mobileMenu) {
        mobileMenu.classList.remove('open');
        document.body.style.overflow = '';
      }
      if (mobileBackdrop) mobileBackdrop.classList.remove('visible');
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

    // Polling System Readiness Probe
    pollReadiness();
  }

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
      } else {
        statusTextEl.textContent = 'Initializing...';
        if (statusIndicatorEl) {
          statusIndicatorEl.className = 'status-indicator status-warning';
        }
      }
    } catch {
      statusTextEl.textContent = 'Backend Offline';
      if (statusIndicatorEl) {
        statusIndicatorEl.className = 'status-indicator status-offline';
        statusIndicatorEl.setAttribute('title', 'FastAPI daemon offline. Verify port 8000.');
      }
    }

    // Repeat poll every 20 seconds
    setTimeout(pollReadiness, 20000);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initNavigation);
  } else {
    initNavigation();
  }
})();
