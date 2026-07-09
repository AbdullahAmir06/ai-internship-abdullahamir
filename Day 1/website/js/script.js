/* ==========================================================================
   NCERT Website — Interactivity
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
  initNavToggle();
  initScrollSpy();
  initHeaderShadowOnScroll();
  initRevealOnScroll();
  initCounters();
  initBackToTop();
  initContactForm();
  initNetworkCanvas();
  document.getElementById('year').textContent = new Date().getFullYear();
});

/* ---------- Mobile nav toggle ---------- */
function initNavToggle() {
  const toggle = document.getElementById('navToggle');
  const navbar = document.getElementById('navbar');

  toggle.addEventListener('click', () => {
    const isOpen = navbar.classList.toggle('open');
    toggle.classList.toggle('open', isOpen);
    toggle.setAttribute('aria-expanded', String(isOpen));
  });

  navbar.querySelectorAll('[data-nav]').forEach(link => {
    link.addEventListener('click', () => {
      navbar.classList.remove('open');
      toggle.classList.remove('open');
      toggle.setAttribute('aria-expanded', 'false');
    });
  });
}

/* ---------- Highlight active nav link while scrolling ---------- */
function initScrollSpy() {
  const sections = document.querySelectorAll('section[id]');
  const navLinks = document.querySelectorAll('.nav-link');

  const observer = new IntersectionObserver(
    entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const id = entry.target.getAttribute('id');
          navLinks.forEach(link => {
            link.classList.toggle('active', link.getAttribute('href') === `#${id}`);
          });
        }
      });
    },
    { rootMargin: '-45% 0px -50% 0px', threshold: 0 }
  );

  sections.forEach(section => observer.observe(section));
}

/* ---------- Header background shrink/shadow on scroll ---------- */
function initHeaderShadowOnScroll() {
  const header = document.getElementById('header');
  window.addEventListener('scroll', () => {
    header.style.boxShadow = window.scrollY > 12 ? '0 8px 24px rgba(0,0,0,0.35)' : 'none';
  }, { passive: true });
}

/* ---------- Fade/slide elements into view ---------- */
function initRevealOnScroll() {
  const targets = document.querySelectorAll('.dept-card, .timeline-item, .about-text, .about-visual, .contact-info, .contact-form');
  targets.forEach(el => el.classList.add('reveal'));

  const observer = new IntersectionObserver(
    entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('in-view');
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.15 }
  );

  targets.forEach(el => observer.observe(el));
}

/* ---------- Animated stat counters ---------- */
function initCounters() {
  const counters = document.querySelectorAll('[data-count]');
  if (!counters.length) return;

  const animate = el => {
    const target = parseInt(el.getAttribute('data-count'), 10);
    const duration = 1400;
    const start = performance.now();

    const step = now => {
      const progress = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      el.textContent = Math.floor(eased * target);
      if (progress < 1) requestAnimationFrame(step);
      else el.textContent = target;
    };
    requestAnimationFrame(step);
  };

  const observer = new IntersectionObserver(
    entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          animate(entry.target);
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.5 }
  );

  counters.forEach(el => observer.observe(el));
}

/* ---------- Back to top button ---------- */
function initBackToTop() {
  const btn = document.getElementById('backToTop');
  window.addEventListener('scroll', () => {
    btn.classList.toggle('show', window.scrollY > 500);
  }, { passive: true });

  btn.addEventListener('click', () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });
}

/* ---------- Contact form validation (client-side only) ---------- */
function initContactForm() {
  const form = document.getElementById('contactForm');
  const success = document.getElementById('formSuccess');
  if (!form) return;

  const validators = {
    name: value => value.trim().length >= 2 || 'Please enter your full name.',
    email: value => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value) || 'Please enter a valid email address.',
    subject: value => value.trim().length >= 3 || 'Please enter a subject.',
    message: value => value.trim().length >= 10 || 'Message should be at least 10 characters.',
  };

  form.addEventListener('submit', e => {
    e.preventDefault();
    let isValid = true;

    Object.keys(validators).forEach(field => {
      const input = form.elements[field];
      const errorEl = form.querySelector(`[data-error="${field}"]`);
      const result = validators[field](input.value);

      if (result !== true) {
        input.classList.add('invalid');
        errorEl.textContent = result;
        isValid = false;
      } else {
        input.classList.remove('invalid');
        errorEl.textContent = '';
      }
    });

    if (isValid) {
      success.classList.add('show');
      form.reset();
      setTimeout(() => success.classList.remove('show'), 5000);
    } else {
      success.classList.remove('show');
    }
  });

  Object.keys(validators).forEach(field => {
    const input = form.elements[field];
    input.addEventListener('input', () => {
      if (input.classList.contains('invalid') && validators[field](input.value) === true) {
        input.classList.remove('invalid');
        form.querySelector(`[data-error="${field}"]`).textContent = '';
      }
    });
  });
}

/* ---------- Hero background: animated network of nodes ---------- */
function initNetworkCanvas() {
  const canvas = document.getElementById('netCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  let width, height, nodes;
  const NODE_COUNT_DENSITY = 14000;
  const LINK_DIST = 150;

  function resize() {
    const hero = canvas.closest('.hero');
    width = canvas.width = hero.offsetWidth;
    height = canvas.height = hero.offsetHeight;
    const count = Math.max(30, Math.floor((width * height) / NODE_COUNT_DENSITY));
    nodes = Array.from({ length: count }, () => ({
      x: Math.random() * width,
      y: Math.random() * height,
      vx: (Math.random() - 0.5) * 0.35,
      vy: (Math.random() - 0.5) * 0.35,
    }));
  }

  function tick() {
    ctx.clearRect(0, 0, width, height);

    nodes.forEach(n => {
      n.x += n.vx;
      n.y += n.vy;
      if (n.x < 0 || n.x > width) n.vx *= -1;
      if (n.y < 0 || n.y > height) n.vy *= -1;
    });

    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const dx = nodes[i].x - nodes[j].x;
        const dy = nodes[i].y - nodes[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < LINK_DIST) {
          ctx.strokeStyle = `rgba(34, 211, 238, ${0.16 * (1 - dist / LINK_DIST)})`;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(nodes[i].x, nodes[i].y);
          ctx.lineTo(nodes[j].x, nodes[j].y);
          ctx.stroke();
        }
      }
    }

    nodes.forEach(n => {
      ctx.fillStyle = 'rgba(34, 211, 238, 0.55)';
      ctx.beginPath();
      ctx.arc(n.x, n.y, 1.6, 0, Math.PI * 2);
      ctx.fill();
    });

    requestAnimationFrame(tick);
  }

  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  resize();
  if (!prefersReducedMotion) requestAnimationFrame(tick);

  let resizeTimer;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(resize, 200);
  });
}
