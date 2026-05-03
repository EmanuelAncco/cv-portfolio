/**
 * Portfolio animations — Engineer's Field Notebook v2.1
 * Progressive enhancement: content is visible by default (CSS),
 * JS only adds polish via IntersectionObserver toggling .is-visible.
 *
 * No GSAP required. Pure CSS transitions + IO. Smaller bundle, more robust.
 */

// Mark <html> so progressive-enhanced CSS rules activate
document.documentElement.classList.add("has-js");

function activate() {
  const revealables = document.querySelectorAll<HTMLElement>(
    ".reveal, .word-fade, .hero-fade"
  );

  // Above-the-fold elements: reveal immediately with a tiny stagger
  const heroFades = document.querySelectorAll<HTMLElement>(".hero-fade, .word-fade");
  heroFades.forEach((el, i) => {
    setTimeout(() => el.classList.add("is-visible"), 80 + i * 120);
  });

  // Below-fold reveals: IntersectionObserver, fires once
  if (!("IntersectionObserver" in window)) {
    revealables.forEach((el) => el.classList.add("is-visible"));
    return;
  }

  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          io.unobserve(entry.target);
        }
      });
    },
    { rootMargin: "0px 0px -8% 0px", threshold: 0.05 }
  );

  document.querySelectorAll<HTMLElement>(".reveal").forEach((el) => {
    // Anything already in initial viewport: reveal immediately
    const rect = el.getBoundingClientRect();
    if (rect.top < window.innerHeight) {
      el.classList.add("is-visible");
    } else {
      io.observe(el);
    }
  });
}

// Counter animation: numbers count up when their parent enters viewport
function initCounters() {
  const counters = document.querySelectorAll<HTMLElement>("[data-count]");
  if (counters.length === 0 || !("IntersectionObserver" in window)) return;

  const animate = (el: HTMLElement) => {
    const target = parseInt(el.dataset.count ?? "0", 10);
    const start = performance.now();
    const duration = 900;
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      el.textContent = Math.round(target * eased).toString();
      if (t < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  };

  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          animate(entry.target as HTMLElement);
          io.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.5 }
  );
  counters.forEach((el) => {
    el.textContent = "0";
    io.observe(el);
  });
}

// Mouse parallax on hero photo (light, eased)
function initHeroParallax() {
  const photo = document.getElementById("hero-photo");
  if (!photo) return;
  let tx = 0, ty = 0, cx = 0, cy = 0;
  document.addEventListener("mousemove", (e) => {
    const w = innerWidth / 2, h = innerHeight / 2;
    tx = ((e.clientX - w) / w) * 10;
    ty = ((e.clientY - h) / h) * 6;
  });
  function tick() {
    cx += (tx - cx) * 0.06;
    cy += (ty - cy) * 0.06;
    photo!.style.transform = `translate(${cx}px, ${cy}px) scale(1.04)`;
    requestAnimationFrame(tick);
  }
  tick();
}

// Fade scroll indicator after a bit of scroll
function initScrollIndicatorFade() {
  const indicator = document.getElementById("scroll-indicator");
  if (!indicator) return;
  addEventListener(
    "scroll",
    () => {
      if (scrollY > 60) indicator.style.opacity = "0";
    },
    { passive: true }
  );
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => {
    activate();
    initCounters();
    initHeroParallax();
    initScrollIndicatorFade();
  });
} else {
  activate();
  initCounters();
  initHeroParallax();
  initScrollIndicatorFade();
}
