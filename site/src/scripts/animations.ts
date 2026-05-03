/**
 * Portfolio animations — Engineer's Field Notebook v2.0
 * Stack: Lenis smooth scroll + GSAP scroll-driven reveals
 */

import Lenis from "lenis";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

gsap.registerPlugin(ScrollTrigger);

// ─── 1. LENIS SMOOTH SCROLL ──────────────────────────────────────────────────
const lenis = new Lenis({
  duration: 1.2,
  easing: (t: number) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
  orientation: "vertical",
  smoothWheel: true,
});

function raf(time: number) {
  lenis.raf(time);
  requestAnimationFrame(raf);
}
requestAnimationFrame(raf);

// Sync Lenis with GSAP ScrollTrigger
lenis.on("scroll", ScrollTrigger.update);
gsap.ticker.add((time: number) => {
  lenis.raf(time * 1000);
});
gsap.ticker.lagSmoothing(0);

// ─── 2. HERO ANIMATIONS (staggered on load) ──────────────────────────────────
function initHeroAnimations() {
  const wordFadeEls = document.querySelectorAll<HTMLElement>(".word-fade");
  const idCard = document.getElementById("id-card");
  const heroSub = document.getElementById("hero-sub");
  const heroCta = document.getElementById("hero-cta");
  const scrollIndicator = document.getElementById("scroll-indicator");

  if (wordFadeEls.length > 0) {
    // Animate each title line in sequence
    gsap.to(wordFadeEls, {
      opacity: 1,
      y: 0,
      duration: 0.9,
      stagger: 0.15,
      ease: "power3.out",
      delay: 0.3,
    });
  }

  if (idCard) {
    gsap.to(idCard, {
      opacity: 1,
      duration: 0.7,
      ease: "power2.out",
      delay: 0.6,
    });
  }

  if (heroSub) {
    gsap.to(heroSub, {
      opacity: 1,
      y: 0,
      duration: 0.7,
      ease: "power2.out",
      delay: 1.0,
    });
  }

  if (heroCta) {
    gsap.to(heroCta, {
      opacity: 1,
      y: 0,
      duration: 0.6,
      ease: "power2.out",
      delay: 1.25,
    });
  }

  if (scrollIndicator) {
    gsap.to(scrollIndicator, {
      opacity: 1,
      duration: 0.8,
      ease: "power1.out",
      delay: 1.6,
    });
  }
}

// ─── 3. MOUSE PARALLAX ON HERO PHOTO ─────────────────────────────────────────
function initHeroParallax() {
  const photo = document.getElementById("hero-photo") as HTMLImageElement | null;
  if (!photo) return;

  let targetX = 0;
  let targetY = 0;
  let currentX = 0;
  let currentY = 0;

  document.addEventListener("mousemove", (e: MouseEvent) => {
    const cx = window.innerWidth / 2;
    const cy = window.innerHeight / 2;
    targetX = ((e.clientX - cx) / cx) * 14;
    targetY = ((e.clientY - cy) / cy) * 8;
  });

  function animateParallax() {
    currentX += (targetX - currentX) * 0.06;
    currentY += (targetY - currentY) * 0.06;
    photo!.style.transform = `translate(${currentX}px, ${currentY}px) scale(1.04)`;
    requestAnimationFrame(animateParallax);
  }
  animateParallax();
}

// ─── 4. SCROLL REVEALS ───────────────────────────────────────────────────────
function initScrollReveals() {
  const revealEls = document.querySelectorAll<HTMLElement>(".reveal");

  revealEls.forEach((el) => {
    gsap.fromTo(
      el,
      { opacity: 0, y: 28 },
      {
        opacity: 1,
        y: 0,
        duration: 0.75,
        ease: "power2.out",
        scrollTrigger: {
          trigger: el,
          start: "top 88%",
          once: true,
        },
      }
    );
  });
}

// ─── 5. COUNTER ANIMATIONS ───────────────────────────────────────────────────
function initCounters() {
  const counters = document.querySelectorAll<HTMLElement>("[data-count]");

  counters.forEach((el) => {
    const target = parseInt(el.dataset.count ?? "0", 10);

    ScrollTrigger.create({
      trigger: el,
      start: "top 90%",
      once: true,
      onEnter: () => {
        gsap.fromTo(
          el,
          { innerText: 0 },
          {
            innerText: target,
            duration: 1.2,
            ease: "power2.out",
            snap: { innerText: 1 },
            onUpdate() {
              el.textContent = Math.round(
                parseFloat(el.textContent ?? "0")
              ).toString();
            },
          }
        );
      },
    });
  });
}

// ─── 6. HIDE SCROLL INDICATOR ON SCROLL ──────────────────────────────────────
function initScrollIndicatorFade() {
  const indicator = document.getElementById("scroll-indicator");
  if (!indicator) return;

  lenis.on("scroll", ({ scroll }: { scroll: number }) => {
    if (scroll > 60) {
      gsap.to(indicator, { opacity: 0, duration: 0.4, ease: "power1.out" });
    }
  });
}

// ─── BOOT ─────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  initHeroAnimations();
  initHeroParallax();
  initScrollReveals();
  initCounters();
  initScrollIndicatorFade();
});
