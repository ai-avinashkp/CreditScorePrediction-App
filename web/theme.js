"use strict";
// Loaded synchronously in <head> so the saved theme applies before first paint (no flash).
(function () {
  const KEY = "theme";
  const root = document.documentElement;
  const read = () => { try { return localStorage.getItem(KEY); } catch { return null; } };
  const write = (v) => { try { localStorage.setItem(KEY, v); } catch { /* storage blocked: stay per-session */ } };
  const systemDark = () => window.matchMedia("(prefers-color-scheme: dark)").matches;
  const current = () => root.dataset.theme || (systemDark() ? "dark" : "light");

  const saved = read();
  if (saved === "light" || saved === "dark") root.dataset.theme = saved;

  function syncButtons() {
    const dark = current() === "dark";
    document.querySelectorAll("[data-theme-toggle]").forEach((b) => {
      b.setAttribute("aria-pressed", String(dark));
      b.textContent = dark ? "☀ Light" : "☾ Dark";
      b.setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
    });
  }

  document.addEventListener("click", (e) => {
    if (!e.target.closest("[data-theme-toggle]")) return;
    const next = current() === "dark" ? "light" : "dark";
    root.dataset.theme = next;
    write(next);
    syncButtons();
    document.dispatchEvent(new Event("themechange"));
  });

  document.addEventListener("DOMContentLoaded", syncButtons);
})();
