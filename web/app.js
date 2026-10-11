"use strict";

// Labels, groups, units and hints come from /api/schema (creditrisk/features.py), so the
// form always matches what the model was trained on. Only display formatting lives here.
const ID_PREFIX = { branch_id: "Branch", state_id: "State", manufacturer_id: "Manufacturer" };
const TOP_N = 8;

const $ = (id) => document.getElementById(id);
const state = { schema: null, features: [], sampleId: null, factors: [], showAll: false, mode: "login" };

const inr = (v) => `₹${Math.round(Number(v)).toLocaleString("en-IN")}`;
const label = (f) => state.schema[f]?.label || f;
const optLabel = (f, v) => (ID_PREFIX[f] ? `${ID_PREFIX[f]} ${v}` : String(v));

// ---------- http ----------
async function api(path, { method = "GET", body } = {}) {
  const res = await fetch(path, {
    method, credentials: "same-origin",
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  let data = null;
  try { data = await res.json(); } catch { /* empty body */ }
  if (!res.ok) {
    const err = new Error(errorText(data) || `Request failed (${res.status})`);
    err.status = res.status;
    throw err;
  }
  return data;
}
function errorText(d) {
  if (!d || d.detail == null) return "";
  if (typeof d.detail === "string") return d.detail;
  if (Array.isArray(d.detail)) return d.detail.map((e) => `${label((e.loc || []).slice(-1)[0])}: ${e.msg}`).join("; ");
  return "";
}

// ---------- auth ----------
function setMode(mode) {
  state.mode = mode;
  const reg = mode === "register";
  $("tab-login").classList.toggle("is-on", !reg); $("tab-register").classList.toggle("is-on", reg);
  $("tab-login").setAttribute("aria-selected", String(!reg)); $("tab-register").setAttribute("aria-selected", String(reg));
  $("auth-submit").textContent = reg ? "Create account" : "Sign in";
  $("auth-hint").hidden = !reg;
  $("auth-pass").autocomplete = reg ? "new-password" : "current-password";
  $("auth-error").textContent = "";
}
$("tab-login").addEventListener("click", () => setMode("login"));
$("tab-register").addEventListener("click", () => setMode("register"));

$("auth-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const btn = $("auth-submit"); btn.disabled = true; $("auth-error").textContent = "";
  try {
    await api(`/api/auth/${state.mode}`, { method: "POST", body: { username: $("auth-user").value, password: $("auth-pass").value } });
    $("auth-pass").value = "";
    await enterApp();
  } catch (err) { $("auth-error").textContent = err.message; }
  finally { btn.disabled = false; }
});
$("logout").addEventListener("click", async () => {
  try { await api("/api/auth/logout", { method: "POST" }); } catch { /* ignore */ }
  showAuth();
});
function showAuth() { $("app-view").hidden = true; $("auth-view").hidden = false; $("auth-user").focus(); }

// ---------- form ----------
function makeInput(name, spec) {
  let input;
  if (spec.kind === "numeric") {
    input = document.createElement("input");
    input.type = "number"; input.inputMode = "decimal"; input.step = spec.unit === "inr" ? 1000 : "any";
    if (spec.min >= 0) input.min = 0;
    input.placeholder = spec.unit === "inr" ? `${inr(spec.min)} – ${inr(spec.max)}` : `${spec.min} – ${spec.max}`;
    // Scrolling over a focused number field would change its value; drop focus so the page scrolls instead.
    input.addEventListener("wheel", () => { if (document.activeElement === input) input.blur(); }, { passive: true });
  } else if (spec.kind === "flag") {
    input = document.createElement("input");
    input.type = "checkbox";
  } else {
    input = document.createElement("select");
    for (const v of spec.options) input.add(new Option(optLabel(name, v), v));
  }
  input.id = `f-${name}`; input.name = name;
  input.addEventListener(spec.kind === "flag" ? "change" : "input", () => {
    clearSample(); input.removeAttribute("aria-invalid"); markStale();
  });
  return input;
}

function buildForm() {
  const host = $("form-groups"); host.replaceChildren();
  const groups = [...new Set(state.features.map((f) => state.schema[f].group))];
  for (const title of groups) {
    const fs = document.createElement("fieldset"); fs.className = "card group";  // one card per section
    const lg = document.createElement("legend"); lg.textContent = title; fs.append(lg);
    const grid = document.createElement("div"); grid.className = "grid2";
    const flags = document.createElement("div"); flags.className = "flags";
    for (const name of state.features.filter((f) => state.schema[f].group === title)) {
      const spec = state.schema[name];
      const input = makeInput(name, spec);
      const wrap = document.createElement("label");
      const span = document.createElement("span");
      span.textContent = spec.unit === "inr" ? `${spec.label} (₹)` : spec.label;
      if (spec.kind === "flag") {
        wrap.className = "check"; wrap.append(input, span); flags.append(wrap); continue;
      }
      wrap.className = "field"; wrap.append(span, input);
      if (spec.hint) {
        const h = document.createElement("small"); h.className = "muted"; h.id = `h-${name}`; h.textContent = spec.hint;
        input.setAttribute("aria-describedby", h.id); wrap.append(h);
      }
      grid.append(wrap);
    }
    fs.append(grid);
    if (flags.childElementCount) fs.append(flags);
    host.append(fs);
  }
  setDefaults();
}

function setDefaults() {
  for (const f of state.features) {
    const el = $(`f-${f}`), s = state.schema[f];
    if (s.kind === "numeric") el.value = Math.round(s.median * 100) / 100;
    else if (s.kind === "flag") el.checked = s.default === "1";
    else el.value = s.default;
  }
}
function readForm() {
  const out = {};
  for (const f of state.features) {
    const el = $(`f-${f}`), kind = state.schema[f].kind;
    out[f] = kind === "numeric" ? (el.value === "" ? null : Number(el.value)) : kind === "flag" ? el.checked : el.value;
  }
  return out;
}
function fillForm(values) {
  for (const [k, v] of Object.entries(values)) {
    const el = $(`f-${k}`); if (!el) continue;
    if (state.schema[k].kind === "flag") el.checked = Boolean(v); else el.value = v;
  }
}
function clearSample() { if (state.sampleId !== null) { state.sampleId = null; $("sample-select").value = ""; } }
// An old result must never sit next to different inputs: hide it until the user scores again.
function markStale() {
  $("result").hidden = true; $("why-card").hidden = true; lastGauge = null;
  $("empty").hidden = false;
  $("empty").replaceChildren("Inputs changed. Press ", Object.assign(document.createElement("strong"), { textContent: "Score applicant" }), " to run the model.");
}
function resetForm() { setDefaults(); clearSample(); $("form-error").textContent = ""; markStale(); }
$("reset-btn").addEventListener("click", resetForm);

// Loading a hold-out case only fills the form; scoring always needs an explicit click.
$("sample-select").addEventListener("change", async (e) => {
  const id = Number(e.target.value);
  if (!id) { state.sampleId = null; return; }
  try {
    const s = await api(`/api/samples/${id}`);
    fillForm(s.applicant); state.sampleId = id;
    $("form-error").textContent = "";
    markStale();
  } catch (err) { $("form-error").textContent = err.message; }
});

$("score-form").addEventListener("submit", (e) => { e.preventDefault(); score(); });

async function score() {
  const payload = readForm();
  let bad = false;
  for (const f of state.features) {
    const s = state.schema[f]; if (s.kind !== "numeric") continue;
    const v = payload[f], el = $(`f-${f}`);
    const invalid = v === null || Number.isNaN(v) || (s.min >= 0 && v < 0);
    if (invalid) { el.setAttribute("aria-invalid", "true"); bad = true; } else el.removeAttribute("aria-invalid");
  }
  if (bad) { $("form-error").textContent = "Please complete the highlighted fields with valid numbers."; return; }
  $("form-error").textContent = "";
  if (state.sampleId) payload.sample_id = state.sampleId;

  const btn = $("score-btn"); btn.disabled = true; btn.textContent = "Scoring…";
  try { renderResult(await api("/api/predict", { method: "POST", body: payload })); }
  catch (err) {
    if (err.status === 401) return showAuth();
    $("form-error").textContent = err.message;
  } finally { btn.disabled = false; btn.textContent = "Score applicant"; }
}

// ---------- result ----------
const pct = (x) => `${(x * 100).toFixed(1)}%`;
const COLORS = { APPROVE: "--good-bar", REVIEW: "#e0a030", DECLINE: "--bad-bar" };

function renderGauge(pd, decision) {
  const NS = "http://www.w3.org/2000/svg", svg = $("gauge"); svg.replaceChildren();
  const cx = 100, cy = 100, r = 80, len = Math.PI * r;
  const d = `M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`;
  const mk = (attrs) => { const p = document.createElementNS(NS, "path"); p.setAttribute("d", d); p.setAttribute("fill", "none");
    p.setAttribute("stroke-linecap", "round"); for (const [k, v] of Object.entries(attrs)) p.setAttribute(k, v); svg.append(p); return p; };
  const css = getComputedStyle(document.documentElement);
  mk({ "stroke-width": 18, stroke: css.getPropertyValue("--dark").trim(), opacity: 0.45 });
  const key = COLORS[decision]; const color = key.startsWith("--") ? css.getPropertyValue(key).trim() : key;
  const fill = mk({ "stroke-width": 18, stroke: color, "stroke-dasharray": `0 ${len}` });
  fill.style.transition = "stroke-dasharray .7s cubic-bezier(.2,.8,.2,1)";
  requestAnimationFrame(() => fill.setAttribute("stroke-dasharray", `${Math.max(pd, 0.01) * len} ${len}`));
}

// The gauge reads theme colours when drawn, so redraw it after a theme switch.
let lastGauge = null;
document.addEventListener("themechange", () => { if (lastGauge) renderGauge(...lastGauge); });

function renderResult(r) {
  lastGauge = [r.probability_of_default, r.decision];
  $("empty").hidden = true; $("result").hidden = false; $("why-card").hidden = false;
  $("pd-val").textContent = pct(r.probability_of_default);
  $("score-val").textContent = r.score;
  const dec = $("decision"); dec.textContent = r.decision; dec.className = `badge ${r.decision}`;
  renderGauge(r.probability_of_default, r.decision);

  const act = $("actual");
  if (r.actual_default !== undefined) {
    const defaulted = r.actual_default === 1;
    act.hidden = false;
    act.replaceChildren("Actual outcome: ");
    const b = document.createElement("b"); b.className = defaulted ? "no" : "ok";
    b.textContent = defaulted ? "missed first EMI" : "paid first EMI"; act.append(b);
  } else act.hidden = true;

  $("why-sub").textContent =
    `Portfolio baseline is ${pct(r.base_probability)}. Each bar shows how much a feature pushed this applicant above or below it (log-odds).`;
  state.factors = r.factors; state.showAll = false; renderFactors();
  $("req-id").textContent = `Audit ID ${r.request_id} · model ${r.model_version}`;
}

function fmtValue(name, v) {
  const s = state.schema[name];
  if (s.kind === "flag") return v ? "Yes" : "No";
  if (s.kind === "numeric") return s.unit === "inr" ? inr(v) : String(Math.round(Number(v) * 100) / 100);
  return optLabel(name, v);
}

function renderFactors() {
  const list = $("factors"); list.replaceChildren();
  const shown = state.showAll ? state.factors : state.factors.slice(0, TOP_N);
  const max = Math.max(...state.factors.map((f) => Math.abs(f.contribution)), 0.001);
  for (const f of shown) {
    const li = document.createElement("li"); li.className = "factor";
    const head = document.createElement("div"); head.className = "factor-head";
    const name = document.createElement("b"); name.textContent = label(f.feature);
    const val = document.createElement("span"); val.textContent = fmtValue(f.feature, f.value);
    head.append(name, val);
    const track = document.createElement("div"); track.className = "track";
    const bar = document.createElement("div"); bar.className = `bar ${f.contribution > 0 ? "up" : "down"}`;
    bar.style.width = "0%"; track.append(bar);
    requestAnimationFrame(() => requestAnimationFrame(() => { bar.style.width = `${(Math.abs(f.contribution) / max) * 50}%`; }));
    const meta = document.createElement("div"); meta.className = "factor-meta";
    // Numeric inputs are grouped into WoE bins by the model; show the bin so the score is traceable.
    const s = state.schema[f.feature];
    const range = s.unit === "inr" ? f.bin.replace(/-?[\d.]+(e[+-]?\d+)?/g, (n) => inr(n)) : f.bin;
    const a = document.createElement("span");
    a.textContent = (s.kind === "numeric" ? `Model range: ${range} · ` : "") + `WoE ${f.woe.toFixed(2)}`;
    const c = document.createElement("span"); c.textContent = `${f.contribution > 0 ? "+" : ""}${f.contribution.toFixed(2)}`;
    meta.append(a, c);
    li.append(head, track, meta);
    li.setAttribute("aria-label", `${label(f.feature)} ${f.direction === "increases_risk" ? "raises" : "lowers"} risk by ${Math.abs(f.contribution).toFixed(2)}`);
    list.append(li);
  }
  const t = $("toggle-factors"); t.hidden = state.factors.length <= TOP_N;
  t.textContent = state.showAll ? "Show top factors only" : `Show all ${state.factors.length} factors`;
}
$("toggle-factors").addEventListener("click", () => { state.showAll = !state.showAll; renderFactors(); });

// ---------- model card ----------
function barList(host, rows, { max, fmt, highlight } = {}) {
  host.replaceChildren();
  const top = max ?? Math.max(...rows.map((r) => r.value), 0.001);
  for (const r of rows) {
    const li = document.createElement("li");
    if (highlight && highlight(r)) li.className = "is-best";
    const n = document.createElement("span"); n.textContent = r.name;
    const tr = document.createElement("div"); tr.className = "track";
    const bar = document.createElement("div"); bar.className = "bar"; bar.style.width = `${Math.max(r.value / top, 0) * 100}%`; tr.append(bar);
    const v = document.createElement("span"); v.className = "num"; v.textContent = fmt ? fmt(r.value) : r.value.toFixed(2);
    li.append(n, tr, v); host.append(li);
  }
}

function renderModel(m) {
  $("model-chip").textContent = `model ${m.model_version || "?"}`;
  $("model-note").textContent = [m.dataset, m.target && `Target: ${m.target}`, m.validation].filter(Boolean).join(" · ");
  const items = [["OOT AUC", m.test_auc], ["Gini", m.test_gini], ["KS", m.test_ks],
                 ["CV AUC", m.cv_auc_mean != null ? `${m.cv_auc_mean} ±${m.cv_auc_std}` : null],
                 ["Train / test", m.n_train ? `${m.n_train.toLocaleString("en-IN")} / ${m.n_test.toLocaleString("en-IN")}` : null],
                 ["Default rate", m.base_default_rate != null ? pct(m.base_default_rate) : null]];
  const dl = $("model-card"); dl.replaceChildren();
  for (const [k, v] of items) {
    if (v == null) continue;
    const d = document.createElement("div"), dt = document.createElement("dt"), dd = document.createElement("dd");
    dt.textContent = k; dd.textContent = v; d.append(dt, dd); dl.append(d);
  }
  // AUC bars start at 0.5 (random ranking) so the differences between models are visible.
  const base = Object.entries(m.baselines || {}).map(([name, s]) => ({ name, value: s.auc - 0.5, auc: s.auc }));
  const best = Math.max(...base.map((b) => b.value));
  barList($("baseline-list"), base, { max: best, fmt: (v) => (v + 0.5).toFixed(3), highlight: (r) => r.value === best });
  barList($("iv-list"), (m.top_iv || []).map((x) => ({ name: label(x.feature), value: x.iv })));
}

// ---------- boot ----------
async function enterApp() {
  const me = await api("/api/auth/me");
  const [schema, samples, model] = await Promise.all([api("/api/schema"), api("/api/samples"), api("/api/model")]);
  state.schema = schema.schema; state.features = schema.features;
  $("user-chip").textContent = me.username;
  buildForm();
  const sel = $("sample-select"); sel.replaceChildren(new Option("Custom applicant (enter details below)", ""));
  for (const s of samples.samples) {
    sel.add(new Option(`#${s.sample_id} · ${inr(s.disbursed_amount)} · LTV ${s.ltv}% · age ${s.age} · ${s.bureau_band}`, s.sample_id));
  }
  renderModel(model);
  $("result").hidden = true; $("why-card").hidden = true; $("empty").hidden = false;
  $("empty").replaceChildren("Fill in an applicant and press ", Object.assign(document.createElement("strong"), { textContent: "Score applicant" }), ".");
  $("auth-view").hidden = true; $("app-view").hidden = false;
}

(async function boot() {
  try { await enterApp(); } catch { showAuth(); }
})();
