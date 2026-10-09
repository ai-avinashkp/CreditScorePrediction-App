"use strict";

// ---------- presentation metadata ----------
const LABELS = {
  checking_status: "Checking account", savings_status: "Savings", credit_history: "Credit history",
  other_payment_plans: "Other payment plans", existing_credits: "Existing credits at bank",
  purpose: "Loan purpose", credit_amount: "Credit amount (₹)", duration: "Duration (months)",
  installment_commitment: "Installment rate level", age: "Age (years)", employment: "Employed for",
  job: "Job type", housing: "Housing", residence_since: "Years at residence", num_dependents: "Dependents",
  property_magnitude: "Property", own_telephone: "Telephone", foreign_worker: "Foreign worker",
  other_parties: "Other debtors / guarantors",
};
const GROUPS = [
  ["Accounts & history", ["checking_status", "savings_status", "credit_history", "other_payment_plans", "existing_credits"]],
  ["Loan request", ["purpose", "credit_amount", "duration", "installment_commitment"]],
  ["Applicant", ["age", "employment", "job", "housing", "residence_since", "num_dependents",
                 "property_magnitude", "own_telephone", "foreign_worker", "other_parties"]],
];
// Staff-friendly wording for the dataset's coded values, in a logical order (best/most-secure first where it applies).
// Keys are the raw values the model/API use; they are never sent in the display form.
const OPTIONS = {
  // {n} tokens are DM thresholds from the dataset, rendered in INR at lookup time (see money()).
  checking_status: [["<0", "Overdrawn (balance below ₹0)"], ["0<=X<200", "Balance ₹0 to under {200}"],
                    [">=200", "Balance {200} or more"], ["no checking", "No checking account"]],
  savings_status: [["<100", "Under {100}"], ["100<=X<500", "{100} to under {500}"], ["500<=X<1000", "{500} to under {1000}"],
                   [">=1000", "{1000} or more"], ["no known savings", "No savings account / unknown"]],
  credit_history: [["no credits/all paid", "No credit taken / all repaid on time"], ["all paid", "All credits at this bank repaid on time"],
                   ["existing paid", "Existing credits repaid on time so far"], ["delayed previously", "Past delays in repayment"],
                   ["critical/other existing credit", "Critical account / credit elsewhere"]],
  purpose: [["new car", "New car"], ["used car", "Used car"], ["furniture/equipment", "Furniture / equipment"],
            ["radio/tv", "Radio / TV"], ["domestic appliance", "Domestic appliance"], ["repairs", "Repairs"],
            ["education", "Education"], ["retraining", "Retraining"], ["business", "Business"], ["other", "Other"]],
  employment: [["unemployed", "Unemployed"], ["<1", "Less than 1 year"], ["1<=X<4", "1 to under 4 years"],
               ["4<=X<7", "4 to under 7 years"], [">=7", "7 years or more"]],
  other_payment_plans: [["none", "None"], ["bank", "With another bank"], ["stores", "With stores"]],
  property_magnitude: [["real estate", "Real estate"], ["life insurance", "Life insurance / savings agreement"],
                       ["car", "Car or other"], ["no known property", "No known property"]],
  housing: [["own", "Owns home"], ["rent", "Rents"], ["for free", "Lives rent-free"]],
  job: [["high qualif/self emp/mgmt", "Highly qualified / self-employed / management"], ["skilled", "Skilled employee / official"],
        ["unskilled resident", "Unskilled (resident)"], ["unemp/unskilled non res", "Unemployed / unskilled (non-resident)"]],
  other_parties: [["none", "None"], ["co applicant", "Co-applicant"], ["guarantor", "Guarantor"]],
  own_telephone: [["yes", "Yes"], ["none", "No"]],
  foreign_worker: [["no", "No"], ["yes", "Yes"]],
};
const HINTS = {
  installment_commitment: "Scale 1 – 4, as recorded in the dataset",
  residence_since: "Scale 1 – 4: 1 = under a year … 4 = four years or more",
  existing_credits: "Number of credits already held at this bank",
};
// Model, API and audit log all work in DM; INR is display/entry only. Rate comes from /api/schema.
const money = (dm) => `${state.currency.symbol}${Math.round(dm * state.currency.per_dm).toLocaleString("en-IN")}`;
const toDm = (inr) => inr / state.currency.per_dm;
const optLabel = (feat, raw) =>
  ((OPTIONS[feat]?.find(([v]) => v === raw) || [, raw])[1]).replace(/\{(\d+)\}/g, (_, n) => money(Number(n)));
const TOP_N = 7;

const $ = (id) => document.getElementById(id);
const state = { currency: { symbol: "₹", per_dm: 90 }, schema: null, features: [], sampleId: null, factors: [], showAll: false, mode: "login" };

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
  if (Array.isArray(d.detail)) return d.detail.map((e) => `${(e.loc || []).slice(-1)[0]}: ${e.msg}`).join("; ");
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
function buildForm() {
  const host = $("form-groups"); host.replaceChildren();
  for (const [title, names] of GROUPS) {
    const fs = document.createElement("fieldset"); fs.className = "card group";  // one card per section
    const lg = document.createElement("legend"); lg.textContent = title; fs.append(lg);
    const grid = document.createElement("div"); grid.className = "grid2";
    for (const name of names.filter((n) => state.schema[n])) {
      const spec = state.schema[name];
      const label = document.createElement("label"); label.className = "field";
      const span = document.createElement("span"); span.textContent = LABELS[name] || name; label.append(span);
      let input;
      if (spec.type === "numeric") {
        input = document.createElement("input");
        input.type = "number"; input.inputMode = "numeric"; input.min = 0; input.step = 1; input.value = spec.median;
        input.placeholder = `${spec.min} – ${spec.max}`;
        if (name === "credit_amount") {  // entered in INR, sent to the model in DM (see readForm)
          input.value = Math.round(spec.median * state.currency.per_dm); input.step = 1000;
          input.placeholder = `${money(spec.min)} – ${money(spec.max)}`;
        }
        // Scrolling over a focused number field would change its value; drop focus so the page scrolls instead.
        input.addEventListener("wheel", () => { if (document.activeElement === input) input.blur(); }, { passive: true });
      } else {
        input = document.createElement("select");
        // Show staff-friendly wording, keep the raw code as the value; unknown codes fall back to themselves.
        const known = (OPTIONS[name] || []).map(([v]) => v).filter((v) => spec.options.includes(v));
        const order = [...known, ...spec.options.filter((v) => !known.includes(v))];
        for (const v of order) input.add(new Option(optLabel(name, v), v));
      }
      input.id = `f-${name}`; input.name = name; input.required = true;
      input.addEventListener("input", () => { clearSample(); input.removeAttribute("aria-invalid"); markStale(); });
      label.append(input);
      const hint = name === "credit_amount"
        ? `Model was trained on 1990s German data (DM). Amounts are scaled at ${state.currency.symbol}${state.currency.per_dm} = 1 DM, a purchasing-power estimate.`
        : HINTS[name];
      if (hint) {
        const h = document.createElement("small"); h.className = "muted"; h.textContent = hint;
        input.setAttribute("aria-describedby", `h-${name}`); h.id = `h-${name}`; label.append(h);
      }
      grid.append(label);
    }
    fs.append(grid); host.append(fs);
  }
}
function readForm() {
  const out = {};
  for (const f of state.features) {
    const el = $(`f-${f}`);
    let v = state.schema[f].type === "numeric" ? (el.value === "" ? null : Number(el.value)) : el.value;
    if (f === "credit_amount" && v !== null) v = toDm(v);  // model works in DM
    out[f] = v;
  }
  return out;
}
function fillForm(values) {
  for (const [k, v] of Object.entries(values)) {
    const el = $(`f-${k}`);
    if (el) el.value = k === "credit_amount" ? Math.round(v * state.currency.per_dm) : v;
  }
}
function clearSample() { if (state.sampleId !== null) { state.sampleId = null; $("sample-select").value = ""; } }
// An old result must never sit next to different inputs: hide it until the user scores again.
function markStale() {
  $("result").hidden = true; $("why-card").hidden = true; lastGauge = null;
  $("empty").hidden = false;
  $("empty").replaceChildren("Inputs changed. Press ", Object.assign(document.createElement("strong"), { textContent: "Score applicant" }), " to run the model.");
}
function resetForm() {
  for (const f of state.features) {
    const el = $(`f-${f}`), s = state.schema[f];
    if (f === "credit_amount") el.value = Math.round(s.median * state.currency.per_dm);
    else if (s.type === "numeric") el.value = s.median; else el.selectedIndex = 0;
  }
  clearSample(); $("form-error").textContent = "";
  markStale();
}
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
    const el = $(`f-${f}`);
    const invalid = payload[f] === null || Number.isNaN(payload[f]) || payload[f] < 0;
    el.toggleAttribute("aria-invalid", invalid); if (invalid) { el.setAttribute("aria-invalid", "true"); bad = true; }
  }
  if (bad) { $("form-error").textContent = "Please complete all fields with valid, non-negative numbers."; return; }
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
  const track = mk({ "stroke-width": 18, stroke: css.getPropertyValue("--dark").trim(), opacity: 0.45 });
  const key = COLORS[decision]; const color = key.startsWith("--") ? css.getPropertyValue(key).trim() : key;
  const fill = mk({ "stroke-width": 18, stroke: color, "stroke-dasharray": `0 ${len}` });
  fill.style.transition = "stroke-dasharray .7s cubic-bezier(.2,.8,.2,1)";
  requestAnimationFrame(() => fill.setAttribute("stroke-dasharray", `${Math.max(pd, 0.01) * len} ${len}`));
  track.setAttribute("data-role", "track");
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
    b.textContent = defaulted ? "defaulted" : "repaid"; act.append(b);
  } else act.hidden = true;

  $("why-sub").textContent =
    `Portfolio baseline is ${pct(r.base_probability)}. Each bar shows how much a feature pushed this applicant above or below it (log-odds).`;
  state.factors = r.factors; state.showAll = false; renderFactors();
  $("req-id").textContent = `Audit ID ${r.request_id} · model ${r.model_version}`;
}

function fmtValue(name, v) {
  if (state.schema[name].type === "numeric") return name === "credit_amount" ? money(Number(v)) : String(v);
  return optLabel(name, String(v));
}

function renderFactors() {
  const list = $("factors"); list.replaceChildren();
  const shown = state.showAll ? state.factors : state.factors.slice(0, TOP_N);
  const max = Math.max(...state.factors.map((f) => Math.abs(f.contribution)), 0.001);
  for (const f of shown) {
    const li = document.createElement("li"); li.className = "factor";
    const head = document.createElement("div"); head.className = "factor-head";
    const name = document.createElement("b"); name.textContent = LABELS[f.feature] || f.feature;
    const val = document.createElement("span"); val.textContent = fmtValue(f.feature, f.value);
    head.append(name, val);
    const track = document.createElement("div"); track.className = "track";
    const bar = document.createElement("div"); bar.className = `bar ${f.contribution > 0 ? "up" : "down"}`;
    bar.style.width = "0%"; track.append(bar);
    requestAnimationFrame(() => requestAnimationFrame(() => { bar.style.width = `${(Math.abs(f.contribution) / max) * 50}%`; }));
    const meta = document.createElement("div"); meta.className = "factor-meta";
    // Numeric values are grouped into ranges by the model; categories already read clearly in the header.
    const numeric = state.schema[f.feature].type === "numeric";
    const a = document.createElement("span");
    const range = f.feature === "credit_amount" ? f.bin.replace(/[\d.]+/g, (n) => money(Number(n))) : f.bin;
    a.textContent = (numeric ? `Model range: ${range} · ` : "") + `WoE ${f.woe.toFixed(2)}`;
    const c = document.createElement("span"); c.textContent = `${f.contribution > 0 ? "+" : ""}${f.contribution.toFixed(2)}`;
    meta.append(a, c);
    li.append(head, track, meta);
    li.setAttribute("aria-label", `${LABELS[f.feature] || f.feature} ${f.direction === "increases_risk" ? "raises" : "lowers"} risk by ${Math.abs(f.contribution).toFixed(2)}`);
    list.append(li);
  }
  const t = $("toggle-factors"); t.hidden = state.factors.length <= TOP_N;
  t.textContent = state.showAll ? "Show top factors only" : `Show all ${state.factors.length} factors`;
}
$("toggle-factors").addEventListener("click", () => { state.showAll = !state.showAll; renderFactors(); });

// ---------- model card ----------
function renderModel(m) {
  $("model-chip").textContent = `model ${m.model_version || "?"}`;
  const items = [["Test AUC", m.test_auc], ["Gini", m.test_gini], ["KS", m.test_ks],
                 ["CV AUC", m.cv_auc_mean != null ? `${m.cv_auc_mean} ±${m.cv_auc_std}` : null],
                 ["Train / test", m.n_train ? `${m.n_train} / ${m.n_test}` : null],
                 ["Base default", m.base_default_rate != null ? pct(m.base_default_rate) : null]];
  const dl = $("model-card"); dl.replaceChildren();
  for (const [k, v] of items) {
    if (v == null) continue;
    const d = document.createElement("div"), dt = document.createElement("dt"), dd = document.createElement("dd");
    dt.textContent = k; dd.textContent = v; d.append(dt, dd); dl.append(d);
  }
  const ul = $("iv-list"); ul.replaceChildren();
  const top = m.top_iv || [], maxIv = Math.max(...top.map((x) => x.iv), 0.001);
  for (const x of top) {
    const li = document.createElement("li");
    const n = document.createElement("span"); n.textContent = LABELS[x.feature] || x.feature;
    const tr = document.createElement("div"); tr.className = "track";
    const bar = document.createElement("div"); bar.className = "bar"; bar.style.width = `${(x.iv / maxIv) * 100}%`; tr.append(bar);
    const v = document.createElement("span"); v.className = "num"; v.textContent = x.iv.toFixed(2);
    li.append(n, tr, v); ul.append(li);
  }
}

// ---------- boot ----------
async function enterApp() {
  const me = await api("/api/auth/me");
  const [schema, samples, model] = await Promise.all([api("/api/schema"), api("/api/samples"), api("/api/model")]);
  state.schema = schema.schema; state.features = schema.features; state.currency = schema.currency;
  $("user-chip").textContent = me.username;
  buildForm();
  const sel = $("sample-select"); sel.replaceChildren(new Option("Custom applicant", ""));
  for (const s of samples.samples) {
    sel.add(new Option(`#${s.sample_id} · ${optLabel("purpose", s.purpose)} · ${money(s.credit_amount)} · ${s.duration}m · age ${s.age}`, s.sample_id));
  }
  renderModel(model);
  $("result").hidden = true; $("why-card").hidden = true; $("empty").hidden = false;
  $("empty").replaceChildren("Fill in an applicant and press ", Object.assign(document.createElement("strong"), { textContent: "Score applicant" }), ".");
  $("auth-view").hidden = true; $("app-view").hidden = false;
}

(async function boot() {
  try { await enterApp(); } catch { showAuth(); }
})();
