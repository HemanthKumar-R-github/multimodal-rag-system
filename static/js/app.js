"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (v, d = 3) => (v === null || v === undefined ? "—" : typeof v === "number" ? v.toFixed(d) : esc(v));

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: opts.body && !(opts.body instanceof FormData) ? { "Content-Type": "application/json" } : {},
    ...opts,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || (data.errors || []).join("; ") || `Request failed (${res.status})`);
  return data;
}

async function loadStatus() {
  try {
    const s = await api("/api/status");
    const b = s.backends;
    $("#status-pill").textContent = `${s.ready} docs · ${s.chunks} chunks · ${b.llm ? "LLM on" : "extractive"} · ${b.reranker.split(":")[0]}`;
    $("#status-pill").title = Object.entries(b).map(([k, v]) => `${k}: ${v ?? "off"}`).join("\n");
    return s;
  } catch {
    $("#status-pill").textContent = "offline";
  }
}

function renderMarkdownish(text) {
  const lines = esc(text).split("\n");
  let html = "", inList = false, table = [];
  const flushTable = () => { if (table.length) { html += `<pre>${table.join("\n")}</pre>`; table = []; } };
  for (const raw of lines) {
    const line = raw.trim();
    if (line.startsWith("|")) { table.push(line); continue; }
    flushTable();
    if (/^[-*] /.test(line)) {
      if (!inList) { html += "<ul>"; inList = true; }
      html += `<li>${line.slice(2)}</li>`;
      continue;
    }
    if (inList) { html += "</ul>"; inList = false; }
    if (line) html += `<p>${line}</p>`;
  }
  flushTable();
  if (inList) html += "</ul>";
  return html
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\[(\d+(?:\s*[,;]\s*\d+)*)\]/g, (_, g) =>
      g.split(/[,;]/).map((n) => `<a class="cite" href="#src-${n.trim()}" data-src="${n.trim()}">${n.trim()}</a>`).join(""));
}

function sourceCard(c) {
  const s = c.scores || {};
  const img = c.image ? `<img class="source-img" src="/api/assets/${encodeURI(c.image)}" alt="Extracted ${esc(c.modality)} from ${esc(c.filename)}${c.page ? ", page " + c.page : ""}" loading="lazy">` : "";
  return `<li class="source ${c.cited ? "cited" : ""}" id="src-${c.n}">
    <div class="source-head">
      <span class="source-n">[${c.n}]</span>
      <span class="source-title">${esc(c.filename)}${c.page ? ` · p.${c.page}` : ""}</span>
      <span class="badge ${esc(c.modality)}">${esc(c.modality)}</span>
      ${c.cited ? '<span class="badge ready">cited</span>' : ""}
    </div>
    <p class="source-snippet">${esc(c.snippet)}</p>${img}
    <div class="scores">
      <span>bm25 #${s.bm25_rank ?? "–"}</span><span>dense #${s.dense_rank ?? "–"}</span>
      <span>fused ${fmt(s.score, 4)}</span>${s.rerank_score != null ? `<span>rerank ${fmt(s.rerank_score, 3)}</span>` : ""}
    </div></li>`;
}

/* Ask page ------------------------------------------------------------ */
function initAsk() {
  const form = $("#ask-form"), btn = $("#ask-btn"), q = $("#question");
  q.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      if (e.nativeEvent?.isComposing || e.isComposing || e.keyCode === 229) return;
      e.preventDefault();
      form.requestSubmit();
    }
  });
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const question = q.value.trim();
    if (!question) return;
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner" aria-hidden="true"></span>Thinking';
    try {
      const r = await api("/api/query", {
        method: "POST",
        body: JSON.stringify({
          question,
          mode: form.querySelector("input[name=mode]:checked").value,
          rerank: $("#rerank").checked,
          use_llm: $("#use-llm").checked,
          top_k: Number($("#top-k").value),
        }),
      });
      $("#empty-state").hidden = true;
      $("#results").hidden = false;
      $("#answer").innerHTML = renderMarkdownish(r.answer);
      $("#answer-meta").textContent = `${r.method} · ${r.mode}${r.reranked ? " + rerank" : ""} · retrieval ${r.timings_ms.retrieval} ms · generation ${r.timings_ms.generation} ms`;
      $("#sources").innerHTML = r.citations.length ? r.citations.map(sourceCard).join("") : '<li class="muted">No sources retrieved.</li>';
    } catch (err) {
      $("#results").hidden = false;
      $("#answer").innerHTML = `<p class="msg-error">${esc(err.message)}</p>`;
      $("#sources").innerHTML = "";
    } finally {
      btn.disabled = false;
      btn.textContent = "Ask";
    }
  });
  $("#answer").addEventListener("click", (e) => {
    const a = e.target.closest(".cite");
    if (!a) return;
    const el = document.getElementById(`src-${a.dataset.src}`);
    if (el) { el.classList.add("flash"); setTimeout(() => el.classList.remove("flash"), 1200); }
  });
}

/* Documents page ------------------------------------------------------ */
function initDocuments() {
  const input = $("#file-input"), zone = $("#upload-form"), msg = $("#upload-msg");
  let pollTimer = null;

  async function refresh() {
    const docs = await api("/api/documents");
    $("#doc-rows").innerHTML = docs.length ? docs.map((d) => {
      const mods = Object.entries(d.stats || {}).filter(([k, v]) => ["text", "table", "image", "chart", "ocr"].includes(k) && v > 0)
        .map(([k, v]) => `<span class="badge ${k}">${k} ${v}</span>`).join("");
      return `<tr>
        <td><button class="btn-link" data-view="${d.id}">${esc(d.filename)}</button>${d.error ? `<div class="small msg-error">${esc(d.error)}</div>` : ""}</td>
        <td><span class="badge ${esc(d.status)}">${esc(d.status)}</span></td>
        <td class="num">${d.pages || "–"}</td><td class="num">${d.chunk_count}</td>
        <td><div class="modality-list">${mods || '<span class="muted small">—</span>'}</div></td>
        <td><button class="btn btn-danger" data-delete="${d.id}" aria-label="Delete ${esc(d.filename)}">Delete</button></td></tr>`;
    }).join("") : '<tr><td colspan="6" class="muted">No documents yet.</td></tr>';
    const pending = docs.some((d) => d.status === "queued" || d.status === "processing");
    clearTimeout(pollTimer);
    if (pending) pollTimer = setTimeout(refresh, 1500);
    loadStatus();
  }

  async function upload(files) {
    if (!files.length) return;
    const fd = new FormData();
    [...files].forEach((f) => fd.append("files", f));
    msg.className = "small";
    msg.innerHTML = '<span class="spinner" aria-hidden="true"></span>Uploading…';
    try {
      const r = await api("/api/documents", { method: "POST", body: fd });
      msg.className = "small msg-ok";
      msg.textContent = `Queued ${r.documents.length} file(s) for ingestion.${r.errors.length ? " Skipped: " + r.errors.join("; ") : ""}`;
    } catch (err) {
      msg.className = "small msg-error";
      msg.textContent = err.message;
    }
    input.value = "";
    refresh();
  }

  input.addEventListener("change", () => upload(input.files));
  ["dragenter", "dragover"].forEach((ev) => zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.remove("drag"); }));
  zone.addEventListener("drop", (e) => upload(e.dataTransfer.files));
  $("#refresh-docs").addEventListener("click", refresh);

  $("#doc-rows").addEventListener("click", async (e) => {
    const del = e.target.closest("[data-delete]");
    const view = e.target.closest("[data-view]");
    if (del && confirm("Delete this document and its index entries?")) {
      await api(`/api/documents/${del.dataset.delete}`, { method: "DELETE" });
      refresh();
    }
    if (view) {
      const r = await api(`/api/documents/${view.dataset.view}/chunks`);
      $("#chunk-dialog-title").textContent = `${r.document.filename} — ${r.chunks.length} chunks`;
      $("#chunk-list").innerHTML = r.chunks.map((c, i) => sourceCard({
        n: i + 1, filename: c.filename, page: c.page, modality: c.modality, snippet: c.content, image: c.meta.image, scores: {},
      }).replace('<div class="scores">', '<div class="scores" hidden>')).join("") || '<p class="muted">No chunks.</p>';
      $("#chunk-dialog").showModal();
    }
  });
  $("#close-dialog").addEventListener("click", () => $("#chunk-dialog").close());
  refresh();
}

/* Evaluate page ------------------------------------------------------- */
function initEvaluate() {
  const msg = $("#eval-msg");
  const setMsg = (text, cls = "") => { msg.className = `small ${cls}`; msg.innerHTML = text; };

  async function loadDatasets(select) {
    const ds = await api("/api/eval/datasets");
    $("#dataset").innerHTML = ds.map((d) => `<option value="${esc(d.file)}">${esc(d.name)} (${d.n})</option>`).join("") || "<option value=''>No datasets</option>";
    if (select) $("#dataset").value = select;
  }

  async function loadRuns() {
    const runs = await api("/api/eval/runs");
    $("#run-list").innerHTML = runs.length ? runs.map((r) => {
      const best = r.summary.retrieval?.["hybrid+rerank"] || Object.values(r.summary.retrieval || {})[0] || {};
      const k = r.summary.k;
      return `<li><button class="btn-link" data-run="${r.id}">#${r.id} · ${esc(r.dataset)}</button>
        <span class="muted"> · ${esc(r.created_at)} · ${r.summary.n_questions} q · recall@${k} ${fmt(best[`recall@${k}`])} · MRR ${fmt(best.mrr)}</span></li>`;
    }).join("") : '<li class="muted">No runs yet.</li>';
  }

  function render(run) {
    const { summary, details } = run;
    const k = summary.k;
    const cols = [`hit@${k}`, `recall@${k}`, `precision@${k}`, "mrr", `ndcg@${k}`, "latency_ms"];
    const rows = Object.entries(summary.retrieval);
    const best = Object.fromEntries(cols.map((c) => {
      const vals = rows.map(([, m]) => m[c]).filter((v) => v != null);
      return [c, c === "latency_ms" ? Math.min(...vals) : Math.max(...vals)];
    }));
    $("#retrieval-table").innerHTML = `<thead><tr><th scope="col">Config</th>${cols.map((c) => `<th scope="col" class="num">${c}</th>`).join("")}</tr></thead>
      <tbody>${rows.map(([name, m]) => `<tr><th scope="row">${esc(name)}</th>${cols.map((c) => {
        const v = m[c];
        const bar = c !== "latency_ms" && v != null ? `<span class="bar" style="width:${Math.round(v * 120)}px"></span>` : "";
        return `<td class="num ${v === best[c] ? "best" : ""}">${c === "latency_ms" ? fmt(v, 1) : fmt(v)}${bar}</td>`;
      }).join("")}</tr>`).join("")}</tbody>`;
    $("#eval-meta").textContent = `${summary.n_questions} questions · k=${k}`;

    const am = summary.answer;
    $("#answer-card").hidden = !am;
    if (am) {
      $("#answer-config").textContent = `answers from ${summary.answer_config}`;
      $("#answer-metrics").innerHTML = Object.entries(am).map(([key, v]) =>
        `<div class="metric"><div class="metric-label">${esc(key.replace(/_/g, " "))}</div><div class="metric-value">${fmt(v, 2)}</div></div>`).join("");
    }
    $("#eval-details").innerHTML = details.map((d, i) => {
      const retr = Object.entries(d.retrieval).map(([name, r]) => `<div><strong>${esc(name)}</strong>
        <span class="hits" aria-label="relevance of top results">${r.relevant_flags.map((f) => `<span class="hit ${f ? "on" : ""}"></span>`).join("")}</span>
        <span class="muted small">MRR ${fmt(r.metrics.mrr, 2)} · recall ${fmt(r.metrics[`recall@${k}`], 2)}</span></div>`).join("");
      const ans = d.answer ? `<div><strong>Answer (${esc(d.answer_method)})</strong><div class="answer">${renderMarkdownish(d.answer)}</div>
        <div class="muted small">${Object.entries(d.answer_metrics).map(([kk, v]) => `${esc(kk)} ${fmt(v, 2)}`).join(" · ")}</div></div>` : "";
      return `<details><summary>${i + 1}. ${esc(d.question)}</summary><div class="detail-body">
        ${d.reference ? `<div><strong>Reference</strong><div>${esc(d.reference)}</div></div>` : ""}${retr}${ans}</div></details>`;
    }).join("");
    $("#eval-results").hidden = false;
  }

  $("#eval-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const btn = $("#run-eval");
    btn.disabled = true;
    setMsg('<span class="spinner" aria-hidden="true"></span>Running evaluation… this can take a while with an LLM.');
    try {
      const run = await api("/api/eval/run", {
        method: "POST",
        body: JSON.stringify({
          file: $("#dataset").value,
          configs: [...document.querySelectorAll("input[name=configs]:checked")].map((i) => i.value),
          k: Number($("#eval-k").value),
          generate: $("#eval-generate").checked,
          use_llm: $("#eval-llm").checked,
          llm_judge: $("#eval-judge").checked,
        }),
      });
      render(run);
      setMsg(`Run #${run.id} complete.`, "msg-ok");
      loadRuns();
    } catch (err) {
      setMsg(esc(err.message), "msg-error");
    } finally {
      btn.disabled = false;
    }
  });

  $("#synth-btn").addEventListener("click", async () => {
    setMsg('<span class="spinner" aria-hidden="true"></span>Generating questions from indexed chunks…');
    try {
      const r = await api("/api/eval/synthesize", { method: "POST", body: JSON.stringify({ n: 10 }) });
      await loadDatasets(r.file);
      setMsg(`Created ${esc(r.file)} with ${r.items.length} questions.`, "msg-ok");
    } catch (err) {
      setMsg(esc(err.message), "msg-error");
    }
  });

  $("#run-list").addEventListener("click", async (e) => {
    const b = e.target.closest("[data-run]");
    if (b) render(await api(`/api/eval/runs/${b.dataset.run}`));
  });

  loadDatasets();
  loadRuns();
}

document.addEventListener("DOMContentLoaded", () => {
  loadStatus();
  const page = document.body.dataset.page;
  if (page === "ask") initAsk();
  if (page === "documents") initDocuments();
  if (page === "evaluate") initEvaluate();
});
