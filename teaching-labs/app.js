const state = {
  split: "random"
};

const controls = [
  "siteCount",
  "siteEffect",
  "noiseLevel",
  "complexity",
  "sampleSize",
  "labelNoise"
];

function value(id) {
  return Number(document.getElementById(id).value);
}

function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

function syncRangeLabel(id, suffix = "") {
  document.querySelector(`[data-value-for="${id}"]`).textContent = `${value(id)}${suffix}`;
}

function clamp(n, min, max) {
  return Math.max(min, Math.min(max, n));
}

function seededNoise(i, scale = 1) {
  const x = Math.sin(i * 97.13 + 13.7) * 10000;
  return (x - Math.floor(x) - 0.5) * 2 * scale;
}

function setupCanvas(canvas) {
  const ctx = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(320, Math.round(rect.width * ratio));
  canvas.height = Math.round((rect.width * 0.46) * ratio);
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { ctx, width: rect.width, height: rect.width * 0.46 };
}

function drawAxes(ctx, width, height, title) {
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#d6dee4";
  ctx.lineWidth = 1;
  for (let i = 0; i < 5; i += 1) {
    const y = 46 + i * ((height - 82) / 4);
    ctx.beginPath();
    ctx.moveTo(52, y);
    ctx.lineTo(width - 26, y);
    ctx.stroke();
  }
  ctx.fillStyle = "#16324f";
  ctx.font = '700 16px "SFMono-Regular", Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace';
  ctx.fillText(title, 18, 26);
  ctx.strokeStyle = "#93a4b5";
  ctx.beginPath();
  ctx.moveTo(52, 42);
  ctx.lineTo(52, height - 36);
  ctx.lineTo(width - 26, height - 36);
  ctx.stroke();
}

function drawPoint(ctx, x, y, color, radius = 4) {
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fill();
}

function drawBar(ctx, x, y, w, h, color) {
  ctx.fillStyle = color;
  ctx.fillRect(x, y, w, h);
}

function renderModelLab() {
  syncRangeLabel("siteCount");
  syncRangeLabel("siteEffect");
  syncRangeLabel("noiseLevel");

  const sites = value("siteCount");
  const siteEffect = value("siteEffect");
  const noise = value("noiseLevel");
  const randomScore = clamp(0.92 - noise * 0.004 + siteEffect * 0.002, 0.58, 0.97);
  const groupScore = clamp(randomScore - siteEffect * 0.009 - noise * 0.003, 0.18, 0.88);
  const risk = randomScore - groupScore > 0.27 ? "High" : randomScore - groupScore > 0.16 ? "Moderate" : "Low";

  setText("randomScore", randomScore.toFixed(2));
  setText("groupScore", groupScore.toFixed(2));
  setText("leakageRisk", risk);

  const { ctx, width, height } = setupCanvas(document.getElementById("modelCanvas"));
  drawAxes(ctx, width, height, state.split === "random" ? "Random split mixes sites" : "Group held out split tests a new site");

  const plotW = width - 90;
  const plotH = height - 86;
  const points = sites * 16;
  for (let i = 0; i < points; i += 1) {
    const site = i % sites;
    const localX = (i * 17) % 100;
    const x = 58 + (localX / 100) * plotW;
    const base = 0.72 * localX + 14;
    const shift = (site - sites / 2) * (siteEffect / 4);
    const yValue = base + shift + seededNoise(i, noise);
    const y = height - 40 - clamp(yValue / 110, 0.02, 0.98) * plotH;
    const heldOut = site === sites - 1;
    const color = heldOut ? "#c2410c" : ["#0f766e", "#2563eb", "#7c3aed", "#b7791f"][site % 4];
    const alpha = state.split === "group" && heldOut ? 1 : 0.72;
    ctx.globalAlpha = alpha;
    drawPoint(ctx, x, y, color, heldOut ? 5 : 3.6);
  }
  ctx.globalAlpha = 1;

  ctx.strokeStyle = state.split === "random" ? "#0f766e" : "#c2410c";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(64, height - 76);
  ctx.lineTo(width - 34, state.split === "random" ? 72 : 112);
  ctx.stroke();
}

function renderMlLab() {
  syncRangeLabel("complexity");
  syncRangeLabel("sampleSize");
  syncRangeLabel("labelNoise", "%");

  const complexity = value("complexity");
  const sampleSize = value("sampleSize");
  const labelNoise = value("labelNoise");
  const train = clamp(58 + complexity * 4.2 - labelNoise * 0.12, 52, 99);
  const sweetSpot = 5 + sampleSize / 120;
  const overfitPenalty = Math.max(0, complexity - sweetSpot) ** 2 * 1.8;
  const test = clamp(train - overfitPenalty - labelNoise * 0.45 + sampleSize * 0.045, 42, 95);
  const gap = clamp(train - test, 0, 55);

  setText("trainAcc", `${Math.round(train)}%`);
  setText("testAcc", `${Math.round(test)}%`);
  setText("overfitGap", `${Math.round(gap)}%`);

  const { ctx, width, height } = setupCanvas(document.getElementById("mlCanvas"));
  drawAxes(ctx, width, height, "Training score can hide generalization risk");

  const left = 66;
  const bottom = height - 38;
  const plotW = width - 108;
  const plotH = height - 86;
  const bars = [
    { label: "Training", value: train, color: "#0f766e" },
    { label: "Test", value: test, color: "#2563eb" },
    { label: "Gap", value: gap, color: "#c2410c" }
  ];
  bars.forEach((bar, i) => {
    const x = left + i * (plotW / 3) + 28;
    const h = (bar.value / 100) * plotH;
    drawBar(ctx, x, bottom - h, 70, h, bar.color);
    ctx.fillStyle = "#334155";
    ctx.font = '700 13px "SFMono-Regular", Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace';
    ctx.fillText(bar.label, x, bottom + 22);
    ctx.fillText(`${Math.round(bar.value)}%`, x + 12, bottom - h - 8);
  });

  ctx.strokeStyle = "#7c3aed";
  ctx.lineWidth = 3;
  ctx.beginPath();
  for (let i = 0; i < 10; i += 1) {
    const x = left + (i / 9) * plotW;
    const y = bottom - clamp((65 + Math.sin(i / 1.5) * 13 - Math.max(0, i - complexity) * 3) / 100, 0, 1) * plotH;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

function renderAll() {
  renderModelLab();
  renderMlLab();
}

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    const target = tab.dataset.target;
    document.querySelectorAll(".tab").forEach((item) => item.classList.toggle("is-active", item === tab));
    document.querySelectorAll("[data-lab-panel]").forEach((panel) => {
      panel.classList.toggle("is-active", panel.id === target);
    });
    document.getElementById(target).scrollIntoView({ behavior: "smooth", block: "start" });
    renderAll();
  });
});

document.querySelectorAll("[data-split]").forEach((button) => {
  button.addEventListener("click", () => {
    state.split = button.dataset.split;
    document.querySelectorAll("[data-split]").forEach((item) => item.classList.toggle("is-selected", item === button));
    renderModelLab();
  });
});

document.querySelectorAll(".reveal").forEach((button) => {
  button.addEventListener("click", () => {
    const answer = document.getElementById(button.dataset.reveal);
    answer.hidden = !answer.hidden;
    button.textContent = answer.hidden ? "Show sample answer" : "Hide sample answer";
  });
});

controls.forEach((id) => {
  document.getElementById(id).addEventListener("input", renderAll);
});

window.addEventListener("resize", renderAll);

renderAll();
