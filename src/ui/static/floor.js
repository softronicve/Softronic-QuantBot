// The quant floor: the last cycle and the live account as five desks, each with its status light.
// The figures come from /api/floor, read only while the floor is the view on screen.
"use strict";

const FLOOR_MS = 10000;
const DESKS = [
  { key: "data", title: "Data intake", role: "Data engineer", hair: "#3b2a1a", shirt: "#5aa9e6" },
  { key: "signal", title: "Donchian 20/10 engine", role: "Signal quant", hair: "#d9a441", shirt: "#f0b429" },
  { key: "risk", title: "Risk guardian", role: "Portfolio risk manager", hair: "#1c1c1c", shirt: "#33c27f" },
  { key: "execution", title: "Trading desk and stops", role: "Execution and settlement", hair: "#7a3b1f", shirt: "#f0616d" },
  { key: "audit", title: "Nightly auditor", role: "Chief auditor", hair: "#c8c8c8", shirt: "#aab6c3" },
];
const LIGHT_NAMES = { ok: "all good", watch: "worth a look", alert: "needs attention", idle: "waiting" };
let floor = null;
let floorDesk = "execution";
let floorView = false;
let floorTimer = null;

// A seated figure at a desk, 16 × 16 pixels: hair, face, shirt, the desk edge and a monitor.
const SPRITE = [
  "................",
  "......HHHH......",
  ".....HHHHHH.....",
  ".....HFFFFH.....",
  ".....FFEFEF.....",
  ".....FFFFFF.....",
  "......FFFF......",
  "....SSSSSSSS....",
  "...SSSSSSSSSS...",
  "...SSSSSSSSSS...",
  "..FSSSSSSSSSSF..",
  "DDDDDDDDDDDDDDDD",
  "DDDDDDDDDDDDDDDD",
  ".LL..........LL.",
  ".LL..........LL.",
  "................",
];

function sprite(desk) {
  const canvas = el("canvas", { class: "sprite", width: "16", height: "16", "aria-hidden": "true" });
  const pen = canvas.getContext("2d");
  const colours = { H: desk.hair, F: "#e0b48c", E: "#1a1f26", S: desk.shirt, D: "#3a4654", L: "#222d3a" };
  SPRITE.forEach((row, y) => [...row].forEach((cell, x) => {
    if (!colours[cell] || !pen) return;
    pen.fillStyle = colours[cell];
    pen.fillRect(x, y, 1, 1);
  }));
  return canvas;
}

const pct = (value, decimals = 2) => (isNumber(value) ? `${signed(value, decimals)} %` : "—");
const utcTime = (iso) => (iso ? `${iso.slice(11, 19)} UTC` : "—");

// What each desk's monitor shows at a glance.
function screen(key, desk) {
  switch (key) {
    case "data": return [desk.status || "—", `${desk.symbols_with_data ?? "—"}/${desk.symbols ?? "—"} symbols`];
    case "signal": return desk.regime && isNumber(desk.regime.breaking_out)
      ? [`${desk.regime.breaking_out} breaking out`, `${desk.regime.pre_alert} on pre-alert`] : ["no regime yet"];
    case "risk": return [`heat ${number(desk.heat_pct)} %`,
      !desk.stops_checked ? "stops not checked yet" : desk.stops_ok ? "stops in place" : "check stops"];
    case "execution": return desk.positions && desk.positions.length
      ? desk.positions.slice(0, 2).map((p) => `${coinOf(p.symbol)} ${signed(p.pnl_usd)} $`) : ["no positions"];
    case "audit": return [desk.audit ? "review ready" : "3 questions", desk.cycle_date || "—"];
    default: return [];
  }
}

function renderFloor() {
  const desks = (floor && floor.desks) || {};
  replace($("floor-room"), DESKS.map((meta) => {
    const desk = desks[meta.key] || { light: "idle" };
    return el("button", {
      type: "button", class: `station ${meta.key}${floorDesk === meta.key ? " active" : ""}`, "data-light": desk.light,
      "aria-pressed": String(floorDesk === meta.key), "aria-label": `${meta.title}: ${LIGHT_NAMES[desk.light] || ""}`,
    },
    el("span", { class: "monitor" }, screen(meta.key, desk).map((line) => el("span", { text: line }))),
    sprite(meta),
    el("span", { class: "plate" }, el("i", { class: "lamp", title: LIGHT_NAMES[desk.light] }),
      el("b", { text: meta.title }), el("small", { text: meta.role })));
  }));
  for (const button of $("floor-room").children) {
    const key = [...button.classList].find((name) => DESKS.some((desk) => desk.key === name));
    button.addEventListener("click", () => {
      floorDesk = key;
      renderFloor();
      // On a narrow screen the detail sits below all five desks: take the reader there.
      if (window.matchMedia("(max-width: 900px)").matches) $("floor-detail").scrollIntoView({ block: "start" });
    });
  }
  $("floor-caption").textContent = floor && floor.cycle_date
    ? `Cycle ${floor.cycle_date} in ${floor.mode}. Pick a desk to see its detail.` : "No cycle yet.";
  renderDesk(floorDesk, desks[floorDesk] || { light: "idle" });
}

function fact2(term, value, detail, extra) {
  return [el("dt", { text: term }), el("dd", {}, el("b", { text: value }), detail ? el("span", { text: detail }) : null, extra || null)];
}

function gauge(value, limit, label) {
  const share = isNumber(value) && limit > 0 ? Math.min(100, (value / limit) * 100) : 0;
  return el("div", { class: "gauge", role: "img", "aria-label": label },
    el("i", { style: `width:${share.toFixed(1)}%`, class: share >= 75 ? "hot" : "" }));
}

function renderDesk(key, desk) {
  const meta = DESKS.find((each) => each.key === key);
  const body = [el("h3", {}, el("i", { class: "lamp", "data-light": desk.light }), meta.title,
    el("small", { text: ` · ${LIGHT_NAMES[desk.light] || ""}` }))];
  if (key === "data") {
    body.push(el("dl", { class: "facts" },
      fact2("Cycle", desk.status || "—", desk.completed_at ? `finished at ${utcTime(desk.completed_at)}` : null),
      fact2("Delay", isNumber(desk.late_minutes) ? `${number(desk.late_minutes, 1)} min` : "—", "after 00:01:30 UTC"),
      fact2("Symbols with candles", `${desk.symbols_with_data ?? "—"} of ${desk.symbols ?? "—"}`,
        desk.missing && desk.missing.length ? `no data: ${desk.missing.map(coinOf).join(", ")}` : "all in sync")));
    body.push(el("p", { class: "hint", text: desk.anomalies && desk.anomalies.length
      ? `Range of 2 ATR or more on the last candle: ${desk.anomalies.map((a) => `${coinOf(a.symbol)} (${number(a.range_in_atr)} ATR, ${pct(a.return_pct)})`).join(", ")}.`
      : "No candle with a range of 2 ATR or more." }));
  } else if (key === "signal") {
    const regime = desk.regime || {};
    const bench = desk.benchmarks;
    body.push(el("dl", { class: "facts" },
      fact2("Breaking out", String(regime.breaking_out ?? "—"), `of ${regime.symbols ?? "—"} coins`),
      fact2("On pre-alert", String(regime.pre_alert ?? "—"), `strength between ${isNumber(regime.pre_alert_from) ? signed(regime.pre_alert_from, 1) : "−1.5"} and 0`),
      fact2("Below their 10-day low", String(regime.below_exit_channel ?? "—"), "at yesterday's close"),
      bench ? fact2(`Since ${bench.since}`, `bot ${pct(bench.bot_pct)}`, `BTC ${pct(bench.btc_pct)} · basket ${pct(bench.basket_pct)}`) : null));
    body.push(el("h4", { text: "The five closest to breaking their 20-day high" }),
      el("ol", { class: "proximity" }, (desk.top || []).map((row) => {
        const progress = isNumber(row.live_progress_pct) ? row.live_progress_pct : row.progress_pct;
        return el("li", {},
          el("button", { type: "button", class: "linkish", text: coinOf(row.symbol), title: "Open in the inspector",
            "data-symbol": row.symbol }),
          el("span", { class: "bar" }, el("i", { style: `width:${Math.max(0, Math.min(100, progress)).toFixed(1)}%` })),
          el("span", { class: "num", text: `${number(progress, 1)} %` }),
          el("span", { class: "num muted", text: `strength ${signed(row.strength)}` }));
      })));
  } else if (key === "risk") {
    body.push(el("dl", { class: "facts" },
      fact2("Portfolio heat", `${number(desk.heat_pct)} % of ${number(desk.heat_limit_pct, 0)} %`, "risk down to the stops",
        gauge(desk.heat_pct, desk.heat_limit_pct, "portfolio heat against its limit")),
      fact2("Notional leverage", `${number(desk.leverage, 3)}x of ${number(desk.leverage_limit, 1)}x`, null,
        gauge(desk.leverage, desk.leverage_limit, "leverage against its limit")),
      fact2("Free collateral", `${number(desk.available_balance)} USDT`, `balance ${number(desk.wallet_balance)}`),
      fact2(`Correlation, ${desk.correlation_days ?? 60} days`, isNumber(desk.correlation_mean) ? signed(desk.correlation_mean) : "—",
        isNumber(desk.tail_mean) ? `on down days ${signed(desk.tail_mean)} (${desk.tail_days} days, ${number(desk.tail_down_share_pct, 0)} % of the coins fall)` : "on down days: not measured"),
      fact2("Stops", !desk.stops_checked ? "not checked yet" : desk.stops_ok ? "100 % protected" : "check them", desk.stops)));
  } else if (key === "execution") {
    const positions = desk.positions || [];
    body.push(positions.length ? positions.map(ladder) : el("p", { class: "hint", text: "No open positions." }));
    const settled = desk.settlement || [];
    if (settled.length) {
      body.push(el("h4", { text: "After each exit" }), el("table", { class: "settle" },
        el("tr", {}, ["Coin", "Exit", "Exit price", "Close today", "Since the exit", "Protected"].map((t, i) => el("th", { class: i > 1 ? "num" : null, text: t }))),
        settled.map((t) => el("tr", {},
          el("td", { text: coinOf(t.symbol) }), el("td", { text: t.reason === "STOP" ? "stop" : "channel" }),
          el("td", { class: "num", text: price(t.exit_price) }), el("td", { class: "num", text: price(t.close_today) }),
          // Neutral on purpose: a rise after a stop is not good news for the stop, a fall is not bad news.
          el("td", { class: "num", text: pct(t.after_exit_pct) }),
          el("td", { class: `num ${tone(t.protected_usd)}`, text: isNumber(t.protected_usd) ? `${signed(t.protected_usd)} $` : "—" })))));
      body.push(el("p", { class: "hint", text: isNumber(desk.protected_total_usd)
        ? `Taken together the exits ${desk.protected_total_usd >= 0 ? "saved" : "gave up"} ${number(Math.abs(desk.protected_total_usd))} USDT against holding until today.`
        : "" }));
    }
  } else if (key === "audit") {
    body.push(el("p", {}, el("button", { type: "button", class: "button", id: "floor-report", text: `Read the report of ${desk.cycle_date || "the day"}` })));
    body.push(desk.audit
      ? el("p", { class: "hint", text: `The automatic auditor (${desk.audit.model}) has answered: see section 6 of the report.` })
      : el("p", { class: "hint", text: "Automatic auditor off: copy each block into your LLM, or turn it on with BOT_AUDIT_LLM__ENABLED=true." }));
    for (const [index, prompt] of (desk.prompts || []).entries()) {
      const answer = desk.audit && desk.audit.answers && desk.audit.answers[index];
      body.push(el("details", { class: "prompt" },
        el("summary", {}, prompt.title, " ", el("button", { type: "button", class: "copy", "data-copy": String(index), text: "Copy" })),
        el("pre", { text: prompt.prompt }),
        answer && answer.answer ? el("div", { class: "answer" }, el("h5", { text: "Review" }), el("p", { text: answer.answer })) : null));
    }
  }
  replace($("floor-detail"), body);
  for (const button of $("floor-detail").querySelectorAll("[data-symbol]")) {
    button.addEventListener("click", () => { showView("inspector"); select(button.dataset.symbol); });
  }
  for (const button of $("floor-detail").querySelectorAll("[data-copy]")) {
    button.addEventListener("click", async (event) => {
      event.preventDefault();
      try {
        await navigator.clipboard.writeText(desk.prompts[Number(button.dataset.copy)].prompt);
        button.textContent = "Copied";
      } catch (error) {
        button.textContent = "Could not copy";
      }
    });
  }
  const read = $("floor-report");
  if (read) read.addEventListener("click", () => openReport(desk.cycle_date));
}

// One position between the levels that matter: exit channel, stop, entry, price now, 20-day high.
function ladder(p) {
  const marks = [
    ["10 d low", p.exit_channel, "exit"], ["Stop", p.stop_price, "stop"], ["Entry", p.entry_price, "entry"],
    ["Mark", p.price, "mark"], ["20 d high", p.high_entry, "high"],
  ].filter(([, value]) => isNumber(value));
  const values = marks.map(([, value]) => value);
  const low = Math.min(...values) * 0.99;
  const high = Math.max(...values) * 1.01;
  const at = (value) => ((value - low) / (high - low)) * 100;
  return el("div", { class: "ladder" },
    el("p", {}, el("b", { text: `${coinOf(p.symbol)} · ${units(p.quantity)} units` }), " ",
      el("span", { class: tone(p.pnl_usd), text: `${signed(p.pnl_usd)} USDT${isNumber(p.r_multiple) ? ` (${signed(p.r_multiple)}R)` : ""}` }), " ",
      el("span", { class: "muted", text: isNumber(p.exit_distance_pct)
        ? `· the ${p.nearest_exit_is_stop ? "stop" : "channel"} would sell first, at ${pct(p.exit_distance_pct)}` : "" })),
    el("div", { class: "track", role: "img", "aria-label": marks.map(([name, value]) => `${name} ${price(value)}`).join(", ") },
      marks.map(([name, value, kind]) => el("span", { class: `tick ${kind}`, style: `left:${at(value).toFixed(2)}%` },
        el("i", {}), el("small", { text: `${name} ${price(value)}` })))));
}

async function openReport(date) {
  const dialog = $("floor-dialog");
  replace($("floor-dialog-body"), el("p", { class: "hint", text: "Reading the report…" }));
  dialog.showModal();
  try {
    const response = await fetch(`/api/report.md${date ? `?date=${encodeURIComponent(date)}` : ""}`, { cache: "no-store" });
    if (!response.ok) throw new Error(`the server answered ${response.status}`);
    replace($("floor-dialog-body"), el("pre", { text: await response.text() }));
  } catch (error) {
    replace($("floor-dialog-body"), el("p", { text: `Could not read the report: ${error.message}` }));
  }
}

async function loadFloor() {
  if (!floorView || document.hidden) return;
  try {
    floor = await fetchJson("/api/floor");
    renderFloor();
  } catch (error) {
    $("floor-caption").textContent = `Could not read the quant floor: ${error.message}`;
  }
}

// The inspector panel holds two views: the chart of one coin, and the floor.
function showView(view) {
  floorView = view === "floor";
  for (const button of $("inspector-views").children) button.setAttribute("aria-pressed", String(button.dataset.view === view));
  $("inspector-body").hidden = floorView;
  $("floor").hidden = !floorView;
  for (const id of ["timeframes", "symbol-picker"]) $(id).hidden = floorView;
  $("inspector-title").textContent = floorView ? "Quant floor" : "Asset inspector";
  try { localStorage.setItem("inspector-view", view); } catch (error) { /* private mode: the choice is not kept */ }
  if (floorTimer) clearInterval(floorTimer);
  floorTimer = null;
  if (floorView) {
    loadFloor();
    floorTimer = setInterval(loadFloor, FLOOR_MS);
  } else {
    redrawChart();
  }
}

for (const button of $("inspector-views").children) button.addEventListener("click", () => showView(button.dataset.view));
$("floor-dialog-close").addEventListener("click", () => $("floor-dialog").close());
document.addEventListener("visibilitychange", () => { if (!document.hidden) loadFloor(); });
try {
  if (localStorage.getItem("inspector-view") === "floor") showView("floor");
} catch (error) { /* no stored choice */ }
