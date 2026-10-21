// Basket dashboard. It only draws what the server read from disk; it computes no decisions.
"use strict";

const LC = LightweightCharts;
// The report, the candles and the history only change with the cycle; prices are fetched much more often.
const REFRESH_MS = 60000;
const LIVE_MS = 10000;
const LIVE_STALE_MS = 45000;
const $ = (id) => document.getElementById(id);

const STATES = {
  IN_POSITION: ["In position", "pos"],
  CANDIDATE: ["Breakout candidate", "cand"],
  WAITING: ["Waiting", "wait"],
  NO_DATA: ["No data", "none"],
};

let snapshot = null;
let selected = null;
let chart = null;
let series = null;
let curve = null;
let lastPrices = {};
// What the other dashboard files add: called after each full read, after each
// price read and when the coin being viewed changes.
const hooks = { full: [], live: [], select: [] };
const runHooks = (kind) => { for (const hook of hooks[kind]) hook(); };

// --- Utilities --------------------------------------------------------------

function el(tag, attributes = {}, ...children) {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(attributes)) {
    if (value === null || value === undefined || value === false) continue;
    if (name === "class") node.className = value;
    else if (name === "text") node.textContent = value;
    else node.setAttribute(name, value === true ? "" : value);
  }
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function replace(node, ...children) {
  node.replaceChildren(...children.flat(Infinity).filter((child) => child !== null && child !== undefined));
}

const isNumber = (value) => typeof value === "number" && Number.isFinite(value);

function number(value, decimals = 2) {
  if (!isNumber(value)) return "—";
  return value.toLocaleString("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
}

// A price with the decimals its size calls for: BTC does not read like DOGE.
function priceDecimals(value) {
  const magnitude = Math.abs(value);
  return magnitude >= 1000 ? 2 : magnitude >= 1 ? 4 : 6;
}
const price = (value) => (isNumber(value) ? number(value, priceDecimals(value)) : "—");
const usd = (value) => (isNumber(value) ? `${number(value, 2)} USDT` : "—");
const signed = (value, decimals = 2) => (isNumber(value) ? `${value >= 0 ? "+" : "−"}${number(Math.abs(value), decimals)}` : "—");
const units = (value) => (isNumber(value) ? value.toLocaleString("en-US", { maximumFractionDigits: 8 }) : "—");
const tone = (value) => (!isNumber(value) || value === 0 ? "" : value > 0 ? "up" : "down");
// The coin of a contract: BTCUSDT -> BTC. Shared by every script of the page.
const coinOf = (symbol) => symbol.replace(/USDT$/, "");

function heatClass(proximity) {
  if (!isNumber(proximity)) return "heat-none";
  if (proximity >= 0) return "heat-out";
  if (proximity >= -2) return "heat-near";
  if (proximity >= -5) return "heat-mid";
  return "heat-far";
}

// --- Header -----------------------------------------------------------------

// The cycle runs on UTC; the second clock is the browser's own time zone. The current time comes from
// the browser clock; the server gives the time of the next cycle.
const ZONES = [["utc", "UTC"], ["local", undefined]].map(([key, timeZone]) => ({
  key,
  weekday: new Intl.DateTimeFormat("en-US", { timeZone, weekday: "short" }),
  moment: new Intl.DateTimeFormat("en-US", {
    timeZone, day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit", second: "2-digit", hourCycle: "h23",
  }),
}));
let nextCycle = null;

// The weekday goes in its own element so it can be hidden on narrow screens.
function stamp(zone, date) {
  return [el("span", { class: "wd", text: `${zone.weekday.format(date)} ` }), zone.moment.format(date)];
}

function duration(ms) {
  const seconds = Math.max(0, Math.round(ms / 1000));
  if (seconds < 60) return `${seconds} s`;
  const minutes = Math.floor(seconds / 60);
  return minutes < 60 ? `${minutes} min` : `${Math.floor(minutes / 60)} h ${minutes % 60} min`;
}

function renderLive(now) {
  // The dot at the top: green if the prices are seconds old, amber if the reading has gone stale.
  const node = $("live");
  const at = snapshot && snapshot.radar ? snapshot.radar.at : null;
  if (!isNumber(at)) {
    node.className = "live off";
    node.textContent = snapshot && snapshot.report ? "No current prices" : "—";
    return;
  }
  const age = Math.max(0, now.getTime() - at);
  node.className = age < LIVE_STALE_MS ? "live on" : "live stale";
  node.textContent = age < LIVE_STALE_MS ? `Live · ${duration(age)} ago` : `Prices from ${duration(age)} ago`;
}

function renderClocks() {
  const now = new Date();
  renderLive(now);
  for (const zone of ZONES) replace($(`clock-${zone.key}`), stamp(zone, now));
  const countdown = $("next-in");
  countdown.className = nextCycle && nextCycle.pending ? "countdown late" : "countdown";
  if (!nextCycle) countdown.textContent = "";
  else {
    const left = nextCycle.at - now.getTime();
    countdown.textContent = nextCycle.pending ? `pending · ${duration(-left)} late`
      : left > 0 ? `in ${duration(left)}` : "now";
  }
}

function renderNextCycle(meta) {
  nextCycle = meta.next_cycle || null;
  for (const zone of ZONES) replace($(`next-${zone.key}`), nextCycle ? stamp(zone, new Date(nextCycle.at)) : "—");
  renderClocks();
}

function renderTop(report, meta) {
  const mode = $("mode");
  const download = $("download");
  replace($("notices"), (meta.notices || []).map((notice) => el("p", { text: notice })));
  if (!report) {
    mode.textContent = "—";
    mode.className = "badge";
    $("cycle").textContent = "No cycle has run yet";
    download.setAttribute("aria-disabled", "true");
    return;
  }
  mode.textContent = report.mode;
  mode.className = `badge ${report.mode.toLowerCase()}`;
  mode.title = report.mode === "DRY_RUN" ? "Paper account: no order is sent" : "";
  const late = report.late_minutes > 5 ? ` · ${number(report.late_minutes, 0)} min late` : "";
  $("cycle").textContent = `Cycle ${report.cycle_date} · ${report.status}${late}`;
  download.removeAttribute("aria-disabled");
  download.href = `/api/report.md?date=${encodeURIComponent(report.cycle_date)}`;
}

function bar(fraction) {
  const share = Math.max(0, Math.min(1, fraction || 0));
  const state = share >= 1 ? "bar full" : share >= 0.9 ? "bar warn" : "bar";
  return el("div", { class: state, role: "presentation" }, el("span", { style: `width:${(share * 100).toFixed(1)}%` }));
}

function metric(label, value, sub, extra) {
  return el("div", { class: "metric" }, el("div", { class: "label", text: label }), el("div", { class: "value" }, value),
    sub ? el("div", { class: "sub" }, sub) : null, extra || null);
}

function renderPortfolio(report, live) {
  if (!report) {
    replace($("portfolio"), metric("Portfolio", "—", "Run the daily cycle to see data"));
    return;
  }
  const a = report.account;
  // With positions, the open P&L is the current one; without them there is nothing to update.
  const current = live && isNumber(live.equity);
  const openPnl = current ? live.open_pnl_usd : a.unrealized_pnl;
  const pnl = el("span", { class: tone(openPnl), text: `${signed(openPnl)} USDT` });
  replace($("portfolio"),
    metric("Balance", [number(a.wallet_balance), el("small", { text: " USDT" })],
      "Realized balance: what is already closed, minus fees"),
    metric("Account at current prices", [number(current ? live.equity : a.equity), el("small", { text: " USDT" })],
      [pnl, " in open positions"]),
    metric("Notional value", [number(a.leverage, 2), el("small", { text: `x of ${number(a.leverage_limit, 1)}x` })],
      `${number(a.nominal_usd)} USDT invested: ${number(a.leverage * 100, 0)} % of the balance`,
      bar(a.leverage / a.leverage_limit)),
    metric("Portfolio heat", [number(a.heat_pct), el("small", { text: ` % of ${number(a.heat_limit_pct, 1)} %` })],
      `If every position touched its stop, ${number(a.heat_usd)} USDT would be lost`, bar(a.heat_pct / a.heat_limit_pct)),
    metric("Margin used", [number(a.margin_used), el("small", { text: " USDT" })],
      `Collateral for the positions · free ${number(a.available_balance)} USDT`),
  );
}

// --- Open positions ---------------------------------------------------------

const POSITION_STATES = {
  HOLD: ["Holding", "pos"],
  STOP_TOUCHED: ["Stop touched", "bad"],
  BELOW_EXIT_CHANNEL: ["Below exit channel", "cand"],
};
const OPENED = new Intl.DateTimeFormat("en-US", {
  timeZone: "UTC", day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit", hourCycle: "h23",
});
const CLOCK = (timeZone) => new Intl.DateTimeFormat("en-US", {
  timeZone, hour: "2-digit", minute: "2-digit", second: "2-digit", hourCycle: "h23",
});
const READ_AT = [CLOCK("UTC"), CLOCK(undefined)];
const percent = (value) => (isNumber(value) ? `${signed(value)} %` : "—");

function cell(kind, main, detail) {
  return el("td", { class: kind }, main, detail ? el("span", { class: "sub" }, detail) : null);
}

function whatHappens(position, simulated) {
  const stop = price(position.stop_price);
  const channel = price(position.exit_channel);
  if (position.status === "STOP_TOUCHED") {
    return simulated
      ? `The price has touched the stop: the sell will be booked in the next cycle, at ${stop}.`
      : "The price has touched the stop: the exchange order will already have sold. It will show in the next cycle.";
  }
  if (position.status === "BELOW_EXIT_CHANNEL") {
    return `It is below the exit channel: if today's candle closes below ${channel}, it is sold in the next cycle.`;
  }
  const levels = [isNumber(position.stop_price) ? `it touches ${stop}` : null,
    isNumber(position.exit_channel) ? `today's candle closes below ${channel}` : null].filter(Boolean);
  return levels.length ? `Still open. It is sold if ${levels.join(" or if ")}.` : "Still open.";
}

function renderPositions(data) {
  const report = data.report;
  const positions = data.positions || [];
  const live = data.live || {};
  const body = $("positions").tBodies[0];
  const simulated = Boolean(report) && report.mode === "DRY_RUN";
  if (report) $("exit-period").textContent = report.parameters.exit_period;
  $("stop-note").textContent = !report ? "" : simulated
    ? "In DRY_RUN the stop is not a real order: each cycle checks it against the day's low and, if it was touched, the sell is booked then at the stop price."
    : "The stop is an order placed on the exchange: it sells the moment the price touches it, without waiting for the cycle.";
  if (!positions.length) {
    $("positions-hint").textContent = "";
    replace(body, el("tr", { class: "empty" }, el("td", { colspan: "7", text: report
      ? `No open positions: the bot is waiting for a coin to close above its ${report.parameters.entry_period}-day high.`
      : "No data yet." })));
    return;
  }
  $("positions-hint").textContent = live.live && isNumber(live.at)
    ? `Price read at ${READ_AT[0].format(new Date(live.at))} UTC (${READ_AT[1].format(new Date(live.at))} local)`
    : "Could not read the current price of every position: where it is missing, the last cycle's is shown";

  const rows = positions.map((p) => {
    const [label, kind] = POSITION_STATES[p.status] || [p.status, "none"];
    const line = el("tr", {},
      cell("mono", el("b", { text: coinOf(p.symbol) }),
        isNumber(p.opened_at) ? `since ${OPENED.format(new Date(p.opened_at))} UTC` : null),
      cell("num", price(p.entry_price), `${units(p.quantity)} units · ${number(p.cost_usd)} USDT`),
      cell("num", price(p.price), p.valued_at_stop ? `exits at the stop: ${number(p.value_usd)} USDT`
        : `${p.price_is_live ? "worth" : "last cycle ·"} ${number(p.value_usd)} USDT`),
      cell(`num ${tone(p.pnl_usd)}`, `${signed(p.pnl_usd)} USDT`,
        p.valued_at_stop ? `${percent(p.pnl_pct)} · at the stop price` : percent(p.pnl_pct)),
      cell("num", price(p.stop_price), isNumber(p.stop_price)
        ? `${percent(p.stop_distance_pct)} away · would exit with ${signed(p.result_at_stop_usd)} USDT` : "no stop"),
      cell("num", price(p.exit_channel), isNumber(p.exit_channel) ? `${percent(p.exit_distance_pct)} from the price` : null),
      cell("wrap", el("span", { class: `chip ${kind}`, text: label }), whatHappens(p, simulated)),
    );
    line.addEventListener("click", () => select(p.symbol));
    return line;
  });
  const total = (field) => positions.reduce((sum, p) => sum + p[field], 0);
  const openPnl = total("pnl_usd");
  const costs = positions.every((p) => isNumber(p.costs_usd)) ? total("costs_usd") : null;
  // A position without a stop has no bounded loss: adding it as zero would understate the risk.
  const atStops = positions.every((p) => isNumber(p.result_at_stop_usd)) ? total("result_at_stop_usd") : null;
  rows.push(el("tr", { class: "total" },
    cell("", "Total"),
    cell("num", `${number(total("cost_usd"))} USDT`),
    cell("num", `${number(total("value_usd"))} USDT`),
    cell(`num ${tone(openPnl)}`, `${signed(openPnl)} USDT`, percent(openPnl / total("cost_usd") * 100)),
    cell("num", isNumber(atStops) ? `${signed(atStops)} USDT` : "—",
      isNumber(atStops) ? "if every stop were hit" : "some positions have no stop"),
    cell("num", ""),
    cell("wrap", "", isNumber(costs) ? `Fees and funding already paid: ${number(costs)} USDT (deducted from the balance).` : null),
  ));
  replace(body, rows);
}

// --- Equity curve -----------------------------------------------------------

function ensureCurve() {
  if (curve) return;
  const style = getComputedStyle(document.documentElement);
  const token = (name) => style.getPropertyValue(name).trim();
  const create = (id) => LC.createChart($(id), {
    autoSize: true,
    layout: { background: { color: "transparent" }, textColor: token("--ink-2"), fontFamily: token("--mono") },
    grid: { vertLines: { color: token("--rule") }, horzLines: { color: token("--rule") } },
    rightPriceScale: { borderColor: token("--rule") },
    timeScale: { borderColor: token("--rule"), rightOffset: 2, timeVisible: true, secondsVisible: false },
    crosshair: { mode: LC.CrosshairMode.Normal },
    handleScroll: false, handleScale: false,
  });
  const equity = create("equity-chart");
  const drawdown = create("drawdown-chart");
  const quiet = { priceLineVisible: false, lastValueVisible: true };
  curve = {
    charts: [equity, drawdown],
    benchmark: equity.addLineSeries({ ...quiet, color: token("--info"), lineWidth: 2, lineStyle: LC.LineStyle.Dashed }),
    equity: equity.addLineSeries({ ...quiet, color: token("--accent"), lineWidth: 2 }),
    drawdown: drawdown.addAreaSeries({
      ...quiet, lineColor: token("--bad"), lineWidth: 2, topColor: "rgba(240, 97, 109, 0.05)",
      bottomColor: "rgba(240, 97, 109, 0.35)", invertFilledArea: true,
      priceFormat: { type: "custom", minMove: 0.01, formatter: (value) => `${value.toFixed(2)} %` },
    }),
  };
}

function renderCurve(performance) {
  const stats = performance && performance.stats;
  const points = (performance && performance.points) || [];
  if (!stats) {
    $("curve-hint").textContent = "";
    replace($("curve-stats"), metric("Equity curve", "—", "It will appear with the first cycle"));
    if (curve) for (const name of ["equity", "benchmark", "drawdown"]) curve[name].setData([]);
    return;
  }
  const edge = stats.return_pct - stats.benchmark_return_pct;
  $("curve-hint").textContent = `Since ${stats.since} · ${stats.cycles} cycles${stats.is_live ? " and the current price" : ""}`;
  replace($("curve-stats"),
    metric("Return since the start", el("span", { class: tone(stats.return_pct), text: percent(stats.return_pct) }),
      `${signed(stats.equity - stats.start_equity)} USDT on ${number(stats.start_equity)}`),
    metric("BTC at the same exposure", el("span", { class: tone(stats.benchmark_return_pct), text: percent(stats.benchmark_return_pct) }),
      [el("span", { class: tone(edge), text: `${signed(edge)} points` }), " for the bot against that benchmark"]),
    metric("Current drawdown", el("span", { class: tone(stats.drawdown_pct), text: percent(stats.drawdown_pct) }),
      "Below the highest the account has been"),
    metric("Max drawdown", el("span", { class: tone(stats.max_drawdown_pct), text: percent(stats.max_drawdown_pct) }),
      "The worst since the start"),
    metric("Average exposure", [number(stats.mean_exposure_pct, 1), el("small", { text: " %" })],
      "Of the account invested, on average per cycle"),
  );
  ensureCurve();
  const line = (field) => points.map((point) => ({ time: point.time, value: point[field] }));
  curve.equity.setData(line("equity"));
  curve.benchmark.setData(line("benchmark"));
  curve.drawdown.setData(line("drawdown_pct"));
  for (const each of curve.charts) each.timeScale().fitContent();
}

// --- Radar ------------------------------------------------------------------

// --- Market list ------------------------------------------------------------

let listSort = { key: "volume", direction: -1 };
let listFilter = "all";
let listSearch = "";

const FILTERS = {
  all: () => true,
  held: (entry) => entry.row.state === "IN_POSITION",
  near: (entry) => isNumber(entry.proximity) && entry.proximity >= -3,
  up: (entry) => isNumber(entry.day) && entry.day > 0,
};

function volume(value) {
  if (!isNumber(value)) return "—";
  if (value >= 1e12) return `${number(value / 1e12, 2)} T`;
  return value >= 1e9 ? `${number(value / 1e9, 2)} B` : `${number(value / 1e6, value >= 1e8 ? 0 : 1)} M`;
}

const median = (values) => {
  const sorted = values.filter(isNumber).sort((a, b) => a - b);
  if (!sorted.length) return null;
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
};

// Each coin in the basket with its last-cycle data and, if they arrive, its current figures.
function listEntries(report, now) {
  const entries = report.basket.map((row) => {
    const live = now[row.symbol] || null;
    const closes = ((snapshot.market[row.symbol] || {}).close || []).filter(isNumber);
    const current = live ? live.price : row.close;
    const since = (days) => (closes.length >= days && isNumber(current) ? (current / closes[closes.length - days] - 1) * 100 : null);
    return {
      row, live, symbol: row.symbol, coin: coinOf(row.symbol), price: current,
      day: live ? live.change_pct : null, h24: live ? live.change_24h_pct : null, d7: since(7), d30: since(30),
      volume: live ? live.volume_24h_usd : null, funding: live ? live.funding_pct : null,
      proximity: live ? live.proximity_pct : row.proximity_pct, cycle: row.proximity_pct,
      strength: row.breakout_strength, decision: row.decision,
      spark: closes.slice(-30).concat(live ? [live.price] : []),
    };
  });
  const byVolume = entries.filter((entry) => isNumber(entry.volume)).sort((a, b) => b.volume - a.volume);
  byVolume.forEach((entry, index) => { entry.rank = index + 1; });
  return entries;
}

function compareEntries(a, b) {
  const { key, direction } = listSort;
  const [x, y] = [a[key], b[key]];
  // Whatever has no data always goes last, however the list is sorted.
  if (x === y) return a.symbol.localeCompare(b.symbol);
  if (x === null || x === undefined) return 1;
  if (y === null || y === undefined) return -1;
  return (typeof x === "string" ? x.localeCompare(y) : x - y) * direction;
}

function sparkline(values, change) {
  const points = values.filter(isNumber);
  if (points.length < 2) return "";
  const [width, height, pad] = [96, 26, 2];
  const [low, high] = [Math.min(...points), Math.max(...points)];
  const y = (value) => (high === low ? height / 2 : pad + (height - 2 * pad) * (1 - (value - low) / (high - low)));
  const path = points.map((value, i) => `${(i * (width / (points.length - 1))).toFixed(1)},${y(value).toFixed(1)}`).join(" ");
  // The empty drawing is in the page: cloning it avoids building SVG elements by hand.
  const svg = $("sparkline").content.firstElementChild.cloneNode(true);
  svg.setAttribute("class", `sparkline ${tone(change)}`);
  svg.setAttribute("aria-label", isNumber(change) ? `Last 30 days: ${signed(change)} %` : "Last 30 days");
  svg.firstElementChild.setAttribute("points", path);
  return svg;
}

function renderBreadth(entries) {
  const live = entries.filter((entry) => entry.live);
  const node = $("breadth");
  if (!live.length) {
    replace(node);
    return;
  }
  const up = live.filter((entry) => entry.day > 0).length;
  const middle = median(live.map((entry) => entry.day));
  const btc = live.find((entry) => entry.symbol === "BTCUSDT");
  const near = live.filter((entry) => entry.proximity >= -3).length;
  const funding = median(live.map((entry) => entry.funding));
  const total = live.reduce((sum, entry) => sum + (entry.volume || 0), 0);
  const item = (label, value, kind) => el("span", { class: "breadth-item" }, el("small", { text: label }), el("b", { class: kind || null, text: value }));
  replace(node,
    item("Up / down today", `${up} / ${live.length - up}`, up * 2 >= live.length ? "up" : "down"),
    item("Basket median today", percent(middle), tone(middle)),
    btc ? item("BTC today", percent(btc.day), tone(btc.day)) : null,
    item("Within 3 % of its high", String(near)),
    isNumber(funding) ? item("Median funding", `${signed(funding, 4)} %`) : null,
    total > 0 ? item("Basket 24 h volume", `${volume(total)} USDT`) : null,
  );
}

function tick(symbol, value) {
  // Up or down from the previous reading: a brief flash, as on a quote screen.
  const before = lastPrices[symbol];
  return isNumber(before) && isNumber(value) && value !== before ? (value > before ? " tick-up" : " tick-down") : "";
}

function renderPreview(data) {
  const rows = (data.radar && data.radar.rows) || {};
  const breaking = Object.keys(rows).filter((symbol) => rows[symbol].breaking).map(coinOf);
  const positions = data.positions || [];
  const stopped = positions.filter((p) => p.status === "STOP_TOUCHED").map((p) => coinOf(p.symbol));
  const exiting = positions.filter((p) => p.status === "BELOW_EXIT_CHANNEL").map((p) => coinOf(p.symbol));
  const node = $("radar-preview");
  if (!data.report || !Object.keys(rows).length) {
    replace(node);
    return;
  }
  const parts = [
    breaking.length ? ["above their high (buy candidates, if there is free risk): ", el("b", { class: "up", text: breaking.join(", ") })] : null,
    stopped.length ? ["stop touched (it is sold): ", el("b", { class: "down", text: stopped.join(", ") })] : null,
    exiting.length ? ["below their exit channel (it would be sold): ", el("b", { class: "down", text: exiting.join(", ") })] : null,
  ].filter(Boolean);
  replace(node, el("b", { text: "If today's candle closed now: " }),
    parts.length ? parts.flatMap((part, i) => (i ? [" · ", ...part] : part)) : "the bot would not buy or sell anything.");
}

function renderRadar(report) {
  const body = $("radar").tBodies[0];
  if (!report) {
    replace($("heatmap"));
    replace($("radar-preview"));
    replace(body, el("tr", { class: "empty" }, el("td", { colspan: "14", text: "No data yet." })));
    return;
  }
  const p = report.parameters;
  const now = (snapshot.radar && snapshot.radar.rows) || {};
  $("th-cycle").textContent = `Last cycle · ${report.cycle_date}`;
  $("th-cycle").title = "What the bot decided in its last cycle, with the closed daily candle of the previous day";
  renderPreview(snapshot);

  replace($("heatmap"), report.basket.map((row) => {
    const live = now[row.symbol];
    const proximity = live ? live.proximity_pct : row.proximity_pct;
    const tile = el("button", {
      type: "button",
      class: `tile ${heatClass(proximity)}${row.symbol === selected ? " selected" : ""}${live ? tick(row.symbol, live.price) : ""}`,
      title: (STATES[row.state] || [row.state])[0],
    },
      el("span", { class: "tile-top" }, el("b", { text: coinOf(row.symbol) }),
        live ? el("i", { class: tone(live.change_pct), text: `${signed(live.change_pct)} %` }) : null),
      el("span", { class: "tile-now", text: isNumber(proximity) ? `${signed(proximity)} %` : "no data" }),
      el("span", { class: "tile-last", text: isNumber(row.proximity_pct) ? `${live ? "close" : "at close"} ${signed(row.proximity_pct)} %` : "" }),
    );
    tile.addEventListener("click", () => select(row.symbol));
    return tile;
  }));

  const entries = listEntries(report, now);
  renderBreadth(entries);
  const shown = entries.filter(FILTERS[listFilter] || FILTERS.all)
    .filter((entry) => !listSearch || entry.symbol.includes(listSearch))
    .sort((a, b) => compareEntries(a, b));
  $("list-count").textContent = shown.length === entries.length ? `${entries.length} coins` : `${shown.length} of ${entries.length}`;
  for (const th of $("list-head").children) {
    const active = th.dataset.sort === listSort.key;
    th.classList.toggle("sorted", active);
    th.setAttribute("aria-sort", active ? (listSort.direction > 0 ? "ascending" : "descending") : "none");
  }
  replace(body, shown.length ? shown.map((entry) => {
    const { row, live } = entry;
    const [label, kind] = STATES[row.state] || [row.state, "none"];
    const line = el("tr", { class: row.symbol === selected ? "selected" : null, "data-symbol": row.symbol },
      el("td", { class: "num rank", text: isNumber(entry.rank) ? String(entry.rank) : "—" }),
      el("td", { class: "coin" }, el("b", { text: entry.coin }),
        row.state === "IN_POSITION" || row.state === "CANDIDATE" ? [" ", el("span", { class: `chip ${kind}`, text: label })] : null,
        live && live.breaking ? [" ", el("span", { class: "chip cand", text: "Breaking out now" })] : null,
        live && live.exiting ? [" ", el("span", { class: "chip bad", text: "Below channel" })] : null),
      el("td", { class: `num now${live ? tick(row.symbol, live.price) : ""}`, text: price(entry.price) }),
      el("td", { class: `num now ${tone(entry.day)}`, text: percent(entry.day) }),
      el("td", { class: `num now ${tone(entry.h24)}`, text: percent(entry.h24) }),
      el("td", { class: `num now ${tone(entry.d7)}`, text: percent(entry.d7) }),
      el("td", { class: `num now ${tone(entry.d30)}`, text: percent(entry.d30) }),
      el("td", { class: "num now", text: volume(entry.volume) }),
      el("td", { class: "num now", text: isNumber(entry.funding) ? `${signed(entry.funding, 4)} %` : "—" }),
      cell(`num now ${live ? heatClass(live.proximity_pct) : ""}`, live ? `${signed(live.proximity_pct)} %` : "—",
        live ? `from ${price(live.entry_channel)}` : null),
      el("td", { class: "now spark" }, sparkline(entry.spark, entry.d30)),
      el("td", { class: `num ${heatClass(row.proximity_pct)}`, text: isNumber(row.proximity_pct) ? `${signed(row.proximity_pct)} %` : "—" }),
      el("td", { class: `num ${tone(row.breakout_strength)}`, text: signed(row.breakout_strength) }),
      cell("mono", row.decision, row.execution && row.execution.status ? `${row.reason} · ${row.execution.status}` : row.reason),
    );
    line.addEventListener("click", () => select(row.symbol));
    return line;
  }) : el("tr", { class: "empty" }, el("td", { colspan: "14", text: "No coin matches the filter." })));
  lastPrices = Object.fromEntries(Object.entries(now).map(([symbol, live]) => [symbol, live.price]));
}

// --- Inspector --------------------------------------------------------------

const TIMEFRAMES = [["5m", "5 min"], ["15m", "15 min"], ["1h", "1 h"], ["4h", "4 h"], ["1d", "1 day"]];
let timeframe = "1d";
let fetched = null;      // { key, data, failed }: candles read from the API for a symbol and an interval
let fitted = null;       // what is fitted on screen, so the chart does not move on every refresh
let levelLines = [];
const chartKey = () => `${selected}:${timeframe}`;

function ensureChart() {
  if (chart) return;
  const style = getComputedStyle(document.documentElement);
  const token = (name) => style.getPropertyValue(name).trim();
  chart = LC.createChart($("chart"), {
    autoSize: true,
    layout: { background: { color: "transparent" }, textColor: token("--ink-2"), fontFamily: token("--mono") },
    grid: { vertLines: { color: token("--rule") }, horzLines: { color: token("--rule") } },
    rightPriceScale: { borderColor: token("--rule") },
    timeScale: { borderColor: token("--rule"), rightOffset: 4, secondsVisible: false },
    crosshair: { mode: LC.CrosshairMode.Normal },
  });
  const line = (color) => chart.addLineSeries({ color, lineWidth: 2, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
  series = {
    volume: chart.addHistogramSeries({ priceFormat: { type: "volume" }, priceScaleId: "", priceLineVisible: false, lastValueVisible: false }),
    candles: chart.addCandlestickSeries({
      upColor: token("--good"), downColor: token("--bad"), borderVisible: false,
      wickUpColor: token("--good"), wickDownColor: token("--bad"), priceLineVisible: false,
    }),
    entry: line(token("--accent")),
    exit: line(token("--info")),
  };
  // Volume goes in the bottom strip, without covering the candles.
  chart.priceScale("").applyOptions({ scaleMargins: { top: 0.84, bottom: 0 } });

  replace($("timeframes"), TIMEFRAMES.map(([value, label]) => {
    const button = el("button", { type: "button", "data-tf": value, "aria-pressed": String(value === timeframe), text: label });
    button.addEventListener("click", () => {
      timeframe = value;
      for (const each of $("timeframes").children) each.setAttribute("aria-pressed", String(each.dataset.tf === value));
      redrawChart();
      loadCandles();
    });
    return button;
  }));
}

const points = (time, values) => time.map((t, i) => ({ time: t, value: values[i] })).filter((point) => isNumber(point.value));

function level(value, color, title, style, width = 1) {
  if (!isNumber(value)) return;
  levelLines.push(series.candles.createPriceLine({ price: value, color, lineWidth: width, lineStyle: style, axisLabelVisible: true, title }));
}

// Draws the candles of the chosen interval with the levels the bot cares about: the two channels, the stop and the buy.
function renderChart(row, daily, position) {
  ensureChart();
  for (const line of levelLines) series.candles.removePriceLine(line);
  levelLines = [];
  const token = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const intraday = timeframe !== "1d";
  const live = fetched && fetched.key === chartKey() && fetched.data ? fetched.data : null;
  const failed = Boolean(fetched && fetched.key === chartKey() && fetched.failed);
  const now = snapshot && snapshot.radar ? snapshot.radar.rows[selected] : null;
  // On daily, until the current candles arrive, the closed ones the cycle used will do.
  const candles = live || (intraday ? null : daily);
  const label = (TIMEFRAMES.find(([value]) => value === timeframe) || [])[1];
  if (!candles) {
    for (const each of Object.values(series)) each.setData([]);
    $("chart-note").textContent = !selected ? "" : failed ? `Could not read the ${label} candles.`
      : intraday ? `Reading the ${label} candles…` : "No candles for this symbol.";
    return;
  }
  const decimals = priceDecimals(candles.close[candles.close.length - 1]);
  const format = { priceFormat: { type: "price", precision: decimals, minMove: 1 / 10 ** decimals } };
  for (const name of ["candles", "entry", "exit"]) series[name].applyOptions(format);
  series.candles.setData(candles.time.map((t, i) => ({
    time: t, open: candles.open[i], high: candles.high[i], low: candles.low[i], close: candles.close[i],
  })));
  series.volume.setData(candles.volume ? candles.time.map((t, i) => ({
    time: t, value: candles.volume[i],
    color: candles.close[i] >= candles.open[i] ? "rgba(51, 194, 127, 0.35)" : "rgba(240, 97, 109, 0.35)",
  })) : []);

  if (intraday) {
    // Intraday the channels are two fixed levels: the ones the next cycle will use.
    series.entry.setData([]);
    series.exit.setData([]);
    const periods = snapshot && snapshot.report ? snapshot.report.parameters : null;
    level(now ? now.entry_channel : row && row.high_entry, token("--accent"), periods ? `${periods.entry_period} d high` : "high", LC.LineStyle.Solid, 2);
    level(now ? now.exit_channel : row && row.low_exit, token("--info"), periods ? `${periods.exit_period} d low` : "low", LC.LineStyle.Solid, 2);
  } else {
    const today = live && daily && live.time[live.time.length - 1] > daily.time[daily.time.length - 1]
      ? live.time[live.time.length - 1] : null;
    const extended = (values, value) => points(daily.time, values).concat(today && isNumber(value) ? [{ time: today, value }] : []);
    series.entry.setData(daily ? extended(daily.entry_channel, now && now.entry_channel) : []);
    series.exit.setData(daily ? extended(daily.exit_channel, now && now.exit_channel) : []);
    if (!live && now) level(now.price, token("--ink"), "now", LC.LineStyle.Dotted);
  }

  const held = position && isNumber(position.stop_price);
  const stop = held ? position.stop_price : !intraday && row && row.sizing ? row.sizing.stop_price_rounded : null;
  $("legend-stop").textContent = held ? "Position stop" : "Stop if it entered today (close − 2 ATR)";
  $("legend-stop").previousElementSibling.classList.toggle("placed", Boolean(held));
  level(stop, token("--bad"), held ? `active stop ${price(stop)}` : "stop", held ? LC.LineStyle.Solid : LC.LineStyle.Dashed, held ? 3 : 2);
  // The real entry of an open position, as visible as its stop, on every interval.
  if (position) level(position.entry_price, token("--good"), `entry ${price(position.entry_price)}`, LC.LineStyle.Solid, 2);
  $("legend-buy").hidden = !position;

  $("chart-note").textContent = intraday ? `${label} candles, UTC time. The last one is still open.`
    : live ? "Daily candles. The last one is today's, still open; the bot will decide when it closes."
    : failed ? "Closed daily candles from the last cycle (the current ones could not be read)."
    : "Closed daily candles from the last cycle.";
  chart.applyOptions({ timeScale: { timeVisible: intraday } });
  const shown = `${chartKey()}:${live ? "now" : "cycle"}`;
  if (shown !== fitted) {
    chart.timeScale().fitContent();
    fitted = shown;
  }
}

function redrawChart() {
  const report = snapshot && snapshot.report;
  if (!report || !selected) return;
  renderChart(report.basket.find((each) => each.symbol === selected), snapshot.market[selected],
    (snapshot.positions || []).find((each) => each.symbol === selected) || report.positions.find((each) => each.symbol === selected));
}

// Asks the public API for the candles of the symbol and the interval being viewed.
async function loadCandles() {
  if (!selected || !snapshot || !snapshot.report) return;
  const key = chartKey();
  let data = null;
  try {
    const response = await fetch(`/api/candles?symbol=${encodeURIComponent(selected)}&interval=${timeframe}`, { cache: "no-store" });
    if (response.ok) data = await response.json();
  } catch (error) {
    data = null;
  }
  if (key !== chartKey()) return;  // the coin or the interval changed in the meantime
  // If a re-read fails, what was already drawn is kept.
  if (data) fetched = { key, data, failed: false };
  else if (!fetched || fetched.key !== key) fetched = { key, data: null, failed: true };
  redrawChart();
}

function step(term, formula, result, verdict) {
  return [el("dt", { text: term }), el("dd", {}, el("span", { class: "formula", text: formula }), " = ", el("b", { text: result }),
    verdict ? [" ", el("span", { class: verdict[1] ? "ok" : "no", text: verdict[0] })] : null)];
}

function renderMath(report, row, position) {
  const s = row && row.sizing;
  if (!s) {
    $("math-caption").textContent = row ? "No candles or no trading rules: it cannot be sized." : "";
    replace($("math"));
    replace($("position"));
    return;
  }
  const p = report.parameters;
  $("math-caption").textContent = row.decision === "ENTER_LONG"
    ? "This is how this cycle's entry was sized."
    : "What it would come to if the symbol entered today, with the same steps the bot uses.";
  replace($("math"),
    step("Distance to the stop", `${number(s.stop_atr_multiple, 0)} × ATR(${p.atr_period}) = ${number(s.stop_atr_multiple, 0)} × ${price(row.atr)}`, price(s.stop_distance)),
    step("Risk in money", `Balance × ${number(s.risk_pct, 2)} % = ${number(s.balance)} × ${number(s.risk_pct / 100, 4)}`, usd(s.risk_budget_usd)),
    step("Units by risk", `${number(s.risk_budget_usd)} / ${price(s.stop_distance)}`, units(Number(s.lots_by_risk.toPrecision(7)))),
    step(`Units by cap (${number(s.max_position_leverage * 100, 0)} % of the balance)`,
      `${number(s.balance * s.max_position_leverage)} / ${price(row.close)}`, units(Number(s.lots_by_leverage.toPrecision(7)))),
    step("Calculated lots → final lots", `${units(Number(s.lots_raw.toPrecision(7)))} at step ${units(s.step_size)}, limited by ${s.limited_by}`, units(s.lots_final)),
    step("Notional against the minimum", `${units(s.lots_final)} × ${price(row.close)} = ${number(s.nominal_usd)} ; minimum ${number(s.min_notional_usd)}`,
      usd(s.nominal_usd), [s.meets_min_notional ? "meets it" : "does not meet it", s.meets_min_notional]),
    step("Stop price", `${price(row.close)} − ${price(s.stop_distance)} = ${price(s.stop_price)} at tick ${units(s.tick_size)}`, price(s.stop_price_rounded)),
    step("Loss if the stop is hit", `${units(s.lots_final)} × ${price(s.stop_distance)}`, `${usd(s.risk_at_stop_usd)} (${number(s.risk_at_stop_pct)} %)`),
  );
  renderExitCard(report, position);
}

// The two prices at which an open position will be sold without anyone touching it, and which comes first.
function renderExitCard(report, position) {
  if (!position) {
    replace($("position"));
    return;
  }
  const now = (snapshot.positions || []).find((each) => each.symbol === position.symbol) || {};
  const current = isNumber(now.price) ? now.price : position.mark_price;
  const channel = isNumber(now.exit_channel) ? now.exit_channel : position.exit_channel;
  const exitPeriod = report.parameters.exit_period;
  const distance = (level) => (isNumber(level) && current > 0 ? `${signed((level / current - 1) * 100)} %` : "—");
  const stopFirst = isNumber(position.stop_price) && (!isNumber(channel) || position.stop_price >= channel);
  const pnl = isNumber(now.pnl_usd) ? now.pnl_usd : position.unrealized_pnl;
  const r = isNumber(position.risk_usd) && position.risk_usd > 0 ? pnl / position.risk_usd : null;
  const item = (term, value, detail, first) => el("div", { class: `exit-level${first ? " first" : ""}` },
    el("dt", { text: term }), el("dd", {}, el("b", { text: value }), el("span", { text: detail })));
  replace($("position"), el("section", { class: "exit-card", "aria-label": "Automatic sell levels" },
    el("h4", { text: "Open position: automatic sell levels" }),
    el("p", { class: "exit-summary" },
      `${units(position.quantity)} units · entry ${price(position.entry_price)} · now ${price(current)} · `,
      el("span", { class: tone(pnl), text: `${signed(pnl)} USDT` }),
      isNumber(r) ? el("span", { class: tone(r), text: ` (${signed(r)}R)` }) : null),
    el("dl", {},
      item(`Stop (${number(report.parameters.stop_atr_multiple, 0)} ATR)`,
        isNumber(position.stop_price) ? `${price(position.stop_price)} USDT` : "no stop",
        `${distance(position.stop_price)} from the current price. An order on the exchange: it sells as soon as the price touches it.`,
        stopFirst),
      item(`${exitPeriod}-day channel (low ${exitPeriod})`, isNumber(channel) ? `${price(channel)} USDT` : "—",
        `${distance(channel)} from the current price. The low of the last ${exitPeriod} candles: it rises as old lows leave the channel, and it sells when a daily candle closes below it.`,
        !stopFirst && isNumber(channel))),
    el("p", { class: "hint", text: stopFirst
      ? "Today the stop is above the channel: it would sell first."
      : "Today the channel is above the stop: it would sell first, at the daily close." })));
}

function renderInspector() {
  const report = snapshot && snapshot.report;
  const picker = $("symbol");
  if (!report) {
    replace(picker);
    renderChart(null, null, null);
    renderMath(null, null, null);
    return;
  }
  const symbols = report.basket.map((row) => row.symbol);
  if (!symbols.includes(selected)) {
    // By default, the most interesting one: an open position or, failing that, the closest to breaking out.
    const ranked = [...report.basket].sort((a, b) => (b.proximity_pct ?? -Infinity) - (a.proximity_pct ?? -Infinity));
    selected = (report.basket.find((row) => row.state === "IN_POSITION") || ranked[0] || {}).symbol || null;
  }
  if (picker.options.length !== symbols.length || [...picker.options].some((option, i) => option.value !== symbols[i])) {
    replace(picker, symbols.map((symbol) => el("option", { value: symbol, text: symbol })));
  }
  picker.value = selected;
  const p = report.parameters;
  $("legend-entry").textContent = `Entry channel: high of ${p.entry_period} candles`;
  $("legend-exit").textContent = `Exit channel: low of ${p.exit_period} candles`;
  const row = report.basket.find((each) => each.symbol === selected);
  // The position's P&L, at the current price if it could be read; if not, the last cycle's.
  const now = (snapshot.positions || []).find((each) => each.symbol === selected);
  const last = report.positions.find((each) => each.symbol === selected);
  const position = last ? { ...last, pnl: now ? now.pnl_usd : last.unrealized_pnl } : null;
  redrawChart();
  renderMath(report, row, position);
  $("levels").textContent = row && isNumber(row.close)
    ? `Last cycle: close ${price(row.close)} · high of ${p.entry_period} candles ${price(row.high_entry)} · ` +
      `low of ${p.exit_period} candles ${price(row.low_exit)} · ATR(${p.atr_period}) ${price(row.atr)}`
    : "";
}

function select(symbol) {
  selected = symbol;
  renderRadar(snapshot.report);
  renderInspector();
  loadCandles();
  runHooks("select");
}

// --- Audit ------------------------------------------------------------------

const names = (orders) => (orders || []).map((order) => `${coinOf(order.symbol)}${order.status && !["FILLED", "SIMULATED"].includes(order.status) ? ` (${order.status})` : ""}`).join(", ") || "—";

function renderAudit(data) {
  const available = new Set(data.meta.report_dates || []);
  const history = data.history || [];
  replace($("history").tBodies[0], history.length ? history.map((cycle) => el("tr", {},
    el("td", { class: "mono", text: cycle.cycle_date }),
    el("td", { class: "mono", text: cycle.status || "—" }),
    el("td", { class: "mono", text: names(cycle.opens) }),
    el("td", { class: "mono", text: names(cycle.closes) + (cycle.stopped && cycle.stopped.length ? ` · stop: ${cycle.stopped.map(coinOf).join(", ")}` : "") }),
    el("td", { class: "mono", text: Object.entries(cycle.holds || {}).map(([reason, count]) => `${reason} ${count}`).join(" · ") || "—" }),
    el("td", {}, available.has(cycle.cycle_date)
      ? [el("a", { href: `/api/report.md?date=${cycle.cycle_date}`, download: true, text: "md" }), " · ",
         el("a", { href: `/api/report.json?date=${cycle.cycle_date}`, download: true, text: "json" })]
      : "—"),
  )) : el("tr", { class: "empty" }, el("td", { colspan: "6", text: "No cycle has run yet." })));

  const trades = data.trades || [];
  replace($("trades").tBodies[0], trades.length ? trades.map((trade) => el("tr", {},
    el("td", { class: "mono", text: trade.symbol }),
    el("td", { class: "mono", text: new Date(trade.closed_at).toISOString().slice(0, 10) }),
    el("td", { class: "mono", text: trade.reason }),
    el("td", { class: "num", text: price(trade.entry_price) }),
    el("td", { class: "num", text: price(trade.exit_price) }),
    el("td", { class: `num ${tone(trade.pnl)}`, text: `${signed(trade.pnl)} USDT` }),
  )) : el("tr", { class: "empty" }, el("td", { colspan: "6", text: "No closed trades yet." })));
}

// --- Refresh cycle ----------------------------------------------------------

function render(data) {
  snapshot = data;
  renderTop(data.report, data.meta);
  renderNextCycle(data.meta);
  renderPortfolio(data.report, data.live);
  renderPositions(data);
  renderCurve(data.performance);
  renderRadar(data.report);
  renderInspector();
  renderAudit(data);
  loadCandles();
  runHooks("full");
}

// A request that hangs is abandoned, and a refresh never starts while the previous one is still out:
// with a stalled server, requests would otherwise pile up and come back in any order.
const REQUEST_TIMEOUT_MS = 15000;
const inFlight = new Set();

async function fetchJson(path) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const response = await fetch(path, { cache: "no-store", signal: controller.signal });
    if (!response.ok) throw new Error(`the server responded ${response.status}`);
    return await response.json();
  } catch (error) {
    throw error.name === "AbortError" ? new Error("the server did not answer in time") : error;
  } finally {
    clearTimeout(timer);
  }
}

async function once(name, task) {
  if (inFlight.has(name)) return;
  inFlight.add(name);
  try {
    await task();
  } finally {
    inFlight.delete(name);
  }
}

function refresh() {
  return once("full", async () => {
    try {
      render(await fetchJson("/api/snapshot"));
    } catch (error) {
      replace($("notices"), el("p", { text: `Could not update the dashboard: ${error.message}` }));
    }
  });
}

// Between two full reads: only what moves with the price.
function refreshLive() {
  if (!snapshot || document.hidden) return undefined;
  return once("live", async () => {
    try {
      Object.assign(snapshot, await fetchJson("/api/live"));
    } catch (error) {
      return;  // the dot at the top already shows that the prices are going stale
    }
    renderNextCycle(snapshot.meta);
    renderPortfolio(snapshot.report, snapshot.live);
    renderPositions(snapshot);
    renderCurve(snapshot.performance);
    renderRadar(snapshot.report);
    const held = snapshot.report && snapshot.report.positions.find((each) => each.symbol === selected);
    if (held) renderExitCard(snapshot.report, held);
    loadCandles();
    runHooks("live");
  });
}

$("symbol").addEventListener("change", (event) => select(event.target.value));
for (const th of $("list-head").querySelectorAll("[data-sort]")) {
  th.addEventListener("click", () => {
    const key = th.dataset.sort;
    // Text starts from A to Z; figures, from high to low (the rank, from first to last).
    const first = ["symbol", "decision", "rank"].includes(key) ? 1 : -1;
    listSort = { key, direction: listSort.key === key ? -listSort.direction : first };
    if (snapshot) renderRadar(snapshot.report);
  });
}
for (const button of $("list-filter").children) {
  button.addEventListener("click", () => {
    listFilter = button.dataset.filter;
    for (const each of $("list-filter").children) each.setAttribute("aria-pressed", String(each === button));
    if (snapshot) renderRadar(snapshot.report);
  });
}
$("list-search").addEventListener("input", (event) => {
  listSearch = event.target.value.trim().toUpperCase();
  if (snapshot) renderRadar(snapshot.report);
});
renderClocks();
setInterval(renderClocks, 1000);
refresh();
setInterval(() => { if (!document.hidden) refresh(); }, REFRESH_MS);
setInterval(refreshLive, LIVE_MS);
// A tab left in the background stops asking; coming back brings it up to date at once.
document.addEventListener("visibilitychange", () => { if (!document.hidden) refresh(); });
