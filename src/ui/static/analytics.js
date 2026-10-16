// Portfolio analytics: closed trades, P&L by coin, execution and correlation.
"use strict";

let corrScope = "focus";
let corrDays = 60;
const CORR_MAJORS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"];
const ratio = (value) => (isNumber(value) ? number(value, 2) : "—");

function renderTradeStats(stats) {
  const none = !stats || !stats.count;
  $("analytics-hint").textContent = none
    ? "No closed trades yet: these figures fill in with the first one."
    : `${stats.count} closed trades · ${stats.wins} with a gain and ${stats.losses} with a loss`;
  const s = stats || {};
  const reasons = Object.entries(s.by_reason || {}).map(([reason, bucket]) =>
    `${reason === "STOP" ? "by stop" : reason === "EXIT_CHANNEL_BROKEN" ? "by channel" : reason} ${bucket.count}`).join(" · ");
  replace($("trade-stats"),
    metric("Closed trades", String(s.count || 0), reasons || "None yet"),
    metric("Win rate", [isNumber(s.win_rate_pct) ? number(s.win_rate_pct, 0) : "—", el("small", { text: " %" })],
      none ? "Trades that closed with a gain" : `${s.wins} of ${s.count}`),
    metric("Closed P&L", el("span", { class: tone(s.net_usd), text: none ? "—" : `${signed(s.net_usd)} USDT` }),
      none ? "Sum of the closed trades" : `Best ${signed(s.best_usd)} · worst ${signed(s.worst_usd)}`),
    metric("Profit factor", ratio(s.profit_factor),
      isNumber(s.profit_factor) ? "Gains divided by losses; above 1 it makes money" : "Gains divided by losses"),
    metric("Average gain and loss", isNumber(s.payoff_ratio) ? `${ratio(s.payoff_ratio)} to 1` : "—",
      `${isNumber(s.average_win_usd) ? signed(s.average_win_usd) : "—"} / ${isNumber(s.average_loss_usd) ? signed(s.average_loss_usd) : "—"} USDT`),
    metric("Expectancy per trade", el("span", { class: tone(s.expectancy_usd), text: isNumber(s.expectancy_usd) ? `${signed(s.expectancy_usd)} USDT` : "—" }),
      isNumber(s.expectancy_r) ? `${signed(s.expectancy_r)} R (multiples of the initial risk)` : "In USDT and in multiples of the initial risk"),
    metric("Days in position", isNumber(s.average_days_held) ? number(s.average_days_held, 1) : "—", "On average per trade"),
    metric("Fees and funding", [number((s.fees_usd || 0) + (s.funding_usd || 0)), el("small", { text: " USDT" })],
      `Fees ${number(s.fees_usd || 0)} · funding ${signed(s.funding_usd || 0)}`),
  );
}

function renderBySymbol(rows) {
  replace($("by-symbol").tBodies[0], rows && rows.length ? rows.map((row) => {
    const line = el("tr", {},
      el("td", { class: "mono" }, el("b", { text: coinOf(row.symbol) }), row.is_open ? [" ", el("span", { class: "chip pos", text: "open" })] : null),
      el("td", { class: "num", text: String(row.trades) }),
      el("td", { class: "num", text: row.trades ? `${row.wins} of ${row.trades}` : "—" }),
      el("td", { class: `num ${tone(row.realized_usd)}`, text: row.trades ? signed(row.realized_usd) : "—" }),
      el("td", { class: `num ${tone(row.open_usd)}`, text: row.is_open ? signed(row.open_usd) : "—" }),
      el("td", { class: `num ${tone(row.total_usd)}`, text: `${signed(row.total_usd)} USDT` }),
    );
    line.addEventListener("click", () => select(row.symbol));
    return line;
  }) : el("tr", { class: "empty" }, el("td", { colspan: "6", text: "No coin has had a position yet." })));
}

function renderExecution(quality) {
  const all = quality && quality.all;
  const part = (label, summary) => (summary && summary.count
    ? `${label}: ${signed(summary.mean_pct, 3)} % on average over ${summary.count}` : null);
  $("execution-summary").textContent = all && all.count
    ? [`Average cost against the signal close: ${signed(all.mean_pct, 3)} % (worst: ${signed(all.worst_pct, 3)} %)`,
      part("buys", quality.entries), part("sells", quality.exits)].filter(Boolean).join(" · ")
    : "No order has been executed yet.";
  const rows = (quality && quality.recent) || [];
  replace($("executions").tBodies[0], rows.length ? rows.map((note) => el("tr", {},
    el("td", { class: "mono", text: note.date || "—" }),
    el("td", { class: "mono", text: coinOf(note.symbol) }),
    el("td", { text: note.action === "OPEN_LONG" ? "Buy" : "Sell" }),
    el("td", { class: "num", text: price(note.signal_close) }),
    el("td", { class: "num", text: price(note.price) }),
    // Positive is against: buying above the close or selling below it.
    el("td", { class: `num ${tone(-note.cost_pct)}`, text: `${signed(note.cost_pct, 3)} %` }),
  )) : el("tr", { class: "empty" }, el("td", { colspan: "6", text: "No executed orders." })));
}

// --- Correlation ------------------------------------------------------------

function dailyReturns(symbol, days) {
  const closes = ((snapshot.market[symbol] || {}).close || []).filter(isNumber);
  const window = closes.slice(-(days + 1));
  return window.length === days + 1 ? window.slice(1).map((close, i) => Math.log(close / window[i])) : null;
}

function pearson(a, b) {
  const mean = (values) => values.reduce((sum, value) => sum + value, 0) / values.length;
  const [ma, mb] = [mean(a), mean(b)];
  let [cov, va, vb] = [0, 0, 0];
  for (let i = 0; i < a.length; i += 1) {
    cov += (a[i] - ma) * (b[i] - mb);
    va += (a[i] - ma) ** 2;
    vb += (b[i] - mb) ** 2;
  }
  return va > 0 && vb > 0 ? cov / Math.sqrt(va * vb) : null;
}

function correlationSymbols() {
  const basket = snapshot.report.basket.map((row) => row.symbol);
  if (corrScope === "all") return basket;
  const held = (snapshot.positions || []).map((position) => position.symbol);
  const focus = [...new Set([...held, ...CORR_MAJORS, selected].filter((symbol) => basket.includes(symbol)))];
  return focus;
}

function renderCorrelation() {
  const node = $("corr-matrix");
  if (!snapshot || !snapshot.report) {
    replace(node);
    $("corr-hint").textContent = "";
    return;
  }
  const series = correlationSymbols().map((symbol) => [symbol, dailyReturns(symbol, corrDays)]).filter(([, values]) => values);
  if (series.length < 2) {
    replace(node);
    $("corr-hint").textContent = `There are not ${corrDays} days of candles to compute it.`;
    return;
  }
  const held = new Set((snapshot.positions || []).map((position) => position.symbol));
  const matrix = series.map(([, a]) => series.map(([, b]) => pearson(a, b)));
  const pairs = [];
  const heldPairs = [];
  series.forEach(([a], i) => series.forEach(([b], j) => {
    if (j <= i || !isNumber(matrix[i][j])) return;
    pairs.push(matrix[i][j]);
    if (held.has(a) && held.has(b)) heldPairs.push(matrix[i][j]);
  }));
  const mean = (values) => (values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null);
  $("corr-hint").textContent = [
    `Daily returns over the last ${corrDays} days.`,
    `Matrix mean: ${signed(mean(pairs))}.`,
    heldPairs.length ? `Among the open positions: ${signed(mean(heldPairs))}.` : null,
  ].filter(Boolean).join(" ");

  const compact = series.length > 16;
  const table = el("table", { class: `corr-table${compact ? " compact" : ""}` });
  table.append(el("tr", {}, el("th", {}), series.map(([symbol]) => el("th", { class: `colhead${held.has(symbol) ? " held" : ""}`, text: coinOf(symbol) }))));
  series.forEach(([symbol], i) => {
    table.append(el("tr", {}, el("th", { class: `rowhead${held.has(symbol) ? " held" : ""}`, text: coinOf(symbol) }),
      series.map(([other], j) => {
        const value = matrix[i][j];
        // Amber: they move together (concentrated risk). Blue: they move in opposite directions.
        const color = !isNumber(value) || i === j ? "transparent"
          : value >= 0 ? `rgba(240, 180, 41, ${(value * 0.85).toFixed(2)})` : `rgba(90, 169, 230, ${(-value * 0.85).toFixed(2)})`;
        return el("td", {
          style: `background:${color}`, class: i === j ? "self" : null,
          title: `${coinOf(symbol)} · ${coinOf(other)}: ${isNumber(value) ? signed(value) : "no data"}`,
          text: compact || i === j || !isNumber(value) ? "" : number(value, 2),
        });
      })));
  });
  replace(node, table);
}

function renderAnalytics() {
  const analytics = (snapshot && snapshot.analytics) || {};
  renderTradeStats(analytics.trades);
  renderBySymbol(analytics.symbols);
  renderExecution(analytics.execution);
  renderCorrelation();
}

for (const [group, attribute, apply] of [
  ["corr-scope", "scope", (value) => { corrScope = value; }],
  ["corr-days", "days", (value) => { corrDays = Number(value); }],
]) {
  for (const button of $(group).children) {
    button.addEventListener("click", () => {
      apply(button.dataset[attribute]);
      for (const each of $(group).children) each.setAttribute("aria-pressed", String(each === button));
      renderCorrelation();
    });
  }
}
hooks.full.push(renderAnalytics);
hooks.select.push(renderCorrelation);
