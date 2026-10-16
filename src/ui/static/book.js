// Order book, trade tape and derivatives of the coin being viewed.
// The browser listens to the book and the tape live on Binance's public streams;
// if it cannot, they are fetched from the server every few seconds.
"use strict";

const BOOK_LEVELS = 10;
const TAPE_ROWS = 28;
const BOOK_POLL_MS = 3000;
const DERIVATIVES_MS = 60000;
// When the book goes out of view (scrolling further down, switching tabs) the connection holds this long:
// that way coming straight back neither drops it nor reopens it.
const BOOK_GRACE_MS = 45000;
// Holding j/k flips through coins many times a second: only the coin it lands on opens connections.
const SELECT_SETTLE_MS = 250;
let book = { symbol: null, bids: [], asks: [], trades: [], mode: "off", dirty: false };
let sockets = [];
let bookPoll = null;
let pollInFlight = false;
let selectSettle = null;
let bookVisible = false;
let bookPause = null;
let streamWorked = false;
let derivatives = { symbol: null, data: null, at: 0 };
const CLOCK_UTC = new Intl.DateTimeFormat("en-US", { timeZone: "UTC", hour: "2-digit", minute: "2-digit", second: "2-digit", hourCycle: "h23" });

function stopBook() {
  for (const socket of sockets) {
    socket.onclose = null;
    socket.onerror = null;
    socket.close();
  }
  sockets = [];
  if (bookPoll) clearInterval(bookPoll);
  bookPoll = null;
}

function pushTrade(trade) {
  book.trades.unshift(trade);
  if (book.trades.length > TAPE_ROWS) book.trades.length = TAPE_ROWS;
  book.dirty = true;
}

function onStream(symbol, message) {
  if (symbol !== book.symbol) return;
  const data = JSON.parse(message.data).data;
  if (!data) return;
  if (data.e === "depthUpdate") {
    book.bids = data.b.map(([p, q]) => [Number(p), Number(q)]);
    book.asks = data.a.map(([p, q]) => [Number(p), Number(q)]);
    book.dirty = true;
  } else if (data.e === "aggTrade") {
    // m: the buyer is the one who had the resting order, so the aggressor is selling.
    pushTrade({ time: data.T, price: Number(data.p), quantity: Number(data.q), sell: Boolean(data.m) });
  } else if (data.e === "forceOrder" && data.o) {
    pushTrade({ time: data.o.T, price: Number(data.o.ap || data.o.p), quantity: Number(data.o.q), sell: data.o.S === "SELL", liquidation: true });
  }
}

async function pollBook(symbol) {
  if (pollInFlight) return;  // a slow server must not pile up requests
  pollInFlight = true;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), BOOK_POLL_MS * 3);
  try {
    const response = await fetch(`/api/book?symbol=${encodeURIComponent(symbol)}`, { cache: "no-store", signal: controller.signal });
    if (!response.ok || symbol !== book.symbol) return;
    const data = await response.json();
    // The coin changed, or the live stream opened, while the answer was on its way.
    if (symbol !== book.symbol || book.mode === "stream") return;
    book.bids = data.bids;
    book.asks = data.asks;
    book.trades = data.trades.slice().reverse().slice(0, TAPE_ROWS);
    book.mode = "poll";  // back from a failed poll
    book.dirty = true;
  } catch (error) {
    if (symbol === book.symbol && book.mode !== "stream") {
      book.mode = "off";
      book.dirty = true;
    }
  } finally {
    clearTimeout(timer);
    pollInFlight = false;
  }
}

function startPolling(symbol) {
  stopBook();
  book.mode = "poll";
  pollBook(symbol);
  bookPoll = setInterval(() => pollBook(symbol), BOOK_POLL_MS);
}

const bookWanted = () => bookVisible && !document.hidden;
const bookConnected = () => sockets.length > 0 || bookPoll !== null;

// Opens the book of the coin being viewed. If it is the same one as before, what is on
// screen is kept while the new data arrives: reconnecting does not clear the book or the tape.
function startBook() {
  stopBook();
  const symbol = selected;
  if (symbol !== book.symbol) book = { symbol, bids: [], asks: [], trades: [], mode: "off", dirty: true };
  book.mode = "off";
  book.dirty = true;
  if (!symbol || !bookWanted() || !snapshot || !snapshot.report) return;
  book.mode = "connecting";
  const base = snapshot.meta && snapshot.meta.stream;
  if (!base || !("WebSocket" in window)) {
    startPolling(symbol);
    return;
  }
  const stream = symbol.toLowerCase();
  let opened = 0;
  // The book comes over the public channel and the trades over the market one: two connections.
  for (const path of [`/public/stream?streams=${stream}@depth20@100ms`, `/market/stream?streams=${stream}@aggTrade/${stream}@forceOrder`]) {
    let socket;
    try {
      socket = new WebSocket(base + path);
    } catch (error) {
      startPolling(symbol);
      return;
    }
    socket.onopen = () => {
      opened += 1;
      streamWorked = true;
      if (symbol === book.symbol) { book.mode = "stream"; book.dirty = true; }
    };
    socket.onmessage = (message) => onStream(symbol, message);
    // If it never opens (a firewall, the machine's network), it is fetched from the server.
    socket.onerror = () => { if (symbol === book.symbol && !opened) startPolling(symbol); };
    socket.onclose = () => { if (symbol === book.symbol && book.mode === "stream") startPolling(symbol); };
    sockets.push(socket);
  }
}

function depthRow(side, [levelPrice, quantity], cumulative, total) {
  const share = total > 0 ? Math.min(100, (cumulative / total) * 100) : 0;
  return el("div", { class: `book-row ${side}`, style: `--fill:${share.toFixed(1)}%` },
    el("span", { class: "p", text: price(levelPrice) }),
    el("span", { class: "q", text: number(levelPrice * quantity, 0) }),
    el("span", { class: "c", text: number(cumulative, 0) }));
}

function renderBook() {
  const status = $("book-status");
  const pair = book.symbol ? book.symbol.replace(/USDT$/, " / USDT") : "";
  for (const id of ["book-pair", "tape-pair", "derivatives-pair"]) $(id).textContent = pair;
  status.className = `live ${book.mode === "stream" ? "on" : book.mode === "off" ? "off" : "stale"}`;
  status.textContent = { stream: "live", poll: "every 3 s", connecting: "connecting…" }[book.mode]
    || (bookWanted() ? "no data" : "paused");
  const bids = book.bids.slice(0, BOOK_LEVELS);
  const asks = book.asks.slice(0, BOOK_LEVELS);
  if (!bids.length || !asks.length) {
    replace($("book"), el("p", { class: "hint", text: selected ? "Waiting for the book…" : "" }));
  } else {
    const sum = (levels) => levels.reduce((acc, [p, q]) => acc + p * q, 0);
    const [bidTotal, askTotal] = [sum(bids), sum(asks)];
    const scale = Math.max(bidTotal, askTotal);
    let running = 0;
    const askRows = asks.map((level) => { running += level[0] * level[1]; return depthRow("ask", level, running, scale); }).reverse();
    running = 0;
    const bidRows = bids.map((level) => { running += level[0] * level[1]; return depthRow("bid", level, running, scale); });
    const spread = asks[0][0] - bids[0][0];
    const imbalance = bidTotal / (bidTotal + askTotal) * 100;
    replace($("book"),
      el("div", { class: "book-row head" }, el("span", { text: "Price" }), el("span", { text: "USDT" }), el("span", { text: "Cumulative" })),
      askRows,
      el("div", { class: "book-mid" }, el("b", { text: price((asks[0][0] + bids[0][0]) / 2) }),
        el("span", { text: `spread ${number(spread, priceDecimals(bids[0][0]))} (${number(spread / bids[0][0] * 100, 3)} %)` })),
      bidRows,
      el("div", { class: "imbalance", title: "Split of the money resting in the first ten levels of each side" },
        el("i", { style: `width:${imbalance.toFixed(1)}%` }),
        el("span", { text: `buy ${number(imbalance, 0)} % · sell ${number(100 - imbalance, 0)} %` })));
  }
  replace($("tape"), book.trades.length ? [
    el("div", { class: "tape-row head" }, el("span", { text: "UTC time" }), el("span", { text: "Price" }), el("span", { text: "USDT" })),
    book.trades.map((trade) => el("div", { class: `tape-row ${trade.sell ? "down" : "up"}${trade.liquidation ? " liq" : ""}` },
      el("span", { text: CLOCK_UTC.format(new Date(trade.time)) }),
      el("span", { text: price(trade.price) }),
      el("span", { text: `${number(trade.price * trade.quantity, 0)}${trade.liquidation ? " LIQ" : ""}` }))),
  ] : el("p", { class: "hint", text: selected ? "Waiting for trades…" : "" }));
  book.dirty = false;
}

// --- Derivatives ------------------------------------------------------------

function fact(term, value, detail, kind) {
  return [el("dt", { text: term }), el("dd", {}, el("b", { class: kind || null, text: value }), detail ? el("span", { text: detail }) : null)];
}

function renderDerivatives() {
  const live = snapshot && snapshot.radar && snapshot.radar.rows[selected];
  const d = derivatives.symbol === selected ? derivatives.data : null;
  const facts = [];
  if (d && live && isNumber(d.circulating_supply)) {
    facts.push(fact("Market cap", `${volume(d.circulating_supply * live.price)} $`, `${number(d.circulating_supply / 1e6, 1)} M in circulation`));
  }
  if (live && isNumber(live.volume_24h_usd)) facts.push(fact("24 h volume", `${volume(live.volume_24h_usd)} $`, "on this contract"));
  if (d) {
    facts.push(fact("Open interest", `${volume(d.open_interest_usd)} $`,
      isNumber(d.open_interest_change_24h_pct) ? `${signed(d.open_interest_change_24h_pct)} % in 24 h` : null, tone(0)));
    facts.push([el("dt", { text: "Open interest, 24 h" }), el("dd", {}, sparkline(d.open_interest_history, d.open_interest_change_24h_pct))]);
    facts.push(fact("Accounts long", `${number(d.long_accounts_pct, 1)} %`, `${number(d.long_short_ratio, 2)} longs per short`));
    facts.push(fact("Aggressive buys against sells", number(d.taker_buy_sell_ratio, 2),
      d.taker_buy_sell_ratio >= 1 ? "buyers dominate (last hour)" : "sellers dominate (last hour)", tone(d.taker_buy_sell_ratio - 1)));
  }
  if (live && isNumber(live.funding_pct)) {
    facts.push(fact("Funding", `${signed(live.funding_pct, 4)} %`, live.funding_pct >= 0 ? "longs pay it" : "longs receive it"));
  }
  replace($("derivatives"), facts.length ? facts : [el("dt", { text: selected ? "Reading…" : "" }), el("dd", {})]);
}

async function loadDerivatives(force) {
  const symbol = selected;
  if (!symbol || !bookVisible) return;
  if (!force && derivatives.symbol === symbol && Date.now() - derivatives.at < DERIVATIVES_MS) {
    renderDerivatives();
    return;
  }
  if (derivatives.symbol !== symbol) derivatives = { symbol, data: null, at: 0 };
  renderDerivatives();
  try {
    const response = await fetch(`/api/derivatives?symbol=${encodeURIComponent(symbol)}`, { cache: "no-store" });
    if (symbol !== selected) return;
    derivatives = { symbol, data: response.ok ? await response.json() : derivatives.data, at: Date.now() };
  } catch (error) {
    derivatives.at = Date.now();
  }
  renderDerivatives();
}

// It listens while the book is in view and the tab is active. Losing sight of it for a moment
// drops nothing: the connection only closes if BOOK_GRACE_MS passes without it being needed again.
function syncBook() {
  if (bookWanted()) {
    if (bookPause) clearTimeout(bookPause);
    bookPause = null;
    if (!bookConnected() || book.symbol !== selected) startBook();
    loadDerivatives(false);
  } else if (bookConnected() && !bookPause) {
    bookPause = setTimeout(() => {
      bookPause = null;
      if (bookWanted()) return;
      stopBook();
      book.mode = "off";
      book.dirty = true;
    }, BOOK_GRACE_MS);
  }
}

new IntersectionObserver((entries) => {
  bookVisible = entries[entries.length - 1].isIntersecting;
  syncBook();
}).observe($("depth-panel"));
document.addEventListener("visibilitychange", syncBook);
setInterval(() => { if (book.dirty) renderBook(); }, 250);
// Picking another coin does switch books; clicking the same one again does not reconnect.
function switchBook() {
  selectSettle = null;
  if (book.symbol === selected && bookConnected()) return;
  if (bookWanted()) startBook(); else { stopBook(); book = { symbol: selected, bids: [], asks: [], trades: [], mode: "off", dirty: true }; }
  loadDerivatives(true);
}
hooks.select.push(() => {
  if (selectSettle) clearTimeout(selectSettle);
  selectSettle = setTimeout(switchBook, SELECT_SETTLE_MS);
});
// If the book was already in view when the first data arrived, it has not been able to connect yet.
// And if the live connection dropped after having worked, it is retried every minute.
hooks.full.push(() => {
  const dropped = streamWorked && book.mode !== "stream";
  if (bookWanted() && (!bookConnected() || book.symbol !== selected || dropped)) startBook();
  loadDerivatives(false);
});
hooks.live.push(() => loadDerivatives(false));
