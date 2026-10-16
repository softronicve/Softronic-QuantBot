// Market context: market cap, sentiment and other markets at the top; headlines and calendar at the bottom.
"use strict";

const CONTEXT_MS = 60000;
const FUNDING_HOURS = [0, 8, 16];
let context = null;
let newsFilter = "all";

const big = (value) => (!isNumber(value) ? "—"
  : value >= 1e12 ? `${number(value / 1e12, 2)} T` : value >= 1e9 ? `${number(value / 1e9, 1)} B` : `${number(value / 1e6, 0)} M`);
const TIME = new Intl.DateTimeFormat("en-US", { hour: "2-digit", minute: "2-digit", hourCycle: "h23" });
const TIME_UTC = new Intl.DateTimeFormat("en-US", { timeZone: "UTC", hour: "2-digit", minute: "2-digit", hourCycle: "h23" });
const DAY = new Intl.DateTimeFormat("en-US", { weekday: "long", day: "numeric", month: "short" });

function ago(ms) {
  const minutes = Math.max(0, Math.round((Date.now() - ms) / 60000));
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  return hours < 48 ? `${hours} h ago` : `${Math.round(hours / 24)} d ago`;
}

function ticker(label, value, detail, kind) {
  return el("div", { class: "ticker" }, el("small", { text: label }), el("b", { text: value }),
    detail ? el("span", { class: kind || null, text: detail }) : null);
}

// The top strip. The other markets quoted on Binance arrive with the prices; the rest, every minute.
function renderContextStrip() {
  const node = $("context");
  const cross = (snapshot && snapshot.radar && snapshot.radar.cross) || [];
  const c = context || {};
  const items = [];
  if (c.global) {
    items.push(ticker("Crypto market cap", `${big(c.global.market_cap_usd)} $`, `${signed(c.global.market_cap_change_24h_pct)} % 24 h`, tone(c.global.market_cap_change_24h_pct)));
    items.push(ticker("BTC dominance", `${number(c.global.btc_dominance_pct, 1)} %`, `ETH ${number(c.global.eth_dominance_pct, 1)} %`));
  }
  if (c.fear_greed) {
    const value = c.fear_greed.value;
    const meter = el("div", { class: "meter", role: "img", "aria-label": `Fear and greed: ${value} of 100` },
      el("i", { style: `left:${Math.max(0, Math.min(100, value))}%` }));
    items.push(el("div", { class: "ticker" }, el("small", { text: "Fear and greed" }),
      el("b", { text: `${value} · ${c.fear_greed.label}` }), meter));
  }
  for (const asset of cross) {
    items.push(ticker(asset.name, price(asset.price), isNumber(asset.change_24h_pct) ? `${signed(asset.change_24h_pct)} % 24 h` : null, tone(asset.change_24h_pct)));
  }
  for (const quote of Object.values(c.macro || {})) {
    if (!quote || !isNumber(quote.price)) continue;
    items.push(ticker(quote.name, number(quote.price, 2), isNumber(quote.change_pct) ? `${signed(quote.change_pct)} % today` : null, tone(quote.change_pct)));
  }
  replace(node, items);
}

function newsItems() {
  const items = (context && context.news && context.news.items) || [];
  const coin = selected ? coinOf(selected) : null;
  if (newsFilter === "basket") return items.filter((item) => item.coins.length);
  if (newsFilter === "coin") return items.filter((item) => item.coins.includes(coin));
  return items;
}

function renderNews() {
  const coin = selected ? coinOf(selected) : "Coin";
  $("news-coin").textContent = coin;
  const all = (context && context.news && context.news.items) || [];
  const items = newsItems();
  $("news-hint").textContent = context && context.news
    ? `${all.length} headlines from ${[...new Set(all.map((item) => item.source))].join(", ")}` : "";
  replace($("news"), items.length ? items.map((item) => el("li", {},
    el("div", { class: "news-meta" }, el("span", { text: ago(item.published_at) }), el("span", { text: item.source }),
      item.coins.map((each) => {
        const chip = el("button", { type: "button", class: "chip wait", text: each });
        chip.addEventListener("click", () => select(`${each}USDT`));
        return chip;
      })),
    // Only https addresses are linked; the text is written as text, never as markup.
    el("a", { href: item.url.startsWith("https://") ? item.url : null, target: "_blank", rel: "noopener noreferrer", text: item.title }),
  )) : el("li", { class: "empty", text: context === null ? "Reading headlines…"
    : !context.news ? "Could not read the headlines (or external data is turned off)."
    : "No recent headline with that filter." }));
}

// What has a set time in the bot's life: its daily cycle and the day's three funding charges.
function botEvents(now) {
  const events = [];
  const next = snapshot && snapshot.meta && snapshot.meta.next_cycle;
  if (next) events.push({ at: next.at, title: next.pending ? "Bot daily cycle (pending)" : "Bot daily cycle", country: "BOT", impact: "Bot" });
  const day = new Date(now);
  for (let offset = 0; offset < 2 && events.length < 3; offset += 1) {
    for (const hour of FUNDING_HOURS) {
      const at = Date.UTC(day.getUTCFullYear(), day.getUTCMonth(), day.getUTCDate() + offset, hour);
      if (at > now && events.filter((event) => event.country === "FUND").length < 2) {
        events.push({ at, title: "Contract funding", country: "FUND", impact: "Bot" });
      }
    }
  }
  return events;
}

function renderCalendar() {
  const now = Date.now();
  const economic = ((context && context.calendar && context.calendar.events) || []).filter((event) => event.at > now - 2 * 3600000);
  const events = [...botEvents(now), ...economic].sort((a, b) => a.at - b.at).slice(0, 24);
  const rows = [];
  let lastDay = null;
  for (const event of events) {
    const day = DAY.format(new Date(event.at));
    if (day !== lastDay) rows.push(el("li", { class: "day", text: day }));
    lastDay = day;
    rows.push(el("li", { class: event.at < now ? "past" : null },
      el("time", { title: `${TIME_UTC.format(new Date(event.at))} UTC`, text: TIME.format(new Date(event.at)) }),
      el("span", { class: `chip ${event.impact === "High" ? "bad" : event.impact === "Bot" ? "pos" : "cand"}`, text: event.country }),
      el("span", { class: "what", text: event.title }),
      event.forecast || event.previous ? el("span", { class: "figures", text: `fcst. ${event.forecast || "—"} · prev. ${event.previous || "—"}` }) : null,
    ));
  }
  replace($("calendar"), rows.length ? rows : el("li", { class: "empty", text: "No events." }),
    rows.length ? el("li", { class: "note", text: "Local time (hover to see it in UTC). In red, the high-impact releases." }) : null);
}

function renderContext() {
  renderContextStrip();
  renderNews();
  renderCalendar();
}

async function refreshContext() {
  try {
    const response = await fetch("/api/context", { cache: "no-store" });
    if (response.ok) context = await response.json();
    else if (context === null) context = {};
  } catch (error) {
    if (context === null) context = {};
  }
  renderContext();
}

for (const button of $("news-filter").children) {
  button.addEventListener("click", () => {
    newsFilter = button.dataset.news;
    for (const each of $("news-filter").children) each.setAttribute("aria-pressed", String(each === button));
    renderNews();
  });
}
hooks.full.push(renderContext);
hooks.live.push(renderContextStrip);
hooks.select.push(renderNews);
refreshContext();
setInterval(refreshContext, CONTEXT_MS);
