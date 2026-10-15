// Driving the dashboard like a terminal: command line, keyboard shortcuts and sections in separate windows.
"use strict";

const SECTIONS = {
  POS: "positions", EVO: "curve", ANA: "analytics", MKT: "market", GP: "inspector", NEWS: "news", AUD: "audit",
  TOP: "portfolio", CTX: "context",
};
const SECTION_KEYS = { p: "positions", e: "curve", a: "analytics", m: "market", g: "inspector", n: "news" };
const SECTION_NAMES = {
  positions: "Positions", curve: "Equity curve", analytics: "Analytics", market: "Market", inspector: "Chart",
  news: "News", audit: "Audit", portfolio: "Portfolio", context: "Context",
};
const TIMEFRAME_WORDS = { "5M": "5m", "15M": "15m", "1H": "1h", "4H": "4h", "1D": "1d" };
const section = (name) => document.querySelector(`[data-panel="${name}"]`);

function goTo(name) {
  const node = section(name);
  if (node) node.scrollIntoView({ behavior: "smooth", block: "start" });
}

function setTimeframe(value) {
  ensureChart();
  const button = document.querySelector(`#timeframes [data-tf="${value}"]`);
  if (button) button.click();
}

function popOut(name) {
  const query = new URLSearchParams({ panel: name });
  if (selected) query.set("symbol", selected);
  window.open(`/?${query}`, `basket-${name}`, "popup,width=1280,height=860");
}

function moveSelection(step) {
  const rows = [...document.querySelectorAll("#radar tbody tr[data-symbol]")];
  if (!rows.length) return;
  const index = rows.findIndex((row) => row.dataset.symbol === selected);
  const next = rows[Math.max(0, Math.min(rows.length - 1, index < 0 ? 0 : index + step))];
  select(next.dataset.symbol);
  const shown = document.querySelector(`#radar tbody tr[data-symbol="${next.dataset.symbol}"]`);
  if (shown) shown.scrollIntoView({ block: "nearest" });
}

// A command is separate words: a coin, an interval, a section, or WIN and a section.
function runCommand(text) {
  const words = text.trim().toUpperCase().split(/\s+/).filter(Boolean);
  if (!words.length) return true;
  if (words[0] === "?" || words[0] === "HELP") {
    $("help").showModal();
    return true;
  }
  if (words[0] === "WIN") {
    const target = SECTIONS[words[1]] || (basketSymbol(words[1]) ? "inspector" : null);
    if (!target) return false;
    if (basketSymbol(words[1])) select(basketSymbol(words[1]));
    popOut(target);
    return true;
  }
  let understood = false;
  let destination = null;
  for (const word of words) {
    if (SECTIONS[word]) {
      destination = SECTIONS[word];
    } else if (TIMEFRAME_WORDS[word]) {
      setTimeframe(TIMEFRAME_WORDS[word]);
      destination = destination || "inspector";
    } else if (basketSymbol(word)) {
      select(basketSymbol(word));
      destination = destination || "inspector";
    } else {
      return false;
    }
    understood = true;
  }
  if (destination) goTo(destination);
  return understood;
}

function basketSymbol(word) {
  if (!word || !snapshot || !snapshot.report) return null;
  const symbols = snapshot.report.basket.map((row) => row.symbol);
  return [word, `${word}USDT`, `1000${word}USDT`].find((symbol) => symbols.includes(symbol)) || null;
}

const command = $("command");
command.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    command.value = "";
    command.blur();
  } else if (event.key === "Enter") {
    const done = runCommand(command.value);
    command.classList.toggle("bad", !done);
    if (done) { command.value = ""; command.blur(); }
  } else {
    command.classList.remove("bad");
  }
});

document.addEventListener("keydown", (event) => {
  const typing = event.target instanceof HTMLElement && ["INPUT", "SELECT", "TEXTAREA"].includes(event.target.tagName);
  if (typing || event.ctrlKey || event.metaKey || event.altKey || $("help").open) return;
  const key = event.key;
  if (key === "/") { event.preventDefault(); command.focus(); }
  else if (key === "?") $("help").showModal();
  else if (key === "j") moveSelection(1);
  else if (key === "k") moveSelection(-1);
  else if (key === "t") window.scrollTo({ top: 0, behavior: "smooth" });
  else if (/^[1-5]$/.test(key)) { setTimeframe(TIMEFRAMES[Number(key) - 1][0]); goTo("inspector"); }
  else if (SECTION_KEYS[key]) goTo(SECTION_KEYS[key]);
});

// Each section with a header gets a button to open it on its own in another window.
for (const node of document.querySelectorAll("[data-panel]")) {
  const head = node.querySelector(":scope > .panel-head");
  if (!head) continue;
  const name = node.dataset.panel;
  const button = el("button", { type: "button", class: "popout", title: `Open ${SECTION_NAMES[name] || "the section"} in another window`, "aria-label": `Open ${SECTION_NAMES[name] || "the section"} in another window`, text: "↗" });
  button.addEventListener("click", () => popOut(name));
  head.append(button);
}

// A window opened with ?panel=... shows only that section, full screen.
(() => {
  const query = new URLSearchParams(window.location.search);
  const only = section(query.get("panel"));
  if (!only) return;
  document.body.classList.add("solo");
  only.classList.add("solo-target");
  document.title = `${SECTION_NAMES[only.dataset.panel] || "Dashboard"} · Softronic QuantBot`;
  if (/^[A-Z0-9]{1,24}$/.test(query.get("symbol") || "")) selected = query.get("symbol");
})();
