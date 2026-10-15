"""Bot configuration, read from environment variables.

Every variable starts with ``BOT_`` and uses ``__`` to enter a group, for
example ``BOT_FUTURES__EXECUTION_MODE=TESTNET`` or ``BOT_TELEGRAM__CHAT_ID=...``.
Credentials are stored as ``SecretStr`` so they never show up in logs, in a
``repr`` or in a traceback.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _blank_as_none(value: object) -> object:
    """An empty variable in the .env file (``BOT_X=``) is the same as leaving it out."""
    if isinstance(value, str) and not value.strip():
        return None
    return value


# The twelve coins the daily breakout rule was measured on (docs/research.md).
MEASURED_FUTURES_BASKET = (
    "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT,AVAXUSDT,DOTUSDT,LINKUSDT,LTCUSDT,DOGEUSDT,TRXUSDT"
)
# Default basket of the daily cycle: the twelve measured coins plus 37 more
# contracts, added to stress the data handling and the risk limits in DRY_RUN.
# There is no historical measurement behind those 37. Names are the contract
# names on Binance USDT-M Futures (1000PEPEUSDT, not PEPEUSDT; RENDER, A and S
# are the successors of RNDR, EOS and FTM).
DEFAULT_FUTURES_BASKET = (
    MEASURED_FUTURES_BASKET
    + ",NEARUSDT,ATOMUSDT,UNIUSDT,BCHUSDT,ETCUSDT,FILUSDT,ICPUSDT,VETUSDT,SANDUSDT,MANAUSDT,THETAUSDT,AXSUSDT"
    ",AAVEUSDT,AUSDT,INJUSDT,RENDERUSDT,OPUSDT,ARBUSDT,APTUSDT,SUIUSDT,FETUSDT,TIAUSDT,SEIUSDT,GALAUSDT,STXUSDT"
    ",KASUSDT,LDOUSDT,IMXUSDT,1000PEPEUSDT,1000SHIBUSDT,WIFUSDT,1000FLOKIUSDT,1000BONKUSDT,SNXUSDT,GRTUSDT"
    ",RUNEUSDT,SUSDT"
)


class FuturesSettings(BaseModel):
    """Daily basket cycle on Binance USDT-M Futures (``src/engine/daily_runner.py``)."""

    model_config = ConfigDict(frozen=True)

    # DRY_RUN: real prices, paper account, no order leaves the machine.
    # TESTNET: real orders against the futures testnet (play money).
    # LIVE: real money. It also needs live_trading_acknowledged and
    # BOT_DRY_RUN=false; if either is missing the cycle runs in DRY_RUN.
    execution_mode: Literal["DRY_RUN", "TESTNET", "LIVE"] = "DRY_RUN"
    live_trading_acknowledged: bool = False
    # Futures API credentials. TESTNET keys and LIVE keys are different.
    api_key: SecretStr | None = None
    api_secret: SecretStr | None = None
    # Basket symbols, comma separated.
    symbols: str = DEFAULT_FUTURES_BASKET
    # Closed daily candles requested per symbol. The 100-period ATR is computed
    # over all of them: 500 is the number the strategy was measured with.
    candles: int = Field(default=500, ge=101, le=1499)
    # Leverage set on the exchange for each symbol before opening. With 1 the
    # margin of a position is its full value and nothing is borrowed.
    leverage: int = Field(default=1, ge=1, le=3)
    # Price that triggers the exchange stop: the contract price (the one in the
    # candles the strategy was measured on) or the mark price.
    stop_working_type: Literal["CONTRACT_PRICE", "MARK_PRICE"] = "CONTRACT_PRICE"
    # Fee per side charged by the paper account (0.05 %, the futures taker fee).
    paper_fee_rate: float = Field(default=0.0005, ge=0.0, lt=0.1)
    # Seconds after 00:00 UTC at which the cycle starts: it gives Binance time
    # to close the daily candle and settle funding.
    run_delay_seconds: float = Field(default=90.0, ge=0.0, lt=86_400.0)
    # Past this delay the cycle no longer opens positions (the price may be far
    # from the close that gave the signal). Exits always run.
    max_entry_delay_minutes: float = Field(default=360.0, gt=0.0)
    # Port of the dashboard (python -m src.ui.dashboard).
    dashboard_port: int = Field(default=8770, ge=1, le=65535)
    # The dashboard also shows data from outside Binance (market cap, fear and
    # greed, dollar index, headlines, calendar) and lets the browser listen to
    # the live order book. Set to false to keep it to what the server itself
    # reads from Binance. None of this takes part in the bot's decisions.
    panel_external_data: bool = True
    # Risk of the rule and of the portfolio. The defaults are the values the
    # daily breakout was measured with. Changing them does not change the
    # signal: it scales the size of each position and how many fit at once.
    # Percentage of the balance lost if the initial stop of a position is hit.
    risk_pct: float = Field(default=1.0, gt=0.0, le=5.0)
    # Largest value of one position, as a multiple of the balance.
    max_position_fraction: float = Field(default=0.1, gt=0.0, le=1.0)
    # Largest open risk of the whole portfolio, in % of the balance.
    max_heat_pct: float = Field(default=4.0, gt=0.0, le=20.0)

    _blank = field_validator("api_key", "api_secret", mode="before")(_blank_as_none)

    @field_validator("symbols")
    @classmethod
    def _normalise_symbols(cls, value: str) -> str:
        symbols = [symbol.strip().upper() for symbol in value.split(",") if symbol.strip()]
        if not symbols:
            raise ValueError("the futures basket cannot be empty")
        for symbol in symbols:
            if not symbol.isalnum() or not symbol.endswith("USDT") or symbol == "USDT":
                raise ValueError(f"the futures basket only takes USDT pairs, got {symbol!r}")
        if len(set(symbols)) != len(symbols):
            raise ValueError("the futures basket has repeated symbols")
        return ",".join(symbols)

    @property
    def symbol_list(self) -> list[str]:
        return self.symbols.split(",")

    @property
    def has_credentials(self) -> bool:
        return self.api_key is not None and self.api_secret is not None


class TelegramSettings(BaseModel):
    """Telegram alerts of the daily cycle (``src/engine/notifier.py``).

    Without a token or a chat nothing is sent.
    """

    model_config = ConfigDict(frozen=True)

    # Bot token, from @BotFather.
    bot_token: SecretStr | None = None
    # Chat that receives the alerts; `python -m src.engine.notifier --chats` finds it.
    chat_id: str | None = None
    # false: only cycles with buys, sells, stops or failures send an alert.
    # true: quiet days send one line too, which doubles as a sign of life.
    quiet_days: bool = True

    _blank = field_validator("bot_token", "chat_id", mode="before")(_blank_as_none)

    @field_validator("chat_id", mode="before")
    @classmethod
    def _as_text(cls, value: object) -> object:
        # A chat id is a number (negative for groups) but it is used as text.
        return str(value) if isinstance(value, int) and not isinstance(value, bool) else value


class DashboardSettings(BaseModel):
    """Where the read-only dashboard listens."""

    model_config = ConfigDict(frozen=True)

    host: str = "127.0.0.1"


class Settings(BaseSettings):
    """Root of the configuration. Immutable once loaded."""

    model_config = SettingsConfigDict(
        env_prefix="BOT_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # Master switch. While it is true nothing reaches a real account, whatever
    # BOT_FUTURES__EXECUTION_MODE says.
    dry_run: bool = True
    # Starting balance of the paper account. It only counts the first time:
    # after that the saved paper account is used.
    paper_capital: float = Field(default=10_000.0, gt=0.0)
    # Directory for persistent state (cycles run, paper account, reports).
    state_dir: Path = Path("data")
    log_level: Literal["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_json: bool = False
    # If set, the log goes to this file (rotated by size) instead of the console.
    log_file: Path | None = None

    futures: FuturesSettings = FuturesSettings()
    telegram: TelegramSettings = TelegramSettings()
    dashboard: DashboardSettings = DashboardSettings()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """The configuration of this process (loaded once)."""
    return Settings()
