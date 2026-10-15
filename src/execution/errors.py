"""Errors raised by the exchange layer."""

from __future__ import annotations


class BrokerError(RuntimeError):
    """The exchange or the network failed. It is transient: the cycle retries."""


class OrderRejectedError(BrokerError):
    """The order breaks the contract rules or there is not enough balance."""


class BrokerUnavailableError(RuntimeError):
    """The exchange cannot be used as it is configured. It is not retried.

    Unlike ``BrokerError``, waiting does not fix it: credentials are missing or
    the account is in a mode the bot does not handle.
    """
