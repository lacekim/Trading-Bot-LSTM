"""Training wrapper around the original train_model.py implementation."""

from __future__ import annotations

from train_model import train_model as legacy_train_model


def train_v4_model(timeframe=None, lookback=None, send_telegram=True):
    return legacy_train_model(timeframe=timeframe, lookback=lookback, send_telegram=send_telegram)
