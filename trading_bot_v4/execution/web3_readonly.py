"""Arbitrum Web3 connectivity checks that never access a private key."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from trading_bot_v4.config_v4 import V4Config as Config


@dataclass(frozen=True)
class Web3Readiness:
    configured: bool
    connected: bool
    chain_id: int | None
    block_number: int | None
    wallet_address: str | None
    eth_balance: float | None
    error: str | None
    rpc_source: str | None = None
    live_signing_enabled: bool = False
    gmx_positions_status: str = "not configured"
    gmx_pending_orders_status: str = "not configured"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_web3_readiness(rpc_url: str | None = None, wallet_address: str | None = None) -> Web3Readiness:
    """Read public chain/wallet state. Signing and transaction APIs are absent."""
    primary = rpc_url or Config.ARBITRUM_RPC_URL
    candidates = [("primary", primary)]
    if not rpc_url and Config.ARBITRUM_BACKUP_RPC_URL:
        candidates.append(("backup", Config.ARBITRUM_BACKUP_RPC_URL))
    address = wallet_address or Config.WEB3_WALLET_ADDRESS
    if not primary:
        return Web3Readiness(False, False, None, None, address or None, None, "ARBITRUM_RPC_URL is not configured")
    try:
        from web3 import Web3
    except ImportError:
        return Web3Readiness(True, False, None, None, address or None, None, "web3 is not installed")
    errors = []
    for source, url in candidates:
        try:
            web3 = Web3(Web3.HTTPProvider(url, request_kwargs={"timeout": 10}))
            if not web3.is_connected():
                raise ConnectionError("RPC did not respond")
            chain_id = int(web3.eth.chain_id)
            if chain_id != Config.ARBITRUM_CHAIN_ID:
                raise ValueError(f"unexpected chain ID {chain_id}; expected {Config.ARBITRUM_CHAIN_ID}")
            checksum = Web3.to_checksum_address(address) if address else None
            balance = float(web3.from_wei(web3.eth.get_balance(checksum), "ether")) if checksum else None
            return Web3Readiness(True, True, chain_id, int(web3.eth.block_number), checksum, balance, None, source)
        except Exception as exc:
            errors.append(f"{source}: {exc}")
    return Web3Readiness(True, False, None, None, address or None, None, "; ".join(errors))
