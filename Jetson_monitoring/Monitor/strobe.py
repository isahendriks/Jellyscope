"""Reads the strobe controller (Opto Engineering LTDVE1CH-40F)'s converter/
driver temperatures over Modbus/TCP -- registers 206/207 per its instruction
manual section 14.2.41-42. Same technique livestream.py already uses.

Modbus/TCP (unlike the Metro M0's serial connection) tolerates being queried
from multiple independent processes without careful connection-lifecycle
handling, but a persistent client is still kept across calls to skip the
reconnect handshake on every sample.

Never raises out to the caller -- returns (None, None) on any failure, so
metadata.py's "must never crash regardless of other hardware" guarantee holds
here too.
"""

from pymodbus.client import ModbusTcpClient

import config

_client: ModbusTcpClient | None = None
_warned_once = False


def _to_signed16(value: int) -> int:
    return value - 65536 if value > 32767 else value


def read_temperatures() -> tuple[float | None, float | None]:
    """Returns (converter_temp_c, driver_temp_c), or (None, None) if the
    strobe controller isn't reachable."""
    global _client, _warned_once
    if _client is None:
        _client = ModbusTcpClient(config.STROBE_IP, port=config.STROBE_MODBUS_PORT)
    try:
        if not _client.connected:
            _client.connect()
        rr = _client.read_holding_registers(address=206, count=2, device_id=config.STROBE_MODBUS_UNIT)
        if rr.isError():
            raise IOError(str(rr))
        converter_raw, driver_raw = rr.registers
        _warned_once = False
        return _to_signed16(converter_raw) * 0.1, _to_signed16(driver_raw) * 0.1
    except Exception as exc:
        _client.close()
        if not _warned_once:
            print(f"[strobe] Temperature unavailable: {exc}")
            _warned_once = True
        return None, None
