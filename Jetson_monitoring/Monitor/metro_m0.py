"""Serial interface to the Arduino Metro M0 (see M0_metadata_BME280.cpp at the
repo root), which in its default 'C' mode reports eight sensor readings on
one line whenever it's read:

    p:<BME280 pressure Pa>,h:<BME280 humidity %>,b:<BME280 temperature C>,P:<Bar3XT pressure mbar>,d:<Bar3XT depth m>,T:<Bar3XT temperature C>,l:<leak 0/1>,t:<DS18B20 temp C>

metadata.py splits these into two categories:
  environmental -- Bar3XT pressure, depth + temperature (P, d, T): the
                    surrounding water column
  device        -- BME280 enclosure pressure, humidity + temperature (p, h, b),
                    leak (l), DS18B20 enclosure temperature (t): housing
                    health, not the environment

Kept as a single persistent serial.Serial connection (opened once, not per
read) since opening a USB-serial port toggles DTR on most Arduino boards,
which resets them -- reopening every sample would keep restarting the M0.

Never raises out to the caller: any missing device/read error returns None,
so metadata.py's "must never crash regardless of other hardware" guarantee
holds for this sensor same as everything else it reads.
"""

import serial

import config

_connection: serial.Serial | None = None
_warned_once = False

REQUIRED_FIELDS = {"p", "h", "b", "P", "d", "T", "l", "t"}


def _get_connection() -> serial.Serial | None:
    global _connection, _warned_once
    if _connection is not None and _connection.is_open:
        return _connection
    try:
        _connection = serial.Serial(
            config.METRO_M0_SERIAL_PORT, config.BAUD_RATE, timeout=config.SERIAL_TIMEOUT_S,
        )
        _warned_once = False
        return _connection
    except serial.SerialException as exc:
        if not _warned_once:
            print(f"[metro_m0] Could not open {config.METRO_M0_SERIAL_PORT}: {exc}")
            _warned_once = True
        _connection = None
        return None


def read_sample() -> dict | None:
    """Reads one line from the M0 and parses it into a dict of raw field
    values, e.g. {"p": 101325.0, "h": 55.2, "b": 22.4, "P": 1029.3, "d": 1.234,
    "T": 18.1, "l": 0.0, "t": 18.5}. Returns None if the M0 isn't connected,
    the read timed out with nothing buffered, or the line didn't parse (a
    partial line right after opening the connection is normal, not an error)."""
    conn = _get_connection()
    if conn is None:
        return None

    try:
        raw = conn.readline().decode("ascii", errors="ignore").strip()
    except serial.SerialException as exc:
        print(f"[metro_m0] Serial read failed, will reconnect next sample: {exc}")
        global _connection
        try:
            conn.close()
        except serial.SerialException:
            pass
        _connection = None
        return None

    if not raw:
        return None

    fields = {}
    for part in raw.split(","):
        key, sep, value = part.partition(":")
        if not sep:
            continue
        try:
            fields[key.strip()] = float(value.strip())
        except ValueError:
            continue

    if not REQUIRED_FIELDS.issubset(fields):
        return None

    return fields
