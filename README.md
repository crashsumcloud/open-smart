# Open-Smaart

Open-Smaart is a real-time, dual-channel acoustic transfer function analyzer for a single synchronous multi-channel audio interface.

## Hardware Routing

- Input channel 1: reference loopback signal
- Input channel 2: measurement microphone signal
- Sample rate: 48 kHz by default

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -e ".[dev]"
open-smart
```

Use `open-smart --list-devices` to inspect available audio devices.
