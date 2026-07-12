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

Use `python -m open_smart.app --list-devices` to inspect available audio devices, then launch with an explicit two-channel input:

```powershell
python -m open_smart.app --device 45
```

Use `--check-audio` to validate a device before opening the GUI:

```powershell
python -m open_smart.app --device 45 --check-audio
```
