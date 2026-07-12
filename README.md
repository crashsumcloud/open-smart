# Open-Smaart

Open-Smaart is a real-time, dual-channel acoustic transfer function analyzer for a single synchronous multi-channel audio interface.

## Hardware Routing

- Input channel 1: reference loopback signal
- Input channel 2: measurement microphone signal
- Sample rate: 48 kHz by default

For transfer-function tuning, use a real two-channel synchronous audio interface. Virtual devices such as broadcast/noise-removal microphones may expose two channels while actually sending duplicated or processed mic audio. If the UI shows similarity near `1.000`, the reference and measurement are probably the same signal.

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

## What the UI Shows

- `Magnitude`: measurement channel divided by reference channel. Use this for room/system tuning when pink noise is playing and coherence is high.
- `Phase`: phase relationship between reference and measurement after delay alignment.
- `Coherence`: confidence that the transfer trace is trustworthy. Values near `1.0` are useful; low values mean noise, bad routing, or insufficient level.
- `SPL(A)`: A-weighted microphone level. It is marked `uncal` until you supply a calibration offset.
- `RT60`: rough reverberation estimate from a decay. Use a clap, starter pistol, balloon pop, or interrupted pink noise. Speech is not a valid RT60 source.

## A-Weighted SPL Calibration

The app can compute A-weighted dBFS immediately, but true dB SPL requires calibration. With an acoustic calibrator:

1. Put a 94 dB SPL calibrator on the measurement mic.
2. Run the app and note the uncalibrated `SPL(A)` value. Example: `-32.0 dBA uncal`.
3. Relaunch with `--spl-offset-db 126` because `94 - (-32) = 126`.

```powershell
python -m open_smart.app --device 45 --spl-offset-db 126
```

## Basic Room-Tuning Workflow

1. Route pink noise to the loudspeaker system.
2. Split the same pink-noise signal back into interface input 1 as the reference.
3. Put the measurement mic into interface input 2.
4. Launch the app with the real interface device index.
5. Adjust delay until the app reports stable delay and high coherence.
6. Tune broad trends first with speaker placement, crossover, polarity, level, and gentle shelf EQ.
7. Use narrow DSP cuts only for repeatable high-coherence peaks. Avoid boosting deep nulls; they are usually placement or cancellation problems.
