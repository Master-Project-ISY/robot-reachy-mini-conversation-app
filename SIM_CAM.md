# Testing with the simulated robot and laptop webcam

Use `--sim-cam` to feed your laptop's camera into the app when running with the Reachy Mini simulator (no physical robot required).

## Setup

Install the `sim` optional extra (adds `opencv-python`):

```bash
uv pip install -e '.[sim]'
```

## Run

```bash
# default webcam (index 0)
reachy-mini-conversation-app --sim-cam

# specific webcam index
reachy-mini-conversation-app --sim-cam 1
```

Combine with any other flags as usual:

```bash
reachy-mini-conversation-app --sim-cam --ui --debug
```

The app logs `Using laptop webcam at index N as camera source` on startup to confirm the webcam is active. All camera-based features (the `camera` tool, Gemini video sender, local vision) work with the laptop feed.

## Notes

- Head tracking (`--head-tracker`) still works alongside `--sim-cam` but requires its own optional extra (`yolo_vision` or `mediapipe_vision`).
- `--no-camera` takes precedence and disables the webcam too.
- If the index is wrong you get a clear error: `Could not open webcam at index N`.
