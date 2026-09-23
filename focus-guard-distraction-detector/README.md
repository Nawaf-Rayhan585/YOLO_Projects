# Focus Guard — Screen Distraction Detector

A self-monitoring productivity tool for your **own** machine: it periodically
screenshots your own screen, runs the image through YOLO to spot distraction
objects (phone, TV remote, streaming UI) and checks your active window title
against a list of social/entertainment keywords, then throws up a hard-to-miss
full-screen warning and a beep when it catches you slipping.

<img width="406" alt="focus guard" src="https://github.com/user-attachments/assets/b596bd39-1747-41ad-830c-861690b69706" />

Everything runs **locally and offline** — no screenshot or window title ever
leaves your machine.

## How it works

1. Every `--interval` seconds, `mss` grabs a screenshot of the primary
   monitor.
2. The screenshot is run through a YOLO model; a `cell phone`, `remote` or
   `tv` detection is treated as a strong distraction signal.
3. In parallel, a background thread polls the OS for the active window's
   title (Win32 API / AppleScript / `xdotool` depending on platform) and
   flags it if it contains a keyword like `instagram`, `tiktok`, `netflix`,
   `reels`, etc.
4. If either signal clears the confidence threshold, a full-screen Tkinter
   warning overlay appears with a countdown and a "Get back to work" button,
   plus an audio alarm.
5. A live terminal dashboard tracks uptime, scan count and distraction rate.

## Usage

```bash
pip install -r requirements.txt

python focus_guard.py
python focus_guard.py --model yolo11n --interval 5 --mute
```

| Flag | Default | Description |
|---|---|---|
| `--model` | `yolo26x` | Ultralytics model name (no `.pt`); use `yolo11n` for a much faster/lighter download |
| `--interval` | `8` | Seconds between screen scans |
| `--conf` | `0.55` | Confidence needed to trigger the warning overlay |
| `--warning-seconds` | `12` | How long the full-screen warning stays up |
| `--mute` | off | Disable the audio alarm |

Press `Ctrl+C` in the terminal to stop.

## Notes

- First run downloads the chosen model checkpoint via `ultralytics` — the
  default `yolo26x` is a large, high-accuracy model; since scans only run
  every few seconds this isn't a bottleneck, but `yolo11n` is a much faster
  download if you just want to try it out.
- This is a personal self-accountability tool, not surveillance software —
  it only ever looks at the screen and active window of the machine it's
  running on.
