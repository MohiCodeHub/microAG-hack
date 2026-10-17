# microAG-hack

A live voice agent for egocentric work video. It captions the worker's camera feed in 5 s chunks, compares the session against reference timelines of the same task, and asks a short spoken question when something looks off. At the end it produces a segmented report with a verdict.

## Setup

```
uv sync
cp .env.example .env   # then fill in OPENAI_API_KEY and ELEVENLABS_API_KEY
```

Keys are read only from `.env` (gitignored) and are never sent to the browser.

## Add a task

Create `tasks/<task_id>/task.json`:

```json
{"id": "make_tea", "name": "Make tea", "steps": [{"id": "boil_water", "label": "boil water"}]}
```

Put example recordings of the task in `tasks/<task_id>/refs/*.mp4`.

## Build reference timelines

```
uv run python scripts/build_refs.py <task_id> [--force]
```

Captions each reference MP4 and writes `<name>.timeline.json` beside it. Existing timelines are skipped unless `--force` is given. The dashboard task list shows how many references each task has.

## Run

```
scripts/run.sh
```

It makes a self-signed certificate for the laptop's LAN IP (again whenever that IP changes, e.g. on a new Wi-Fi), then prints two URLs:

- Phone page: `https://<laptop-ip>:8443/phone`. Open it on the phone (same Wi-Fi) and accept the self-signed certificate warning, otherwise camera and microphone access is blocked.
- Dashboard: `https://localhost:8443/` on the laptop.

The IP is taken from the first of `en0`, `en1`, … that has one. To pick it yourself: `LAN_IP=192.168.1.23 scripts/run.sh`.

The server prints a warning at startup for each missing API key.

If the phone will not open the page, or the page loads but cannot connect (some browsers refuse a WebSocket to a self-signed certificate), use a tunnel with a real certificate instead, and open the `https://…` URL it prints, adding `/phone`:

```
cloudflared tunnel --url https://localhost:8443 --no-tls-verify
# or: ngrok http https://localhost:8443
```

## Demo notes

The duplicate-footage check remembers frame hashes from every session in `data/hashes.json`. Before the real demo, clear the rehearsals so they cannot be matched:

```
scripts/run.sh --fresh       # or scripts/reset_data.sh, then scripts/run.sh
```

## File mode

On the dashboard, click "Play a video file", pick an MP4 and a task. The video is reviewed as if it were a live feed, with no phone needed.

## Switching the monitor model

The monitor defaults to `gpt-5.6-luna`. To try `gpt-6-astra`, set `MONITOR_MODEL=gpt-6-astra` in `.env` and restart. The captioner model is unchanged.
