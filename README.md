# Drone roof inspection for claims

Point a vision-language model at drone footage of a storm-damaged roof and get back what a claims
adjuster needs before a site visit:

- **Boxed findings per frame:** missing shingles, exposed decking, temporary tarps, debris, fallen
  limbs and structural damage, each with a severity.
- **A pre-inspection damage summary:** overall severity, the estimated damaged share of the roof, a
  recommendation (repair, emergency tarp, partial or full replacement) with its reasoning, damage
  items linked to evidence frames, and what to verify on site.

Everything streams to the page as it happens, and each finding links back to the moment in the video.

```
drone video ─▶ ffmpeg keyframes ─▶ Qwen3-VL-8B (per frame, JSON + boxes) ─▶ Qwen3-VL-8B (summary) ─▶ report
```

## Files

| File | Role |
| --- | --- |
| `deploy.py` | Deploys two `chalkcompute.ScalingGroup`s: vLLM serving Qwen3-VL-8B on one L40S, and the inspection app. |
| `app.py` | Samples keyframes, asks the model for findings frame by frame, then for the summary, and streams NDJSON. |
| `index.html` | Sample picker and upload, video player, boxed keyframes, and the adjuster report. |

The model runs on stock [vLLM](https://github.com/vllm-project/vllm) through its OpenAI-compatible
API. Each call uses a JSON schema (`response_format: json_schema`), so every answer parses: finding
types and severities come from fixed lists, and boxes are `[x1, y1, x2, y2]` on Qwen-VL's 0–1000 grid.

## Run it

You need [`uv`](https://docs.astral.sh/uv/) and a Chalk environment that can schedule an L40S (or
the GPU choice is set directly in `deploy.py`). Credentials come from your existing `chalk login` in the current
directory.

```bash
./deploy.py
```

Open the printed `web_url`, pick a sample or upload a drone video (MP4 or MOV, up to five minutes),
and press **Inspect roof**. The model server is `authenticated=True`. The app calls it with its own
workload identity (`chalk_identity=True`).

## API

`POST /api/analyze?sample=ian` (or a multipart `video` upload) streams one JSON object per line:

| Event | Fields |
| --- | --- |
| `start` | `duration`, `frames`, `model` |
| `frame` | `i`, `t` (seconds), `image` (JPEG, base64), `scene`, `roofs_visible`, `frame_quality`, `findings[]` with `type`, `severity`, `description`, `bbox` |
| `report` | `report`: `headline`, `overall_severity`, `estimated_damaged_area_pct`, `recommendation`, `damage_items[]`, `verify_on_site[]`, `confidence`, … |
| `done` | `total_s` |

## Footage

Both samples are public-domain U.S. Army Corps of Engineers drone footage from
[DVIDS](https://www.dvidshub.net), shot during Operation Blue Roof:

- `ian`: a single home in Fort Myers, FL, after Hurricane Ian, October 2022
  ([DVIDS 860100](https://www.dvidshub.net/video/860100)), trimmed to the close pass over the roof.
- `ida`: a residential street in LaPlace, LA, after Hurricane Ida, 2021
  ([DVIDS 815901](https://www.dvidshub.net/video/815901)).

## Notes

- This is decision support from footage alone, not a coverage determination. The summary asks the
  adjuster to verify what video cannot show, such as decking under tarps or interior water damage.
- One inspection samples up to 14 keyframes and runs six frames at a time. vLLM batches them on the
  GPU, and at most three inspections run at once.

## License

The code is Apache-2.0. Qwen3-VL-8B-Instruct is Apache-2.0 (Alibaba Cloud). The sample footage is in
the public domain.

## Local authentication

The SDK reads credentials, API host, and the selected environment from your existing
`chalk login` in `~/.chalk.yml`, including project-scoped and default logins.
For unattended runs, `CHALK_CLIENT_ID` and `CHALK_CLIENT_SECRET` remain supported.
The examples use the current environment selected by your Chalk login.
