# Live testing protocol

Scores on the WLASL test split are always higher than what a webcam gives. This protocol measures
the gap the same way every time so the number in the README is comparable and honest.

## Setup

```bash
uv run signlang demo --source 0
```

Keep the default gate settings (`--window 64 --stride 4 --threshold 0.6 --stable 3`) for the
official run. Change one setting at a time afterwards and note it in the log.

## Procedure

1. Pick **10 signs** from the 100-sign vocabulary (`models/gru.pt` stores the list; `signlang info`
   prints it). Take a mix of easy and visually similar pairs, e.g. `book`/`paper`.
2. Show each tester a reference video of each sign first (WLASL clips or an ASL dictionary).
   Testers who do not know ASL are fine; say so in the log, it changes what the number means.
3. Each tester signs each of the 10 signs **5 times**, in random order, with the hand resting
   between attempts.
4. An attempt is **correct** if the intended word is the one shown within 3 seconds of the sign and
   no other word is shown in that time. **Wrong** if another word is shown. **No answer** otherwise.
5. Repeat with a different tester, lighting or background. Record the conditions.

## Log template

Fill in from your own sessions. Do not copy numbers from elsewhere.

| Tester | Knows ASL | Lighting | Background | Attempts | Correct | Wrong | No answer |
| ------ | --------- | -------- | ---------- | -------- | ------- | ----- | --------- |
|        |           |          |            |          |         |       |           |

## Reporting

- Live accuracy = correct / attempts, with the number of attempts and testers next to it.
- Frames per second from the line `signlang demo` prints on exit, measured on the same machine.
- The gap to the test-split top-1 and what plausibly causes it (unfamiliar signers, camera angle,
  lower resolution, signing speed, missed hand detections).
