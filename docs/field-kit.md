# Field kit: the jobs only a person can do

These need a real person, and nothing here is filled in yet. Copy, send, go.

## 1. Talk to a real responder (highest impact)

Who to try, in order: a local fire station (Karnataka State Fire and Emergency Services), a traffic police officer, a 108 ambulance
crew or trainer. Walking in with a phone showing the app works better than a cold message.

**Message to send**

> Namaste. We are engineering students at [college name] building a student prototype for a hackathon. It identifies a crashed car
> from a phone photo and warns if it may be a hybrid or electric car with high-voltage cables. Could we ask you 3 quick questions, in
> person or on a call? It takes 5 minutes. If you are comfortable, we would like to quote you by your role only, not your name.

**The 3 questions**

1. How do you find out what vehicle you are dealing with at a crash today?
2. Have you ever had to deal with an electric or hybrid car? What was different?
3. Would a photo-based check like this help you, and what would stop you trusting it?

**Before you quote anyone**

- [ ] Write down their exact words, then read them back.
- [ ] Ask: "May we quote this in our presentation as '[role], [service], [city]'?" Save the yes (a message is best).
- [ ] If they say it wouldn't help, keep that too. An honest "here is what they told us to change" is a strong pitch.
- [ ] Don't tidy the quote and never invent one. If nobody answers by the code freeze, leave it out of the pitch.

Put the result here:

| Role, service, city | Date | Exact quote | Permission saved? |
|---|---|---|---|
| | | | |

## 2. Photograph 30+ cars in the college parking lot

**Why:** a second, independent test on photos nobody on the team trained or tuned against, plus realistic demo inputs.

| Shot | Count | How |
|---|---|---|
| Daylight, front three-quarter | 8 | The easy baseline |
| Side and rear | 8 | Unusual angles the training set has less of |
| Dusk or low light | 6 | Same evening, no flash |
| Partly hidden | 6 | Behind a pillar, another car, or a bike |
| Close and far | 4 | Fill the frame, then from 20 m |

**Rules**

- Ask the owners or the college first where needed. Avoid faces. Don't share readable plates outside the team.
- Keep the photos in the shared Drive, not in git: they aren't ours to redistribute.
- Write down the true make, model and year in a spreadsheet **before** running anything, so nobody marks their own homework.
- Split the results in two, because most cars here won't be in the 196 classes:
  - **Not in the 196:** the right answer is "unknown". This is the second unknown-car number.
  - **In the 196:** the right answer is the correct model, or an honest "not certain".

Priyanshu runs this on the laptop, into its own folder so the existing `models/unknown_eval.json` isn't overwritten:

```bash
python tools/eval_unknown.py private_photos/parking_not_in_196 --models models_parking_eval
```

`models_parking_eval` must be a copy of `models/`. Report the number with its sample size, and report it even if it's bad.

## 3. Ask the organisers

> Hello, this is Team T50 "Clover" (P5). Two quick questions before Saturday's 14:00 deadline:
> 1. The rubric mentions "every template section" for the design document. Is there a specific template we should follow?
> 2. The rubric text says the other criteria carry 15, but the table shows 25/20/15/15/25. Which weights are correct?
> Thank you.

If a template exists, map the sections of `docs/design.md` onto it.

## 4. Record the backup demo video

Saturday morning, once the live URL works:

```bash
cd frontend && npx playwright install chromium && npm run demo:record -- https://YOUR-LIVE-URL
```

It follows the 3-minute script in `docs/BUILD_PLAN.md` section 2.5 and saves a captioned `.webm` in `frontend/demo-output/`. Watch all of
it before trusting it, then keep a copy on two laptops and a phone. If the venue Wi-Fi fails, this is the demo.

Also take the 3 failure-case screenshots for the design doc from the live app: an "ambiguous" stress test, Thar -> Wrangler and a
heavy blur.

## 5. Before the demo: run the samples through the real models

The two new samples have **not** been run through the trained models, because the model files aren't in the repo. Priyanshu:

```bash
python tools/check_samples.py
```

If a sample doesn't give its expected answer, swap the photo instead of changing the label. Also check the live camera on a
real phone, on the live https URL: browsers only open the camera on https or localhost.

## 6. Decide on offline mode (team decision)

Offline mode is built but **off**, because switching it on makes the model weights publicly downloadable from the live URL (the same
question as the GitHub Release in `docs/BUILD_PLAN.md` section 2.3). To try it, add `"OFFLINE_MODELS": "1"` to the `.env({...})` call in
`modal_app.py` and redeploy, then:

1. On a real phone, on Wi-Fi, tap **Make available offline**.
2. Turn on airplane mode, reopen the app and scan a sample.
3. Time it. If it's too slow or the phone runs out of memory, leave offline mode out of the demo and describe it as work in progress.

It has only been tested with stand-in models. Don't quote accuracy for it until `eval`-style numbers exist for the browser path.
