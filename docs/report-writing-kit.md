# Report writing kit (not for submission)

The hackathon's AI policy says AI use must be declared and the report's AI content must stay at 30% or below. Above that, the report loses marks equal to the AI percentage: 40% AI costs 40% of its marks.

Open `Solution_Report_P5_T50.docx` and replace every yellow `[WRITE ...]` with your own writing.

## How to keep it genuinely yours

1. **Write from what you did.** Close this kit once you have the facts, then explain the point as you would to a friend in your class.
2. **Don't reword AI text.** Paraphrasing an AI draft doesn't make it yours, and the earlier draft of this report shouldn't be your starting point.
3. **Keep notes on your process.** Drafts and edit history in Google Docs or Word show the writing was yours if anyone asks.
4. **Fill in the declaration truthfully.** Each member lists their own AI use. Run the check last and report the number you get.

Short and plain is fine. Every member will be asked about their section in the viva, so write what you can defend.

## Ask the organisers first

- Which tool measures AI content, and can we run it ourselves before submitting?
- Do tables, figures, captions, code and references count towards the percentage?
- The policy names AI for system design. How should AI-assisted code be treated?

## Who writes what

The owners are a suggestion, based on the draft roles. Swap them freely. The targets total about 1,650 words. The shell has about 200 words of fixed text and 520 in tables, so if tables count as AI content those targets land at exactly 30%, with no margin. Aim for 1,900 or more unless the organisers say tables don't count. This is word arithmetic only: their tool decides the real figure.

| Section | Owner | Words |
| :--- | :--- | :---: |
| Declaration statement | Whole team | 40 |
| 1. Challenge | Rachana | 170 |
| 1. Methodology | Saatwik | 90 |
| 2. Diagram walkthrough | Priyanshu | 50 |
| 2.1 Classical | Disnitha | 170 |
| 2.2 Deep learning | Priyanshu | 190 |
| 2.3 Fusion | Priyanshu | 110 |
| 2.3 Decision logic | Saatwik | 120 |
| 3.1 Metrics note, analysis | Disnitha | 135 |
| 3.1 Reading 94.55% | Priyanshu | 50 |
| 3.2 Figures | Saatwik | 70 |
| 3.3 Ablation | Priyanshu | 90 |
| 4. Failures | Rachana | 210 |
| 4. Limits, ethics | Disnitha | 90 |
| 5. Deployment | Saatwik | 60 |

## Where to find the facts

| You need | Look at |
| :--- | :--- |
| Every metric | `eval_P5_T50.md` |
| Reasoning behind model choices, the failure list | `docs/design.md` sections 5 to 9 |
| Classical stages and their settings | `backend/app/classical.py`, and the pipeline steps in the live app |
| Decision rules | `backend/app/safety.py` |
| Training setup | `training/train_classifier.py`, `finetune_detector.py`, `train_brand_parts.py` |
| Live behaviour | Try the samples and the sweep on the live site |
| Guide sources and their caveats | `docs/hazard-sources.md` |

## Questions to get you started

Answer these out loud first, then write down what you said.

- **Rachana:** If you were a rescuer, which wrong answer would worry you most, and why? Why might a Thar pass the unknown check?
- **Saatwik:** What do you see in the app when a photo is analysed, and why does the warning use a sum? What did you find surprising when testing it?
- **Priyanshu:** What did you try that didn't work? Why did the classical fusion weight come out as 0, and what does the crop actually change for the network?
- **Disnitha:** Why does darkness barely hurt when blur does? If you had one more week, which limitation would you fix first?

## Before submitting

- [ ] No `[WRITE` left in the document, including the commit hash.
- [ ] All four members filled in their declaration row.
- [ ] AI percentage measured, with the real number recorded.
- [ ] Numbers you typed match `eval_P5_T50.md`.
- [ ] Exported to the format the organisers want.
