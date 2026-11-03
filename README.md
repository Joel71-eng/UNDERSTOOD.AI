# Understood (multimodal build)

Learning app for African learners: labelled biology/chemistry diagrams, step-by-step process infographics, molecules and reactions, study guides with practice questions, free image creation and editing,
Mwalimu (the tutor) to chat with, narrated lesson videos and PDF notes. Levels: Ordinary (S1-S4), Advanced (S5-S6), University. English only.

## Two parts
| Where | Files | Job |
|---|---|---|
| Colab (GPU) | `Understood_MM_Server.ipynb` (contains `mm_server.py`) | runs the tutor and the image model, nothing else |
| Your Windows PC | `mm_app.py`, `mm_engine.py`, `mm_core.py`, `mm_chem.py`, `mm_sources.py`, `mm_files.py`, `mm_media.py`, `mm_pdf.py`, `mm_doc.py`, `mm_info.py` | interface, routing, drawing labels, chemistry, video, PDF |

The PC owns the conversation; the Colab service is stateless, so a dropped connection only parks the request (press Resume).

## One-time
1. Run `Understood_MM_Provision.ipynb` (already done: models are on Drive in `understood_mm/models`). Put the speed-up weights in `understood_mm/models/lightning_lora/`.
2. PC: `pip install -r requirements.txt` (Python 3.10+).

## Every session
1. Colab: open `Understood_MM_Server.ipynb`, GPU runtime, run cells 1-6 (cell 1 restarts once, then run it again). Cell 6 opens the **ngrok** tunnel on the permanent address `https://outdoors-graph-tremor.ngrok-free.dev`, so the app never needs a new address. One-time: add a Colab secret `ngrok_token` (key icon, Notebook access on). If anything looks wrong, run cell 8 (server and tunnel logs).
2. PC: `streamlit run mm_app.py`. First run asks for the address (paste the ngrok one once). Later: `http://localhost:8501/?admin=1` to change it (set `UNDERSTOOD_ADMIN_PIN` to protect it).

## Test without a GPU
`set UNDERSTOOD_MOCK=1` then `python mm_server.py` (port 8000), and `set UNDERSTOOD_SERVER=http://localhost:8000`, `streamlit run mm_app.py`.
Fake shapes are drawn instead of real images; everything else (routing, labels, chemistry, PDF, video) is real.
`UNDERSTOOD_MOCK_NAMING` sets how the fake tutor behaves when naming parts, to test the checking: `right` (default), `shifted` (names the wrong part), `blind` (confident but wrong at first look), `unsure` (never sure - nothing should be labelled).

## What is new in this version
- **Process infographics**: see 'How a process is made' below. Processes that do not happen inside one organ (water cycle, mitosis) still use one small picture per step. On screen it is a step-by-step viewer (Previous / Next, step chips showing ready / still to come); you can read steps 1-3 while 4-6 are still being drawn. "See the whole process in one picture" and the video are made from the same steps.
- **PDF and video without any picture**: "Notes" = PDF of the discussion (condensed, or say "full transcript"); "Guide" = study guide with key terms and practice questions + answer key, from built-in notes, Wikipedia, or your own attached file. The video then is slide-style narration of the key points.
- **One chat for everything**: no modes. Type what you want; the sidebar's "More tools" (Picture, Edit, Learn, Export) is only a shortcut form. Under an answer: "Show it as: Diagram | Process | Picture".
- **Background runs**: the page stays usable while a request is being worked on; Resume continues after a lost connection.
- Intent detection: your words first (infographic / step by step / diagram of / study guide ...), then the tutor, and if still unclear it asks once (three buttons).

## How a diagram is made (verified naming)
Built-in notes or Wikipedia -> tutor lists parts -> code keeps only parts literally in the source -> short prompt -> image model draws (no text) -> code finds regions in the picture. Then, instead of asking "which numbered region is which part", the labels are *earned*:
1. **Understand** - the tutor studies the whole picture first (and a free-licence reference picture from Wikimedia Commons when one passes a check; credit with author and licence is shown under the answer).
2. **Name** - each region is shown on its own, outlined, and the tutor names it in its own words. No numbers.
3. **Match** - code maps that name to the verified part list (synonyms, spelling, plurals; ambiguous names are refused).
4. **Check** - the server (`/choose`) reads how sure the tutor is for each part, with the options shuffled between passes. A label is kept only if confidence is at least 0.6; if two regions claim one part, the more confident wins.
5. **Second look** - regions still unnamed and parts still missing are paired one-to-one, and the tutor may say "none of these".
A part nobody can confirm is reported as "not shown", never guessed. Every run is written to `naming_log.jsonl` (in the data folder) so accuracy can be measured on a set of ~10 diagrams.
If the Colab server is an old version without `/choose`, labels are left off with a clear note rather than guessed.

## How a process is made (on one verified diagram)
A process that happens inside an organ, system or apparatus (blood through the heart, a reflex, digestion, how the brain handles a signal) is **not** drawn step by step by the image model. Instead:
1. The organ's diagram is built exactly as above (verified labels), with the parts the process needs placed on it.
2. The steps come from the built-in notes, or the tutor plans them from the notes using only those parts.
3. Code checks each step: supported by the notes, **inside the organ** (a step that names another organ's structures, e.g. the spinal cord in a brain process, is dropped and the tutor is asked once more, told what to leave out), and involving at least one drawn part.
4. Code decides which parts a step highlights and which arrows to draw, from the step's own wording (an arrow needs both ends named in the step; direction follows the words, e.g. 'receives ... from X'). The arrow label says what moves.
5. Each step is the SAME diagram with that step's parts highlighted and the arrows drawn on top, so the organ cannot change between steps, nothing outside it can appear, and no image has to be generated per step (faster, and works with a short Colab session).
If fewer than 3 parts can be placed, the app says so and falls back to one small picture per step.

## Conversation depth
Every typed message is handled by the tutor in the same chat. It recognises what you are doing: a question (answer, how it works, a real example, a common mistake, a question to think about, checked against trusted notes), your reply to its own question (it says what is right, corrects gently, goes one step deeper), 'tell me more' (the mechanism underneath, an example, a limit), or a short follow-up (linked to what came before).

## One conversation, remembered
The tutor decides where each typed request goes (diagram, process, picture, edit, explain, chat, PDF, video); the router sees what is on screen and recent turns. Edits are rewritten using the conversation ("make it brighter" after "make the liver darker"), and the rewrite is shown. Discussion of a diagram or process names the exact labels and says where each part is; a process is discussed before its panels are drawn; chat stays tied to the picture on screen.

## Retention
Whole conversations (messages, pictures, saved state) are deleted after 24 hours without activity. Change with `UNDERSTOOD_RETENTION_HOURS` (0 = keep forever). Runs in progress and pinned conversations (a `.keep` file in the conversation folder) are skipped. It runs at start and every hour. PDFs/videos you already downloaded are yours and are not touched.

## Voice for lessons
Windows built-in voice works after `pip install pyttsx3`. Better: `pip install piper-tts` and put a `.onnx` voice (+ `.json`) in `%USERPROFILE%\understood_data\voices`.

## Not yet verified on real hardware (debug in this order)
1. Server start + tutor warm-up (cell 5).
2. Image generation: staged load (`encode_prompt` on the 2.1 pipeline) and the speed-up weights on the 4-bit model.
3. Region finding, free-text naming and the `/choose` confidence check on real images (compare against `naming_log.jsonl`); the live Wikimedia Commons reference fetch.
4. Image editing (input image into the encoder; falls back to unloading the tutor for that job).
5. Wikipedia/PubChem fetch, voice, MP4 on Windows.
Also unverified: panel style consistency across process steps (look at the real pictures and tell me if they need a shared reference).
Qwen models are research/non-commercial licensed; this is a non-commercial learning project, so that is fine. Check before any commercial use.

## Known limits of this build
- Tested with the mock server only (plumbing, abstention, resume, retention). Real tutor accuracy is for you to confirm on the GPU.
- Text drawn inside generated pictures can still be garbled; the generator prompt is unchanged.
- Replace the files as a set: `mm_app.py`, `mm_engine.py`, `mm_core.py`, `mm_sources.py`, `mm_info.py`, `mm_doc.py`, `mm_server.py` (the Colab copy only changed in its test mode, so re-pasting the notebook is optional).
- The new curated processes (heart, reflex, neuron, brain, digestion) and brain/reflex diagrams are written from standard O-level content; please skim them for your syllabus.

## Diagram bank (bank-first, generator fallback)
For anatomy and biology diagrams the app first looks in the bank (`UNDERSTOOD_BANK_DIR`, default `<data>/bank`, a local copy of the Drive master).
A topic is served only if its id is in `approved.txt`. The bank picture is cropped, padded to a square and resized; the part names and
anchors come from the source drawing, our own label layout draws the text, and the arrow tip is the source's leader-line end snapped inside
the part's region. No generator call is made, for diagrams or for process steps (steps re-render the same picture). The credit line shows
the author and licence. A topic that is not in the bank is drawn by the generator and recorded in `<data>/miss_log.jsonl`.
Build the bank with `bank_build.py` on Colab (see its docstring); review the numbered contact sheet, fix names, then approve the ids.

## Labelling (generator pictures)
The drawing is made in light colours and segmented by code. The tutor then looks at the whole picture ONCE, with a small numbered circle on
every region, and names each numbered structure in its own words (one call, no part list, no per-region questions, no verification loops).
Code only maps a name onto the notes' part name when it is a synonym (so the explanation sentences attach), lets the bigger region keep a
name given twice, and drops names that mean "nothing". `show_naming_log.py` prints `naming_log.jsonl`. Bank pictures skip all of this:
the tutor only explains them. Learning levels are retired for now: one standard depth (university-usable, up to 13 parts).

## Exports, slides and the chat controls
Settings (Output) sit directly above the prompt box. Output = Auto lets the tutor decide and says "I understood this as ...";
any other choice (Diagram, Process, Picture, Explain, Edit) skips the router. The export bar at the end of each answer offers PDF, Slides
(PDF + editable PowerPoint with speaker notes), Video and Notes. "What to include" is expanded: this picture or our discussion, the last N
exchanges (default 4) or the whole chat. "How much detail" (Brief / Standard / Deep) sets how much the notes, slides and video say:
Brief keeps the old short summary; Standard and Deep are built from the answers you were given (their own sections, one heading per question,
a "Key points" list) with no extra summarising call, so nothing is invented. Deep keeps every section and about 130 spoken words per slide.
Slides and video are built from the same scenes; for diagrams and processes the tutor writes a spoken introduction, 2-4 sentences per part or
step, and a recap (a few items per call, cached on disk, so a second export costs nothing). Long narration scrolls in the video caption. Install `python-pptx` for the PowerPoint.
`UNDERSTOOD_PROMPT_STYLE=light` selects the lighter-colour generator prompt; run `ab_style.py` first to compare it with the classic one.

## Stop button
While a request runs, the live view shows **Stop**. It cancels the picture job on the Colab server (the GPU is freed at the next denoising step),
drops any tutor answer still arriving, restores the conversation state and removes the question being answered, so the chat is exactly as
before. A tutor call already being computed finishes first (it cannot be interrupted half-way) and its answer is thrown away.

## Server log and build stamp
`GET /health` reports `build` and `features` (`cancel`). `server.log` (Colab cell 8) has one line per request (`POST /tutor -> 200 in 6.1s`),
per tutor task (`tutor LOCATE: 2 messages, 1 image(s)`) and per picture job (started / done / CANCELLED). The app's admin panel shows its own
build and which bank folder and topics it sees, so you can tell the bank-powered build from an old one.

## Building a bigger bank
`bank_build.py` now looks in three places per topic: file search for SVGs, file search for PNG/JPG (several pages deep), and the Commons
categories that match the topic (plus their sub-categories). Only English is kept: file names ending in another language's code, or with
accents / other scripts, are skipped before download, and labels that are not plain English are rejected (`langdetect` helps when installed).
SVGs are used first. If no SVG gives 5 anchored labels, the PNG/JPG route runs: `bank_raster.py` reads the words with Tesseract (English only),
finds the straight pointer line that starts at each word, and paints words and pointers out. It needs `apt install tesseract-ocr`,
`pip install pytesseract opencv-python-headless`, and is always flagged "RASTER ... check every label position". Per topic the best picture
is saved as `<id>` and the next best as `<id>__alt2`, `<id>__alt3` (`--alts N`); approve the one you like, the first approved one wins.
`--png file.png --id topic` builds one topic from a local picture. 40 starter topics are listed in `TOPICS`.
Pictures with no pointer lines (names written inside the parts) give names without anchors and are flagged for you to place by hand.

### Reviewing the bank
`streamlit run bank_review.py -- --bank <bank folder>` shows each built picture with its numbered anchors and automatic flags (few
labels, labels without a leader line, anchors on blank background, duplicate names). Fix part names there, then Approve or Reject; it
writes `approved.txt` / `rejected.txt` for you. "Approve all that passed the checks" approves every unflagged picture in one click.
