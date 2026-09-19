# Training pipeline (C6-Prep)

Reproducible dataset and label generation, extracted from `chess_model_FINAL.ipynb`.

**This pipeline does not train anything.** It produces an inspectable, deterministic
labelled dataset so the planned A0–A3 experiments can be run fairly. No model
artifact, engine file or evaluation input is read or written.

---

## Why this exists

`docs/C6_TRAINING_PIPELINE_AUDIT.md` found four defects in the notebook's data, all
verified by replaying its own code. This pipeline fixes them and makes the result
auditable.

| # | Notebook behaviour | Here |
|---|---|---|
| 1 | Mate scores read via `['value']` with no `['type']` check — mate-in-1 stored as `1`, **an already-checkmated position stored as `0`** ("equal") | Mates mapped onto the centipawn axis; checkmate is ±2000 |
| 2 | `np.clip(val, -1500, 1500)` commented "clip outlier mate scores" — clipped **0 of 10,000** values | Clip is explicit, configurable, and applied to **cp labels only** |
| 3 | Labels side-to-move relative (`turn_perspective=True` default), so 446 of 10,000 were sign-inverted | Every label **White-positive**, converted explicitly from the recorded side to move |
| 4 | One Stockfish process, hash never cleared → labels depended on iteration order | `ucinewgame` before every position → each label independent |

Also changed: FEN deduplication, malformed SAN recorded rather than swallowed,
Stockfish located via `config.find_stockfish()` rather than a hardcoded path.

**Deliberately preserved:** the population, `df.sample(10000, random_state=42)`, and
the `tokens[:20]` derivation — **at most** 20 plies, not exactly 20. 8.4% of
positions are shorter because the game was. Changing that is not C6-Prep's job.

---

## Usage

```bash
# full artifact (~3 min: ~45s CSV load + SAN replay, ~55s labelling)
python training/build_dataset.py --out training/artifacts/dataset_v1 --verify-source

# quick smoke run
python training/build_dataset.py --out /tmp/smoke --limit 40
```

| Flag | Default | Meaning |
|---|---|---|
| `--source` | `games.csv` | Input dataset |
| `--out` | `training/artifacts/dataset_v1` | Prefix for `.jsonl` + `.manifest.json` |
| `--sample-size` | `10000` | Games sampled |
| `--seed` | `42` | Sampling seed |
| `--cp-clip` | `1500` | Centipawn clip bound |
| `--limit` | none | Label only the first N deduplicated positions |
| `--verify-source` | off | **Fail** if `games.csv` does not match the recorded identity |

Without `--verify-source` a checksum mismatch is a warning, and the manifest records
both the expected and the actual identity — a future dataset may legitimately differ.
Nothing is ever downloaded automatically.

### Input

`games.csv` — **not committed** (gitignored). Recorded identity:

```
sha256      e7aadff104a610afb5403caf81c1461babecb0dc87760ff5eea7dc0e7a4a8129
size        7,672,655 bytes
rows        20,058
```

### Output

| File | Size | Contents |
|---|---:|---|
| `<out>.jsonl` | ~5.0 MB | One JSON object per position |
| `<out>.manifest.json` | ~5 KB | Full provenance, policy and statistics |

Each record carries everything needed to audit its label:

```json
{
  "game_id": "ECQZchLt", "source_row_index": 19390,
  "fen": "r1b2k1r/ppp3pp/2nb1n2/3NppB1/2B5/4P3/PPP2PPP/2KR2NR w - - 6 11",
  "side_to_move": "white", "plies_played": 20, "reached_ply_cap": true,
  "truncated_by_bad_san": false, "bad_san_token": null,
  "is_checkmate": false, "legal_move_count": 39,
  "eval_type": "cp", "raw_stockfish_value": 217,
  "raw_value_perspective": "side_to_move",
  "label": 217, "label_perspective": "white", "was_clipped": false
}
```

The raw side-to-move-relative value is kept alongside the final White-positive
label, so every transformation can be re-derived and checked.

---

## Label convention

**Perspective: `white`.** Positive = good for White, negative = good for Black,
**regardless of whose turn it is**.

Stockfish's native output is side-to-move relative. The conversion is done in
`training/labels.py` from the recorded `side_to_move`, not via the wrapper's
`turn_perspective=False` mode — that mode decides perspective by testing whether the
substring `"w"` appears in the FEN, which works but is a string heuristic. Doing the
arithmetic ourselves keeps the rule visible and unit-testable.

### Mate policy

```
magnitude(d) = 2000 - 10 * min(|d|, 49)

d = 0   -> 2000    mate already on the board (most decisive)
d = 1   -> 1990
d = 5   -> 1950
d >= 49 -> 1510    floor, still strictly above the 1500 cp clip
```

Deterministic, colour-symmetric, monotonic in mate distance (a faster mate always
scores better), and every mate magnitude exceeds the cp clip so **any mate outranks
any non-mate evaluation**.

### The `mate == 0` ambiguity

**VERIFIED** against Stockfish 17.1 / `stockfish` 4.0.8: an already-checkmated
position returns `{"type": "mate", "value": 0}`. Zero carries no sign, so the value
alone cannot say who won.

**Interpretation:** `mate == 0` means **the side to move is checkmated and has
lost**. The sign comes from the side to move, not the value. If `mate == 0` is
reported for a board that is not checkmate, the record is flagged
(`mate_zero_consistent: false`) and counted in the manifest's `anomalies` rather
than silently guessed. In the `dataset_v1` run: **148 such records, 0 anomalies.**

This is the largest single correction in the pipeline — under the notebook, all 148
were labelled `0`.

### Ordering

```
1. mate mapping      mate distance -> signed magnitude on the cp axis
2. perspective       side-to-move relative -> White-positive
3. clipping          CENTIPAWN LABELS ONLY
```

Mates are never clipped: they are produced above the clip bound by construction, so
clipping would collapse the mate scale back into the cp range and undo the policy.

---

## Stockfish configuration

The Phase 3 reference configuration, so labels match the evaluation harness's
conventions and are order-independent:

| Setting | Value |
|---|---|
| Depth | 8 |
| Threads | 1 |
| Hash | 16 MB |
| Clear hash | **before every position** (`ucinewgame`) |
| Perspective requested | native side-to-move relative; converted in `labels.py` |

Version and wrapper version are recorded in the manifest. The **executable path is
deliberately not part of dataset identity** — it is machine-specific.

> This is a **new** labelling pipeline, not an attempt to reproduce the original
> model's labels. The Stockfish version used for those is **unknown** and they were
> order-dependent, so they cannot be recovered.

---

## Deduplication

By **exact FEN**, keeping the first occurrence in the deterministic sample order.
Exact FEN — not piece placement — so side to move, castling rights and en-passant
state remain distinguishing. On `dataset_v1`: 10,000 positions → **9,667 unique**
(333 duplicate rows removed; one FEN occurred 13 times).

---

## Reproducibility guarantees

Deterministic for a fixed `games.csv`, Stockfish binary/version, configuration, seed
and pipeline version. **Verified:** two independent full runs produced byte-identical
output (`sha256 5a689e3f37156a05…`) and identical manifests apart from timestamp and
elapsed time.

The only RNG is the pandas sample seed. No model is trained, so there is no weight
initialisation or dropout to vary — which is precisely the non-reproducibility that
makes the *original* model impossible to recreate.

---

## Should the artifact be committed?

| File | Recommendation |
|---|---|
| `dataset_v1.manifest.json` (~5 KB) | **Commit.** It is the provenance record |
| `dataset_v1.jsonl` (~5.0 MB) | **Probably not.** It is fully regenerable from `games.csv` plus this code, and its checksum is in the manifest. The repo already carries 107 MB of models without LFS |
| `games.csv` (7.3 MB) | **Do not commit** — third-party data, licence unrecorded |

If the `.jsonl` is left out of Git, add `training/artifacts/*.jsonl` to `.gitignore`.
That change is **not** made here because it edits a tracked file outside this phase's
scope — see the C6-Prep report.

---

## Layout

```
training/
├── labels.py           pure label policy: perspective, mate mapping, clipping
├── build_dataset.py    CLI: load -> sample -> positions -> dedup -> label -> write
├── artifacts/          generated output (regenerable)
└── README.md           this file
```

`labels.py` has no Stockfish or I/O dependency, so the whole policy is unit-tested in
milliseconds — mirroring the `evaluation/metrics.py` + `evaluation/evaluate.py` split.

Tests: `tests/unit/test_training_labels.py` (34) and
`tests/unit/test_training_build_dataset.py` (26, one marked `needs_stockfish`).
