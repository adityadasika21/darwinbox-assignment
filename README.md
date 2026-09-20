# Darwinbox Assignment

**Live:** <https://darwinbox-assignment-app.web.app>  ·  API: <https://darwinbox-assignment-api.onrender.com/api/health>

Two deployments, because the frontend is static and the API is not: the page is on
Firebase Hosting and the API is a container on Render's free plan. Nothing depends on a
laptop being awake, and nothing in the hosted path can be billed — see
[Hosting](#hosting), where "free" is a stronger claim than it sounds. The model is the
only thing that differs between there and here; everything runs locally with `make dev`.

> **The hosted demo runs on Gemini's free tier, which has a daily request cap.** When it
> is spent the API says so immediately rather than retrying into a timeout, and the cap
> resets at midnight Pacific. The first request after fifteen idle minutes also pays
> about a minute while Render wakes the instance. Neither limit applies locally: with
> Ollama there is no quota and no cold start, which is the configuration the numbers in
> §5 were measured on.

Upload several messy CSV/Excel files in one session and ask analytical questions in plain
English. Darwinbox Assignment works out how the files relate to each other, answers with a number, table
or chart, and shows the SQL, the rows behind it and every step it took to get there. The
hard part is not the language model — it is the ingestion and relationship-discovery layer
that turns unstructured spreadsheets into a queryable schema, and everything in the design
is arranged so a human can inspect and correct that layer rather than trust it.

---

## 1. Setup

```bash
make setup     # venv, python deps, npm install, ollama pull qwen2.5-coder:7b-instruct-q4_K_M
make fixtures  # generate the corrupted eval fixtures from the clean seed
make dev       # uvicorn on :8000 and vite on :5173
```

Then open <http://localhost:5173>.

Without the frontend:

```bash
python -m darwinbox.cli eval/fixtures/orders_renamed.csv eval/fixtures/customers.csv --schema
python -m darwinbox.cli eval/fixtures/*.csv -q "total order amount by customer region" --sql
```

**Hardware note.** This targets a single RTX 5060 with 8 GB of VRAM, which is why the model
is a 7 B coder at `q4_K_M` and why the prompt budget is ~3000 tokens. That constraint is not
incidental — it forces two-stage table routing and a rendered data dictionary instead of
"put the whole schema in the prompt", which is also what makes the system work on more than
a handful of tables. Warm planning latency is ~4–6 s; the first call after a cold start
pays ~50 s to load the model into VRAM.

`DARWINBOX_LLM=fake` runs the whole API with a canned client, so the test suite needs no GPU.

---

## 2. Architecture

```
  FILES                    │  SCHEMA LAYER (the part that matters)      │  ANSWER LAYER
                           │                                           │
  ┌──────────┐             │  ┌───────────┐   ┌──────────────────┐     │  ┌─────────┐
  │ .xlsx    │─ loader ──► │  │  blocks   │──►│    profiling     │     │  │ BM25    │
  │ .csv     │   grids     │  │  header   │   │ types, signatures│     │  │ router  │
  └──────────┘             │  │  detection│   └────────┬─────────┘     │  └────┬────┘
                           │  └───────────┘            │               │       │
                           │                           ▼               │       ▼
                           │              ┌────────────────────────┐   │  ┌─────────┐
                           │              │  RELATIONSHIP GRAPH    │   │  │ planner │
                           │              │                        │◄──┼──┤  (7B)   │
                           │              │  containment           │   │  └────┬────┘
                           │              │  name similarity       │   │       │
                           │              │  signatures            │   │       ▼
                           │              │  projections           │   │  ┌─────────┐
                           │              │  composite keys        │   │  │sqlglot  │
                           │              │  derived granularity   │   │  │validator│
                           │              │  ── LLM adjudication ──│   │  └────┬────┘
                           │              │     (third signal only)│   │       │
                           │              └───────────┬────────────┘   │       ▼
                           │                          │                │  ┌─────────┐
                           │                          ▼                │  │ DuckDB  │
                           │                   human confirms /        │  │ + chart │
                           │                   rejects / adds          │  └─────────┘
```

The relationship graph is the centrepiece. Everything upstream exists to build it; everything
downstream is constrained by it. The model is only allowed to join on edges the graph
contains, and the middle pane of the UI exists so a human can overrule any of them.

**Module boundaries.** Block detection knows nothing about DuckDB. The planner knows nothing
about HTTP. The validator knows nothing about prompts. The LLM is reached only through an
`LLMClient` protocol, so every module imports and tests without Ollama.

---

## 3. How it handles messy data

The nine corruption cases from the spec, each generated by `eval/corrupt.py` and each
verified by the eval harness.

| Case | What the file looks like | What Darwinbox Assignment does |
|---|---|---|
| `stack_tables` | Two tables in one sheet, separated by blank rows | Builds an occupancy mask, cuts the grid on fully-empty row and column runs, and recurses. Two blocks, each with its own header. |
| `add_title_rows` | Title + subtitle + blank line above the header | The blank line splits the title into its own region, which is dropped for having fewer than 3 populated cells. Header lands on row 3. |
| `add_footer` | A `Total:` row and a `Source:` footnote below the data | Trailing rows whose first cell matches a junk pattern **and** which are not fully populated are dropped. A genuine data row named "Total" keeps its other columns and survives. |
| `rename_key` | `customer_id` → `cust_ref` on one side, `EmployeeCode` on another | Containment on distinct values finds both at 100%. Name tokens come from the **raw** header, so `EmployeeCode` still tokenises to `employee` + `identifier`. |
| `split_sheets` | One table across 3 sheets, plus an unrelated sheet | Tables with an identical column signature are fused into one before discovery. The unrelated sheet is left alone. |
| `drop_fk` | No ID at all — only `(month, region)` links the files | Single-column FKs require a near-unique side, so `region` (4 values, a key on neither side) is rejected and the pair falls through to composite search, which finds `(month, region)`. |
| `change_granularity` | Daily on one side, monthly on the other | `date_trunc` month/week columns are materialised and retried. On success the derived column is added to DuckDB **and** rendered in the data dictionary, so generated SQL can name it. |
| `noisy_keys` | Case, whitespace and punctuation noise in the key | Projections (`lower(trim(·))`, punctuation-stripped, digits-only) are retried, chosen by **row-level coverage** rather than distinct-set containment, and applied to both sides of the generated predicate. |
| `unrelated_pair` | Two datasets with nothing in common | They stay in separate connected components. The UI says "these files are not related to each other" and the prompt carries a `NOT RELATED` block, so a question spanning them is refused. |

### Beyond the nine: a 78-case CSV matrix

A systematic sweep of encodings, dialects, shapes, locales and hostile input. It lives
in `backend/tests/test_csv_matrix.py` and runs with `make test`, because a number in a
README that nothing reproduces is not evidence. Every case asserts what the file
*means* — the round-tripped text, the coerced value, the column count — rather than
that reading it failed to crash.

| Area | Covered |
|---|---|
| **Encoding** | UTF-8, UTF-8+BOM, UTF-16 LE/BE **with and without a BOM**, UTF-32, latin-1, **cp1252**, **Shift-JIS**, **EUC-JP**, **GBK**, **KOI8-R**, **cp1251**, **ISO-8859-5**, **cp1253/ISO-8859-7 (Greek)**, 4-byte emoji, corrupt bytes mid-file |
| **Dialect** | `,` `;` `\t` `|`, quoted delimiters, quoted newlines, escaped `""`, no trailing newline, CRLF, **classic-Mac CR**, whitespace padding |
| **Shape** | empty, header-only, **single column**, single row, ragged, trailing footer row, duplicate headers, blank headers, all-blank column, **side-by-side tables**, **600 columns**, **blank row inside a table**, Unicode/SQL-keyword/numeric column names, 5 KB cell values |
| **Locale** | **`1.234,56` vs `1,234.56`**, **`DD/MM/YYYY` vs `MM/DD/YYYY`**, ISO, timezone-aware, `(50)` negatives, `%`, currency, scientific notation, **leading zeros**, **int64 overflow**, **`NA`/`N/A`/`#N/A`/`null`/`-`** |
| **Hostile** | `=cmd|…` formula injection, SQL injection in values **and in column names**, path traversal in filename |

Every locale convention is decided **per column, never per value** — a uniform mistake
is recoverable, a mixed one is not.

Writing it as tests rather than a script immediately earned its keep by finding two
silent-wrong-answer bugs that the previous ad-hoc run had reported as passing:

- **A 30-digit reference number became `-9223372036854775808`.** `astype("Int64")`
  does not raise on overflow, it wraps, so the existing `OverflowError` guard never
  fired. Such a value is an identifier rather than a quantity — Float64 would round it
  and break the same joins — so it is now kept exactly, as a leading zero already was.
- **A trailing footer row deleted the first row of data.** "A row of consecutive small
  integers under a header is column numbering" returned early, before the test that
  would have recognised `1,2,3` above `4,5,6` as data. The row disappeared with
  nothing to indicate it had existed.

**Short-file encoding detection.** Frequency statistics need enough *variety*, not
enough bytes. With ordinary vocabulary — four different city names — Cyrillic and Greek
resolve correctly at two rows and CJK at three, because candidates are re-scored on
**script coherence**: real words are letters of one script, while mojibake wedges
quotation marks and currency signs inside them (`Москва` misread as `нѕ”Ћ„Ѕ`). Two
further rules: a verdict is accepted only when it yields non-Latin script, so cp1252
wins any Latin-vs-Latin disagreement (the detector otherwise calls a Windows export
cp775, Baltic DOS); and endianness for BOM-less UTF-16 is decided from NUL positions
rather than guessed.

Two limits are asserted as tests rather than left to be discovered:

- **One token repeated, in a tiny file.** KOI8-R, cp1251, ISO-8859-5 and mac_cyrillic
  all turn those bytes into equally coherent Cyrillic and all score 1.0. Supplying a
  shortlist and picking the best-scoring one was tried and reverted: it produces
  confident, plausible, wrong Russian, where declining leaves visible mojibake that a
  human can see and fix.
- **BOM-less UTF-16 holding non-Latin text.** NUL counting is an ASCII-content signal,
  and Russian in UTF-16 carries a quarter the NULs. The generalisation that catches it
  — "one side of each code unit has few distinct values" — also fires on `a,b,c` over
  `1,2,3` over `4,5,6`, an ordinary CSV. Corrupting common input to rescue rare input
  is the wrong trade, and a BOM resolves the case anyway.

### Genres: 28 kinds of spreadsheet

A CSV matrix covers encodings and shapes; it says nothing about the *documents* people
actually upload. `eval/make_genres.py` builds one of each, with the mess its genre
really has, and `eval/score_genres.py` scores ingestion on three objective checks with
no LLM: how many tables were found, whether the columns a human would name appear, and
what fraction of names came out as `col_N`.

| | |
|---|---|
| Architectural room schedule | merged group header over field names |
| Bill of quantities | title block, hierarchical item numbers, subtotals |
| P&L statement | months as columns, category rows |
| Budget tracker | category groups with blank parent rows |
| Shopping list · todo list | small, informal, checkbox column |
| Inventory | SKU table plus a summary block below it |
| Sales pivot · timesheet · gradebook | wide, periods as columns |
| Payroll register | two-row header, totals row |
| Directory · lookup table | plain tables with gaps |
| Category hierarchy | indentation carrying the structure |
| Expense claim | a form header **above** a line-item table |
| Mixed workbook | three genres in one file |
| Bank statement · asset register · multi-currency | running balances, currency symbols mixed into a column |
| Invoice | header block, line items, then a GST/TOTAL block |
| Attendance register | days numbered `1`..`31` as column headings |
| Survey · meeting minutes | sentence-length headers; prose above an action table |
| Price list | two-level tier header |
| Crosstab · project plan | row *and* column totals; week columns with `x` markers |
| Sheet with `#N/A` / `#DIV/0!` | formula errors as values |
| Bill of materials | level numbers carrying the hierarchy |

**28/28.** Getting there fixed two whole classes of failure, both common in the wild:

**Multi-row headers.** A group label spans several columns above the field names —
`Room | Room | Finishes` over `No. | Name | Floor`. It scores *worse* than the row
below it, so the search lands on the lower row and the label is lost. Both spellings
are now handled: Excel's (value in the first cell of a merged range, blanks after) via
gap-filling, and the CSV export's (the label repeated across its columns) via adjacent
duplicates. The guard that keeps this from over-reaching is that an all-text table with
no group label above must not lose its first row to the header.

**Form headers above tables.** An invoice, expense claim or BOQ opens with a few
labelled fields and then gets on with its line items. Split on blank columns those
rows become a scatter of one-row fragments named `employee / ravi_kumar`. They now
collapse into a single `(section, item, value)` table beside the real one, together
with the `Subtotal / GST / TOTAL` block that trails the line items — also labelled
figures rather than another table. A one-row region used to vanish entirely, its only
row eaten as a header leaving no data, which is how an invoice lost its number and
date; those rows are now kept.

The discriminator between a field label and a column group is the **trailing colon**:
`Invoice No:` heads a metadata line, `Pay Matrix` heads columns. Without it the
invoice's header line was swallowed into the line-item table.

**Two more, from the second round of genres.** A header that is *only* digits — days
`1`..`31` across an attendance register — was erased by the rule that strips leading
digits to make a usable identifier, turning every day into `col_N`. And a **blank row
inside a header band** split it in two before the band logic ever ran: real sheets put
a spacer between the group label and the field names, and the upper row was being lost
with the title block.

### Scale

| Case | Result |
|---|---|
| 100k-row fact + 50k-row dimension (6 MB) | ingest+profile+discovery **4.7 s**, 0.34 GB RSS, FK found at 100% |
| **1,000,000-row fact (45 MB) + 50k dimension** | **34.3 s**, 1.14 GB RSS, FK found at 100% |
| 1M × 50k join + aggregate | **0.02 s** (DuckDB) |
| 2,000 × 600 columns | 6.0 s |
| 42 files / 854 columns / 861 table pairs | 11.9 s |
| Row cap, file ceiling, query timeout | 5,000-row cap + `truncated` flag; >100 MB → `413`; 20 s timeout enforced |

Profiling the 1M-row case showed type coercion was 33.7 s of 45.9 s — a Python call
per cell, six million of them. A vectorised fast path for columns needing no cleaning,
plus rejecting non-numeric columns on a 1,000-row sample instead of the whole column,
brought the total to 34.3 s.

Three of these were only fixed because the eval caught them; see §5.

---

## 4. How it avoids making things up

**Statistics propose, the model only adjudicates.** Relationship discovery is entirely
statistical: containment on distinct value sets (not Jaccard — a 50k-row fact table joining
a 50-row dimension is the normal case, and Jaccard scores that near zero), token-Jaccard name
similarity over abbreviation-expanded names, and shared regex signatures. Only candidates
that already passed the statistical filter and landed in the uncertain 0.5–0.75 band are
shown to the model, at most 8 per session, one tiny prompt each. **The model can never
propose an edge from names alone** — it can only confirm or deny one the numbers already
surfaced, and an LLM-assisted edge is never auto-confirmed.

**Five gates stop coincidences becoming joins.** Each was added because something concrete
slipped through:

1. A single-column FK needs a **near-unique side** — or `region` matching `region` at 100%
   becomes a "key", and the composite search that should find `(month, region)` never runs.
2. It must join on an **identifier** — or `payments.paid` matches `orders.amount` at 100%
   because every payment equals its order total.
3. A derived date key must **discriminate** — or three attendance sheets all from March
   truncate to one month each and "join" perfectly to each other and to a holidays sheet.
4. Below a reasonable key cardinality, containment needs **corroboration** from name or
   value format. The `min(|A|,|B|)` denominator that lets a 50-row dimension match a 50k-row
   fact table also makes a two-value column trivially contained in anything — real Treasury
   data scored 1.00 against a column of row numbers on `{2025, 2026}` alone.
5. A candidate in the adjudication band must be **strong on something other than
   containment** before it spends one of the 8 model calls, or noise crowds out the real key.

**And the model cannot veto the statistics.** Adjudication breaks genuine ties; it is not a
gate the evidence has to pass. Asked about `branches.cert` vs `institutions.cert` — the real
FDIC foreign key, identical name, 58% overlap — the 7 B answered *"different tables and
columns"* and rejected it. Pairs whose normalized names are identical and whose values
already overlap now bypass adjudication and are surfaced as *proposed* for a human to judge.

**Nothing generated reaches DuckDB unvalidated.** Every statement is parsed with `sqlglot`,
must be exactly one `SELECT`, must contain no write keyword anywhere, must reference only
real tables and columns, and must survive `EXPLAIN`. Aliases are rewritten to physical table
ids on the parsed tree rather than by string substitution. On failure the model gets exactly
**one** repair attempt with the precise error; if that fails too, the system asks the user to
rephrase rather than return a query it does not trust.

**It asks instead of inventing.** If a question needs two tables with no relationship between
them, or names a metric that is not in the schema, the planner returns a clarification and no
SQL. Two of the twenty eval questions exist purely to check this, and both pass.

**The chart is not the model's choice.** Chart type is computed from the shape of the result;
`chart_intent` is only a tiebreaker. A test asserts that a wrong intent cannot override it.

**Everything is visible.** Every stage emits a trace event that streams to the UI and stays
after the answer: which tables were routed and on what terms, the plan, the validation
result, any repair, the row count, and the chart decision.

---

## 5. Eval results

`make eval` runs all three stages. Ground truth is computed from the **clean** source and
frozen into `expect_rows`; the questions are then asked against the **corrupted** files, so a
pass means the messy-data layer did its job.

**Ingestion** — block counts, header rows and typing on the corrupted fixtures: **5/5 pass**.

**Relationship discovery** — against the known FK set, over all six corruption cases:

| Case | Recall | Precision |
|---|---|---|
| `rename_key` | 1.00 | 1.00 |
| `noisy_keys` | 1.00 | 1.00 |
| `drop_fk` (composite) | 1.00 | 1.00 |
| `change_granularity` (derived) | 1.00 | 1.00 |
| `split_sheets` | 1.00 | 1.00 |
| `unrelated_pair` | 1.00 | 1.00 |
| **overall** | **1.00** | **1.00** |

Target was recall ≥ 0.80 at precision ≥ 0.90.

**Question answering** — 20 questions × 3 runs, `qwen2.5-coder:7b-instruct-q4_K_M`:

| Set | Mean | Spread | Runs | Median latency |
|---|---|---|---|---|
| `questions.yaml` (tuned on) | **100%** | 0% | 100 / 100 / 100 | 4.3 s |
| `holdout.yaml` (**never looked at while tuning**) | **100%** | 0% | 100 / 100 / 100 | 6.0 s |

Failure taxonomy (`bad_sql`, `wrong_table_routed`, `wrong_join`, `refused_wrongly`,
`answered_wrongly`, `timeout`):

| Bucket | Count (of 60 question-runs) |
|---|---|
| `refused_wrongly` | 3 — all the same question, q20 |
| everything else | 0 |

**The last question to fall.** q07 asks for a monthly trend, and the model wrote `EXTRACT(MONTH FROM order_date)`. The eval reported that as a formatting miss — month numbers where dates were expected, identical totals — and it took a second look to see it was a real defect: `EXTRACT` discards the year, so January 2024 and January 2025 collapse into one row. The totals matched only because that fixture covers a single year. The planner is now told to use `date_trunc`, with the reason attached so it is not read as style.

### External validity: real US government data

Every fixture above is synthetic, so it only contains corruptions I thought of.
`python eval/fetch_gov.py` pulls genuinely foreign data — schemas, key names, date
formats and null conventions I did not choose — from the **FDIC BankFind** API
(institutions / branches / financials / failures, keyed by `CERT`) and **US Treasury
Fiscal Data** (debt-to-penny and operating cash balance, keyed by `record_date`).

On first contact the system found the 2 real relationships **and 9 false ones**. Every
false positive traced to a single flaw, which the synthetic corpus could never have
exposed: `containment = |A∩B| / min(|A|,|B|)` — the very denominator that lets a
50-row dimension match a 50k-row fact table — makes a *tiny* set trivially "contained"
in anything. `failures.id` scored 1.00 against a Treasury `record_calendar_year`
holding only `{2025, 2026}`, because both years appear somewhere among 1500 row
numbers. Bank names stripped to digits matched a fiscal-quarter column of `{1,2,3,4}`
the same way. Three fixes followed (see §4); after them, on the same data:

| | |
|---|---|
| `financials.cert = institutions.cert` | found, confirmed (100%) — the real FK |
| `branches.cert = institutions.cert` | found, proposed (58%) — the real FK |
| `treasury_cash ⋈ treasury_debt` on `(record_date, record_fiscal_quarter)` | found, confirmed — composite |
| `branches.name = institutions.name` | found, proposed — real but weaker; the human decides |
| FDIC ↔ Treasury | correctly **unrelated**, separate components |
| false positives | **0** |

Answers were checked against pandas computed directly on the same files: institution
counts by state, assets and deposits by bank class, and a cross-file `CERT` join for
average net income per bank all matched exactly. Questions spanning FDIC and Treasury,
and a question naming a column that does not exist (`interest income`), were both
refused with a question rather than answered.

**The most interesting failure was architectural.** `branches.cert = institutions.cert`
sits at 58% containment — inside the LLM-adjudication band — and the 7 B model *denied
it*, reasoning "different tables and columns", i.e. answering "are these the same
column?" rather than "do these refer to the same thing?". A confused model was
vetoing a real foreign key. Statistics propose and the model adjudicates, but it must
never be able to overrule them: pairs whose normalized names are identical and whose
values already overlap now bypass adjudication entirely and are surfaced as *proposed*
for a human to judge.

### External validity: questions I did not write

Every number above the line is measured on questions I wrote, against fixtures I built.
That is worth having and it is not external validity: 100% there says the system handles
the cases I thought of.

`make external` measures something harder. It generates a QA set over the 42 real
open-data CSVs — traffic crashes, restaurant inspections, library circulation, building
permits — whose schemas, column names, null conventions and distributions were chosen by
the agencies that published them. The profiler picks which column is a category and which
is a measure; the expected answer is produced by executing SQL against the ingested
table. No question and no answer is hand-written.

| External set, 44 questions over 8 real tables | |
|---|---|
| Accuracy | **100%**, spread 0% across 3 runs |
| Median latency | 5.2 s |

What is still mine is the *shape* of the questions — the six templates in
`eval/make_external_questions.py` (row counts, group-bys, averages, superlatives). A
reviewer should read the number as "handles real schemas it has never seen", not as
"handles arbitrary questions".

**The first run scored 80%, and one failure was a serious bug.** `traffic_crashes_1.csv`
— an ordinary 71-column extract — was being read as a *form* rather than a table, its
every column name replaced by a hash of its values. Form detection keyed on density, and
that file carries forty optional columns (`vehicle_defect`, `towed_by`,
`area_00_i`..`area_06_i`) that are almost always empty, putting it under any density
threshold worth setting. Sparsity is not formness; a missing header is. The rule now
requires that no usable column names were found, which is what a form actually lacks —
and the tax workbook still reads as two forms and two tables.

The remaining failures were the harness being unfair rather than the system being wrong:
asked "which permit type has the highest total processing time", it answered with the
type *and* the figure, where the generated ground truth held only the type. Naming the
winner without the number answers less of the question, so the expected answer was
corrected rather than the behaviour.

### A browser smoke test

`make smoke` drives the real page in Chromium: loads a sample, checks all five tables
render, asks a question, waits for the prose answer and the chart, opens the trace to
confirm the SQL is shown, and forces a `SESSION_NOT_FOUND` to prove an expired session
explains itself instead of dead-ending. **15/15 checks pass.**

It exists because the two bugs that made this app look broken to a user — SSE frames
split on `\n\n` when the server sends `\r\n\r\n`, and a spinner that never stopped —
were both invisible to a fully passing backend suite. A browser is the only place they
show up.

### Genres: 28 kinds of spreadsheet

A CSV matrix covers encodings and shapes; it says nothing about the *documents* people
actually upload. `eval/make_genres.py` builds one of each, with the mess its genre
really has, and `eval/score_genres.py` scores ingestion on three objective checks with
no LLM: how many tables were found, whether the columns a human would name appear, and
what fraction of names came out as `col_N`.

| | |
|---|---|
| Architectural room schedule | merged group header over field names |
| Bill of quantities | title block, hierarchical item numbers, subtotals |
| P&L statement | months as columns, category rows |
| Budget tracker | category groups with blank parent rows |
| Shopping list · todo list | small, informal, checkbox column |
| Inventory | SKU table plus a summary block below it |
| Sales pivot · timesheet · gradebook | wide, periods as columns |
| Payroll register | two-row header, totals row |
| Directory · lookup table | plain tables with gaps |
| Category hierarchy | indentation carrying the structure |
| Expense claim | a form header **above** a line-item table |
| Mixed workbook | three genres in one file |
| Bank statement · asset register · multi-currency | running balances, currency symbols mixed into a column |
| Invoice | header block, line items, then a GST/TOTAL block |
| Attendance register | days numbered `1`..`31` as column headings |
| Survey · meeting minutes | sentence-length headers; prose above an action table |
| Price list | two-level tier header |
| Crosstab · project plan | row *and* column totals; week columns with `x` markers |
| Sheet with `#N/A` / `#DIV/0!` | formula errors as values |
| Bill of materials | level numbers carrying the hierarchy |

**28/28.** Getting there fixed two whole classes of failure, both common in the wild:

**Multi-row headers.** A group label spans several columns above the field names —
`Room | Room | Finishes` over `No. | Name | Floor`. It scores *worse* than the row
below it, so the search lands on the lower row and the label is lost. Both spellings
are now handled: Excel's (value in the first cell of a merged range, blanks after) via
gap-filling, and the CSV export's (the label repeated across its columns) via adjacent
duplicates. The guard that keeps this from over-reaching is that an all-text table with
no group label above must not lose its first row to the header.

**Form headers above tables.** An invoice, expense claim or BOQ opens with a few
labelled fields and then gets on with its line items. Split on blank columns those
rows become a scatter of one-row fragments named `employee / ravi_kumar`. They now
collapse into a single `(section, item, value)` table beside the real one, together
with the `Subtotal / GST / TOTAL` block that trails the line items — also labelled
figures rather than another table. A one-row region used to vanish entirely, its only
row eaten as a header leaving no data, which is how an invoice lost its number and
date; those rows are now kept.

The discriminator between a field label and a column group is the **trailing colon**:
`Invoice No:` heads a metadata line, `Pay Matrix` heads columns. Without it the
invoice's header line was swallowed into the line-item table.

**Two more, from the second round of genres.** A header that is *only* digits — days
`1`..`31` across an attendance register — was erased by the rule that strips leading
digits to make a usable identifier, turning every day into `col_N`. And a **blank row
inside a header band** split it in two before the band logic ever ran: real sheets put
a spacer between the group label and the field names, and the upper row was being lost
with the title block.

### Scale: 42 real CSVs across ~20 unrelated domains

The FDIC test used 6 files. This one uses **42 real open-data CSVs** — 854 columns,
30k rows, 861 table pairs — pulled from ~20 domains via the Socrata catalog (restaurant
inspections, traffic crashes, school enrolment, air quality, farmers markets, elections,
salaries, energy, water quality, bus ridership …) plus the 6 banking files. The banking
files have **known** relationships and act as signal; everything else is from an
unrelated domain, so any edge between two of them is a false positive by construction.

It exposed two failures that only appear at scale.

**Performance was quadratic in the wrong place.** Discovery on 10 tables took 45 s, and
76% of it was the composite-key search: the candidate pool was unbounded, so a
116-column table made `combinations()` explode, and tuple rendering repeated the same
pandas work 23,492 times uncached. Capping the pool at the 6 best-scoring columns (the
tail cannot improve a composite) and caching rendered columns took 10 tables from
**45.4 s → 0.6 s**, and the full 42-file corpus now ingests, profiles and discovers in
**11.9 s**.

**Precision collapsed to 3.9%** — 102 relationships found, 4 real. One column caused 63
of them: `bus_ridership.row_id`, an auto-increment surrogate key holding 1…600, which
"contains" every count, score, percentage and code column in that range across the whole
corpus. The fix is the distinction that matters: **integers collide by chance, structured
strings do not.** A join whose overlapping values are purely numeric must also agree on a
content word or share a value format; a join on values like `C-88` stands on containment
alone — which is what keeps the noisy-key case working, since lowercase-and-punctuation
noise destroys the regex signature. Two supporting changes: the token `identifier` is
excluded from name similarity (nearly every key expands to it, so `id` and
`industry_code` were scoring 0.5 on nothing), and every part of a composite must agree on
names, not just the best part.

| 42 files, 861 table pairs | before | after |
|---|---|---|
| Relationships found | 102 | **5** |
| Expected edges recovered | 4 of 4 | **3 of 3** |
| Precision | 3.9% | **80%** ¹ |
| Discovery time | 45 s at 10 tables | **14.2 s at 42 tables** |

Reproduce it with `python eval/fetch_gov.py --wide`. Datasets are discovered through
the Socrata catalog rather than pinned by id, because ids rot and a pinned list decays
silently into "3 of 36 downloaded"; the chosen set is then written to a manifest and
reused, so a second run rebuilds the same corpus rather than whatever the catalog ranks
highest that week (`--refresh` re-discovers). Reproducible and repeatable are different
properties and this needed both. The numbers above are from a freshly fetched corpus,
not inherited from an earlier run.

¹ The single remaining edge is a `(year, month)` composite between two unrelated
library-circulation datasets: a calendar join. It is unwanted here, but suppressing it
in general would break `change_granularity`, where aligning a daily table to a monthly
one is the required behaviour. It is *proposed*, not confirmed, so it surfaces for a
human to reject rather than silently affecting an answer.

Re-running this after rebuilding the corpus also found one more: `farmers_markets`
joined to `library_circulation` on `market_location` = `young_adult_mass_market_paperback_books`.
Those columns share no values at all — the edge came from the digits-only projection,
which reduced the street address "204 Lisha Kill Rd Colonie" to `204` and matched it
against a count of paperback loans. Digits-only exists to match "EMP-0042" to a bare
42, so it now also requires the non-numeric side to be shaped like a code — one short
token — rather than prose that merely contains a number.

**Routing held up.** Asked 8 questions spanning the 42 tables, BM25 routed to the right
table **7 times**; answers for restaurant grades, crashes by weather and average bus
ridership matched pandas exactly (and it picked `riders`, not the `row_id` or
`calendar_year` columns sitting beside it). The prompt-budget drop fired on 2 of the 8,
as designed, without affecting correctness. The one routing miss — "total assets by bank
class", where `bkclass` does not tokenise to `class` — **refused rather than answering
from the wrong table.**

**What the eval caught that tests did not.** The QA stage took accuracy from 75% to 95% by
exposing three real bugs, all of which produced a plausible-looking wrong answer rather than
an error: noisy keys joining on raw values and silently dropping three quarters of the rows;
derived granularity columns being named in the relationships block but never listed in the
schema the model sees; and `split_sheets` leaving one logical table as three, putting a
correct answer out of reach. That is the argument for having an eval at all.

---

## 6. Deliberately out of scope

| Not built | Why |
|---|---|
| Auth, users, multi-tenancy | Nothing here is shared between people; a session is one person's working set. |
| Persistence beyond the session | The interesting state is the *schema graph*, and re-deriving it is cheap and always current. |
| Multi-turn follow-ups | Each question being independent keeps the trace honest — no hidden context influencing an answer. |
| RAG / embeddings | The retrieval problem here is over table *names and columns*, which BM25 handles exactly; an embedding model would also cost VRAM the 7 B needs. |
| Fuzzy entity resolution (trigram/Levenshtein) | Deliberately deferred. The projections cover mechanical noise; genuine fuzzy matching needs blocking to be tractable and a human to be trustworthy. |
| Agent loops, LangChain, LlamaIndex | The pipeline is a fixed sequence with one repair step. A framework would hide exactly the steps the trace panel exists to show. |
| Files > 100 MB / 2M rows | Everything is held in memory as pandas frames; past that, Parquet and a different ingestion path. |
| Write queries | Read-only by construction, enforced in the validator, not by convention. |

---

## 7. What I'd build next

1. **Fuzzy entity resolution with blocking** — the deferred item above. Trigram similarity
   over blocked candidate sets, surfaced as proposed edges with a match rate the user confirms.
2. **A persisted semantic layer** — let a user name a join, save a metric definition
   ("revenue = sum of orders.amount"), and have it survive the session.
3. **Query result caching** — keyed on (question, schema hash); the schema hash invalidates
   correctly when a relationship is confirmed or rejected.
4. **Larger files via Parquet** — stream to Parquet on upload and let DuckDB read it, instead
   of holding pandas frames in memory.
5. **Multi-turn follow-ups** — with the caveat above: only if the carried context is shown
   in the trace as explicitly as everything else.
6. **Column-level lineage** — trace an answer's number back through the join path to the
   source cells in the original spreadsheet.

---

## 8. Layout and commands

```
backend/darwinbox/
  models.py          all pydantic contracts, written before any logic
  ingest/            file -> grids -> table blocks
  profile/           type coercion, semantic typing, DuckDB registry
  relate/            candidates, projections, graph      <- the core
  llm/               client protocol, BM25 router, prompts, planner
  execute/           sqlglot validator, DuckDB runner, chart selection
  api/               FastAPI routes, one error shape
  session.py         the aggregate root
eval/                corrupt.py, make_questions.py, run_eval.py, questions.yaml, holdout.yaml
web/src/             three panes, no routing
```

```bash
make test    # 353 pytest tests + 5 node tests, no GPU needed
make smoke   # 15 checks in a real browser (needs `make dev` running)
make external # real open data, questions and answers derived from the files
make lint    # ruff
make eval    # all three stages
```

<a name="hosting"></a>

### Hosting

The design point is that this runs on your own hardware: a 7B model on 8 GB of
consumer VRAM, no data leaving the machine. That is what the eval numbers above were
measured against, and it stays the default.

A hosted demo cannot depend on a laptop being awake, and no serverless runtime ships
a GPU. So the deployed API talks to Gemini instead. That is a configuration change
rather than a rewrite, because everything above `LLMClient` — the router, the
planner's repair loop, the sqlglot validator, the computed-facts guard that stops the
model deciding which row is highest — never knew which model it was talking to:

```bash
DARWINBOX_LLM=ollama   # default: local, private, needs a GPU
DARWINBOX_LLM=gemini   # hosted: GEMINI_API_KEY, or VERTEX_PROJECT for a service account
DARWINBOX_LLM=fake     # canned replies, so the suite runs with neither
```

The hosted path has to cost nothing, so every piece of it is something that refuses
service when it runs out rather than billing for the overage:

| | | |
|---|---|---|
| API | Render, free plan (`render.yaml`) | suspends at 750 instance-hours/month |
| Frontend | Firebase Hosting, Spark plan | no billing account attached |
| Model | Gemini 2.5 Flash, AI Studio free tier | rate-limited, not billed |

That is a narrower claim than "within the free tier" and the difference is the whole
point. Cloud Run's free tier is the most generous of the three and the container runs
there unchanged — but the project behind it has billing enabled, so exceeding the
tier is charged rather than refused. Vertex is worse again: no free tier for Gemini at
all, billed from the first token. Both work; neither is free by construction.

Deploy is connecting the repo once in Render's dashboard, which reads `render.yaml`,
plus the key set there by hand. Then:

```bash
make deploy-web API_BASE=https://<render-url>   # frontend -> Firebase
```

What free costs: 512 MB, a fraction of a CPU, and a spin-down after fifteen idle
minutes that puts about a minute on the next request. The bundled samples are fine on
that; a million-row workbook is not, and the numbers in §5 were measured locally.
Sessions live in process memory, so the single free instance is also a correctness
requirement — a second one would answer 404 for a session the first created. Moving
sessions out of memory is the change that lifts both limits, and it is not worth
making for a demo.
