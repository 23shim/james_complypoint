# ComplyPoint — Document Discovery Tool

Classifies documents from file server exports (TreeSize XLSX or CSV) using only filenames and folder paths. No document content is opened.

---

## First-Time Setup (Windows)

You only need to do this once.

### Step 1 — Install Python

1. Go to https://www.python.org/downloads/
2. Click the big yellow **"Download Python"** button
3. Run the installer
4. **Important**: Tick the box that says **"Add Python to PATH"** at the bottom of the first screen
5. Click **Install Now** and wait for it to finish

### Step 2 — Install the tool's dependencies

1. Open the project folder in File Explorer
2. Click in the address bar at the top, type `cmd`, and press Enter — this opens a command prompt in the project folder
3. Type this and press Enter:

```
pip install -r requirements.txt
```

Wait for it to finish. You're ready to go.

---

## Running the Tool

Open a command prompt in the project folder (same method as Step 2 above), then type:

```
python src/main.py
```

That's it. This processes all estates listed in `config/multi_estate.yaml`, detects duplicates across them, and produces the output files.

### Where to find the results

After a run, look in the **output/** folder:

- `combined_classified.jsonl` — the main output, used for Dataverse import
- `combined_classified.csv` — same data as a spreadsheet, open in Excel to browse results

Reports appear in the **reports/** folder:

- **Schemes report** — detected development schemes with confidence scores
- **Duplicate report** — files that appear in multiple estates
- **Folder tree** — Excel summary of the folder structure at different depths

---

## How-To Guide

### Load a new input file

1. Place your TreeSize export (`.xlsx`) or CSV file in the **context/** folder
2. Make sure it's listed in `config/multi_estate.yaml` (see "Add an estate" below)
3. Run: `python src/main.py`

### Create a config for a new estate

Each estate needs its own small config file. The easiest way is to copy an existing one.

1. Open `config/run_config.yaml` in a text editor (right-click > Open with > Notepad)
2. Save a copy with a new name, e.g. `config/run_config_newestate.yaml`
3. Edit these three lines:

```yaml
source:
  format: "treesize"                              # "treesize" for XLSX, "csv" for CSV files
  root_prefix: "\\server\share\path"              # the network path prefix to strip
  source_system: "MyEstateName"                   # a short label for this estate
```

- **format** — use `"treesize"` for TreeSize XLSX exports, or `"csv"` for CSV files
- **root_prefix** — the network path that appears at the start of every file path in the export. This gets stripped off so the tool sees clean relative paths. Copy it exactly from one of the paths in your export.
- **source_system** — a short name for this estate (appears in the output)

Leave everything else as-is. Then add the estate to `config/multi_estate.yaml` (see below).

### Add an estate to multi-estate processing

Open `config/multi_estate.yaml` in Notepad and add a new block under `estates:`:

```yaml
estates:
  - name: "TechServe"
    config: "config/run_config.yaml"
    input: "context/techserve_export.xlsx"

  - name: "Win95"
    config: "config/run_config_win95.yaml"
    input: "context/win95_export.xlsx"

  # Add your new estate here:
  - name: "NewEstate"
    config: "config/run_config_newestate.yaml"
    input: "context/newestate_export.xlsx"
```

Make sure the spacing matches the entries above it (two spaces before the dash).

Then run: `python src/main.py`

### Add a new document type to the dictionary

Open `config/dictionaries/industry/housing.yaml` in Notepad. Under the `types:` section, add a new entry:

```yaml
  "My New Type":
    tokens: ["keyword1", "keyword2"]           # words that identify this document type
    abbreviations: ["kw1"]                     # short forms (optional)
    belongs_to: "Finance"                      # category it falls under
```

The tool will now recognise files containing those keywords as your new type.

**Available categories**: Finance, Legal, HR, Health & Safety, IT, Administration, Construction, Sales, Planning, Utilities, Maintenance.

### Add a retention rule for a document type

Open `config/retention/housing.yaml` in Notepad. Under the `rules:` section, add:

```yaml
  "My New Type":
    years: 7
    basis: "Reason for this retention period"
    sensitivity: "medium"
```

- **years** — how many years to keep it (or `permanent`)
- **basis** — why (regulatory reference or business reason)
- **sensitivity** — `high`, `medium`, or `low`

### Adjust confidence thresholds

Open `config/classification/weights.yaml` in Notepad. The key settings to look at:

```yaml
thresholds:
  minimum_confidence: 0.15    # below this = classified as "Unknown"
  high_confidence: 0.60       # above this = "High" confidence

readiness:
  high_threshold: 0.60        # "Ready" for Phase 2
  medium_threshold: 0.40      # "Needs review"
```

- **Lowering** thresholds = more files classified as Ready (but potentially less accurate)
- **Raising** thresholds = fewer files classified as Ready (but higher accuracy)

### Exclude certain files or folders

Open `config/schema.yaml` in Notepad. Under `exclusion_patterns:`, you can add:

```yaml
exclusion_patterns:
  filenames: ["Thumbs.db", "desktop.ini", ".DS_Store"]    # exact filenames to skip
  prefixes: ["~$"]                                         # filename prefixes to skip
  extensions: ["tmp", "bak"]                               # file extensions to skip
```

---

## Troubleshooting

**"python is not recognized"** — Python wasn't added to PATH during installation. Reinstall Python and make sure you tick "Add Python to PATH".

**"No module named pandas"** — You need to install dependencies. Run `pip install -r requirements.txt` from the project folder.

**Wrong results / paths look broken** — Check that `root_prefix` in your config matches the actual path prefix in your export file. Open the export in Excel, look at the Path column, and copy the common prefix exactly.

---

## Advanced: Running a Single Estate or a Different Config

By default `python src/main.py` runs `config/multi_estate.yaml`. You can override this:

```
# Use a different multi-estate config
python src/main.py --multi config/my_other_multi.yaml

# Run a single estate on its own
python src/main.py --config config/run_config.yaml --input context/your_export.xlsx
```
