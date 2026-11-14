"""
Saraga Census -- Kaggle-mirror version
=============================================================
Reads Saraga Carnatic directly from the Kaggle dataset
(desolationofsmaug/saraga-carnatic-music-dataset), parsing the per-recording
.json metadata files. No mirdata dependency.

Run AFTER:
    import kagglehub
    path = kagglehub.dataset_download("desolationofsmaug/saraga-carnatic-music-dataset")

Then set DATA_PATH = path (or paste the printed path).

PASS 1 -> raga inventory + per-raga recording/artist counts  (no pair list needed)
PASS 2 -> allied-pair depth overlay                          (fill in AFTER pass 1)
-------------------------------------------------------------
"""

import os
import re
import json
import glob
import unicodedata
from collections import defaultdict

import pandas as pd


# =============================================================
# CONFIG  -- set DATA_PATH to the kagglehub path
# =============================================================
# DATA_PATH = None  # e.g. "/root/.cache/kagglehub/datasets/desolationofsmaug/..."
DATA_PATH = "/kaggle/input/saraga-carnatic-music-dataset"
OUT_DIR = "/content/census_out"
os.makedirs(OUT_DIR, exist_ok=True)


# =============================================================
# NORMALIZATION
# =============================================================
def normalize_name(s):
    """Strip diacritics so 'Ṣanmukhapriya' == 'Shanmukhapriya'-ish.
    We fold to ASCII and lowercase for grouping, but keep a display version."""
    if not s:
        return "UNKNOWN"
    # NFKD decomposition drops combining marks
    decomposed = unicodedata.normalize("NFKD", s)
    ascii_form = decomposed.encode("ascii", "ignore").decode("ascii")
    ascii_form = ascii_form.strip()
    return ascii_form if ascii_form else s.strip()


def raga_key(name):
    """Grouping key: normalized + lowercased + spaces collapsed."""
    n = normalize_name(name).lower()
    n = re.sub(r"\s+", " ", n).strip()
    return n


# =============================================================
# PARSE ONE RECORDING
# =============================================================
def parse_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # --- raga ---
    raga_disp = "UNKNOWN"
    raaga = meta.get("raaga") or meta.get("raga")
    if isinstance(raaga, list) and raaga:
        raga_disp = normalize_name(raaga[0].get("name"))
    elif isinstance(raaga, dict):
        raga_disp = normalize_name(raaga.get("name"))

    # --- lead artist: prefer the artist flagged lead==True in `artists`,
    #     fall back to album_artists ---
    lead_artist = None
    artists = meta.get("artists") or []
    for a in artists:
        if a.get("lead") and a.get("artist", {}).get("name"):
            lead_artist = a["artist"]["name"]
            break
    if not lead_artist:
        alb = meta.get("album_artists") or []
        if isinstance(alb, list) and alb:
            lead_artist = alb[0].get("name")
    lead_artist = normalize_name(lead_artist) if lead_artist else "UNKNOWN"

    # --- lead instrument / voice type (useful later, cheap to grab now) ---
    lead_instrument = "UNKNOWN"
    for a in artists:
        if a.get("lead"):
            lead_instrument = (a.get("instrument") or {}).get("name", "UNKNOWN")
            break

    # --- concert / form / taala ---
    def first_name(field):
        v = meta.get(field)
        if isinstance(v, list) and v:
            return normalize_name(v[0].get("name") or v[0].get("title"))
        if isinstance(v, dict):
            return normalize_name(v.get("name") or v.get("title"))
        return "UNKNOWN"

    concert = first_name("concert")
    form = first_name("form")
    taala = first_name("taala")

    # --- annotation coverage: check sibling files in the same folder ---
    folder = os.path.dirname(json_path)
    stem = os.path.splitext(os.path.basename(json_path))[0]
    siblings = os.listdir(folder)
    def has(suffix):
        return any(fn.startswith(stem) and fn.endswith(suffix) for fn in siblings)

    return {
        "track_id": os.path.relpath(folder, DATA_PATH),
        "title": meta.get("title", "UNKNOWN"),
        "raga": raga_disp,
        "raga_key": raga_key(raga_disp),
        "artist": lead_artist,
        "lead_instrument": lead_instrument,
        "concert": concert,
        "form": form,
        "taala": taala,
        "has_tonic": has(".ctonic.txt"),
        "has_pitch": has(".pitch.txt"),
        "has_pitch_vocal": has(".pitch-vocal.txt"),
        "has_phrases": has(".mphrases-manual.txt"),
        "has_sections": has(".sections-manual.txt"),
        "has_mp3": has(".mp3"),
    }


def build_metadata_table():
    assert DATA_PATH, "Set DATA_PATH to the kagglehub download path first."
    json_files = glob.glob(os.path.join(DATA_PATH, "**", "*.json"), recursive=True)
    print(f"Found {len(json_files)} JSON metadata files.")
    rows = []
    for jp in json_files:
        try:
            rows.append(parse_json(jp))
        except Exception as e:
            print(f"  [skip] {jp}: {e}")
    df = pd.DataFrame(rows)
    return df


# =============================================================
# PASS 1  --  RAGA INVENTORY
# =============================================================
def pass1_inventory(df):
    print("\n" + "=" * 60)
    print("PASS 1 : RAGA INVENTORY")
    print("=" * 60)
    print(f"Total recordings: {len(df)}")
    print(f"Distinct ragas:   {df['raga_key'].nunique()}")
    print(f"Distinct artists: {df['artist'].nunique()}")
    print(f"Distinct concerts:{df['concert'].nunique()}")

    per_raga = (
        df.groupby("raga_key")
          .agg(display_name=("raga", "first"),
               recordings=("track_id", "count"),
               artists=("artist", "nunique"),
               concerts=("concert", "nunique"))
          .sort_values("recordings", ascending=False)
    )

    print("\n--- Per-raga depth (all ragas) ---")
    with pd.option_context("display.max_rows", None):
        print(per_raga[["display_name", "recordings", "artists", "concerts"]])

    print("\n--- Annotation coverage ---")
    for col in ["has_tonic", "has_pitch", "has_pitch_vocal",
                "has_phrases", "has_sections", "has_mp3"]:
        print(f"  {col:18s}: {df[col].mean():.1%}")

    learnable = per_raga[(per_raga["artists"] >= 2) & (per_raga["recordings"] >= 3)]
    print(f"\nRagas with >=2 artists & >=3 recordings (learned-path eligible): "
          f"{len(learnable)} of {len(per_raga)}")
    print("\nLearned-eligible ragas:")
    print(learnable[["display_name", "recordings", "artists"]].to_string())

    per_raga.to_csv(os.path.join(OUT_DIR, "per_raga_depth.csv"))
    df.to_csv(os.path.join(OUT_DIR, "recordings_metadata.csv"), index=False)
    print(f"\nSaved: per_raga_depth.csv  and  recordings_metadata.csv")
    print(">>> Use per_raga_depth.csv as your MENU to build the allied-pair list. <<<")
    return per_raga


# =============================================================
# PASS 2  --  ALLIED-PAIR DEPTH  (fill in AFTER pass 1)
# =============================================================
# Use the raga_key spelling (normalized, lowercase) as it appears in
# per_raga_depth.csv's index. Group by why the pair is hard.
ALLIED_PAIRS = [
    # ("bilahari", "sankarabharanam", "scale_identical"),
    # ("saveri",   "gaula",           "same_janaka"),
]


def pass2_allied_pairs(per_raga):
    if not ALLIED_PAIRS:
        print("\n[PASS 2 skipped] ALLIED_PAIRS empty. Fill from per_raga_depth.csv, re-run.")
        return None

    print("\n" + "=" * 60)
    print("PASS 2 : ALLIED-PAIR DEPTH")
    print("=" * 60)

    def depth(rk):
        if rk in per_raga.index:
            r = per_raga.loc[rk]
            return int(r["recordings"]), int(r["artists"])
        return 0, 0

    rows = []
    for a, b, cat in ALLIED_PAIRS:
        ar, aa = depth(a); br, ba = depth(b)
        min_recs, min_art = min(ar, br), min(aa, ba)
        if min_recs == 0:
            status = "MISSING (one side absent)"
        elif min_art >= 2 and min_recs >= 3:
            status = "LEARNED-ELIGIBLE"
        elif min_recs >= 1:
            status = "FLOOR-ONLY"
        else:
            status = "UNKNOWN"
        rows.append(dict(raga_A=a, recs_A=ar, artists_A=aa,
                         raga_B=b, recs_B=br, artists_B=ba,
                         category=cat, min_recs=min_recs,
                         min_artists=min_art, status=status))

    pair_df = pd.DataFrame(rows).sort_values(["category", "min_artists"],
                                             ascending=[True, False])
    print(pair_df.to_string(index=False))
    pair_df.to_csv(os.path.join(OUT_DIR, "allied_pair_depth.csv"), index=False)
    print(f"\nSaved: allied_pair_depth.csv")
    print(">>> Decides: full learned stack vs. transition-vector floor. <<<")
    return pair_df


# =============================================================
# MAIN
# =============================================================
def main():
    df = build_metadata_table()
    per_raga = pass1_inventory(df)
    pass2_allied_pairs(per_raga)


if __name__ == "__main__":
    main()