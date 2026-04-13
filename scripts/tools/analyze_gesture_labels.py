#!/usr/bin/env python3
"""
Comprehensive analysis of gesture label files for the SutureBot ACT project.
Computes per-gesture statistics, per-subtask sequence patterns, and anomaly detection.

Output saved to ~/paper_results/gesture_label_statistics.txt
"""

import os
import sys
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

LABELS_DIR = Path(os.path.expanduser("~/data/labels"))
OUTPUT_DIR = Path(os.path.expanduser("~/paper_results"))
OUTPUT_FILE = OUTPUT_DIR / "gesture_label_statistics.txt"

KNOWN_GESTURES = {"G2", "G3", "G6", "G7", "G10", "G11", "G13", "G14", "G15", "G16"}

# JIGSAWS gesture descriptions
GESTURE_DESCRIPTIONS = {
    "G2": "Positioning needle tip",
    "G3": "Pushing needle through tissue",
    "G6": "Pulling suture with left hand",
    "G7": "Pulling suture / extracting needle",
    "G10": "Loosening more suture",
    "G11": "Dropping suture and moving to end points",
    "G13": "Reaching for suture tail / positioning",
    "G14": "Wrapping / looping suture around instrument",
    "G15": "Pulling / tightening knot",
    "G16": "Transferring needle / suture handoff",
}


def classify_subtask(dirpath):
    """Classify a directory as NT (needle throw) or KT (knot tying), with recovery flag."""
    dirname = os.path.basename(dirpath)
    is_recovery = "recovery" in dirname.lower()
    if "needle_throw" in dirname:
        return "NT", is_recovery
    elif "knot_tying" in dirname:
        return "KT", is_recovery
    else:
        return "UNKNOWN", is_recovery


def parse_label_file(filepath):
    """Parse a label file and return list of (start, end, gesture) tuples."""
    segments = []
    with open(filepath, "r") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                segments.append(("PARSE_ERROR", line_no, line))
                continue
            try:
                start = int(parts[0])
                end = int(parts[1])
                gesture = parts[2].strip()
                segments.append((start, end, gesture))
            except ValueError:
                segments.append(("PARSE_ERROR", line_no, line))
    return segments


def detect_anomalies(segments, filepath):
    """Detect anomalies in a list of segments."""
    anomalies = []
    valid_segments = [(s, e, g) for s, e, g in segments if s != "PARSE_ERROR"]

    # Check for unknown gestures
    for s, e, g in valid_segments:
        if g not in KNOWN_GESTURES:
            anomalies.append(f"  Unknown gesture '{g}' at frames {s}-{e} in {filepath}")

    # Check for overlapping segments
    for i in range(len(valid_segments) - 1):
        s1, e1, g1 = valid_segments[i]
        s2, e2, g2 = valid_segments[i + 1]
        if s2 <= e1:
            anomalies.append(
                f"  Overlap: {g1}({s1}-{e1}) and {g2}({s2}-{e2}) in {filepath}"
            )

    # Check for gaps between segments
    for i in range(len(valid_segments) - 1):
        s1, e1, g1 = valid_segments[i]
        s2, e2, g2 = valid_segments[i + 1]
        if s2 > e1 + 1:
            gap_size = s2 - e1 - 1
            anomalies.append(
                f"  Gap of {gap_size} frames between {g1}(end={e1}) and {g2}(start={s2}) in {filepath}"
            )

    # Check for zero or negative duration
    for s, e, g in valid_segments:
        duration = e - s + 1
        if duration <= 0:
            anomalies.append(
                f"  Non-positive duration: {g} frames {s}-{e} (duration={duration}) in {filepath}"
            )

    # Check for non-monotonic starts
    for i in range(len(valid_segments) - 1):
        if valid_segments[i + 1][0] < valid_segments[i][0]:
            anomalies.append(
                f"  Non-monotonic: segment {i+2} starts at {valid_segments[i+1][0]} < segment {i+1} start {valid_segments[i][0]} in {filepath}"
            )

    # Check for parse errors
    for item in segments:
        if item[0] == "PARSE_ERROR":
            anomalies.append(
                f"  Parse error at line {item[1]}: '{item[2]}' in {filepath}"
            )

    # Check if first segment starts at 0
    if valid_segments and valid_segments[0][0] != 0:
        anomalies.append(
            f"  First segment starts at frame {valid_segments[0][0]} (not 0) in {filepath}"
        )

    return anomalies


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all label files
    all_files = []
    for root, dirs, files in os.walk(LABELS_DIR):
        for fname in files:
            if fname.endswith("_labels.txt"):
                all_files.append(os.path.join(root, fname))

    all_files.sort()

    # 1. File counts
    tissue_subtask_counts = defaultdict(lambda: defaultdict(int))
    subtask_counts = Counter()  # NT, KT
    recovery_counts = Counter()
    files_by_subtask = defaultdict(list)  # "NT" -> [filepaths], "KT" -> [filepaths]
    files_by_tissue_subtask = defaultdict(list)

    for fpath in all_files:
        # Extract tissue number from path
        parts = Path(fpath).parts
        tissue_dir = [p for p in parts if p.startswith("tissue_")]
        tissue_num = tissue_dir[0] if tissue_dir else "unknown"

        subtask_dir = os.path.basename(os.path.dirname(fpath))
        subtask, is_recovery = classify_subtask(os.path.dirname(fpath))

        subtask_counts[subtask] += 1
        if is_recovery:
            recovery_counts[subtask] += 1
        files_by_subtask[subtask].append(fpath)
        tissue_subtask_counts[tissue_num][subtask] += 1
        files_by_tissue_subtask[(tissue_num, subtask)].append(fpath)

    # 2. Parse all files and collect gesture data
    all_segments = []  # (start, end, gesture, subtask, tissue, filepath)
    gesture_durations = defaultdict(list)  # gesture -> [durations]
    gesture_subtasks = defaultdict(set)  # gesture -> {subtasks}
    gesture_tissues = defaultdict(set)  # gesture -> {tissues}
    gesture_counts = Counter()  # gesture -> total segment count
    all_anomalies = []

    # Per-subtask sequence tracking
    subtask_sequences = defaultdict(list)  # subtask -> [sequence_tuples]

    for fpath in all_files:
        tissue_dir = [p for p in Path(fpath).parts if p.startswith("tissue_")]
        tissue_num = tissue_dir[0] if tissue_dir else "unknown"
        subtask, is_recovery = classify_subtask(os.path.dirname(fpath))

        segments = parse_label_file(fpath)
        valid_segments = [(s, e, g) for s, e, g in segments if s != "PARSE_ERROR"]

        # Anomaly detection
        anoms = detect_anomalies(segments, os.path.relpath(fpath, LABELS_DIR))
        all_anomalies.extend(anoms)

        # Sequence pattern
        gesture_sequence = tuple(g for s, e, g in valid_segments)
        subtask_sequences[subtask].append(gesture_sequence)

        for s, e, g in valid_segments:
            duration = e - s + 1
            gesture_durations[g].append(duration)
            gesture_subtasks[g].add(subtask)
            gesture_tissues[g].add(tissue_num)
            gesture_counts[g] += 1
            all_segments.append((s, e, g, subtask, tissue_num, fpath))

    # 3. Compute statistics
    lines = []

    def out(s=""):
        lines.append(s)

    out("=" * 80)
    out("GESTURE LABEL STATISTICS  -- SutureBot ACT Project")
    out(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    out(f"Labels directory: {LABELS_DIR}")
    out("=" * 80)

    # Section 1: File Counts
    out()
    out("-" * 80)
    out("1. LABEL FILE COUNTS")
    out("-" * 80)
    out(f"\nTotal label files found: {len(all_files)}")
    out(f"  Needle Throw (NT):  {subtask_counts['NT']}")
    out(f"  Knot Tying (KT):    {subtask_counts['KT']}")
    if subtask_counts.get("UNKNOWN", 0):
        out(f"  Unknown subtask:    {subtask_counts['UNKNOWN']}")
    out(f"\n  Recovery episodes included:")
    out(f"    NT recovery: {recovery_counts.get('NT', 0)}")
    out(f"    KT recovery: {recovery_counts.get('KT', 0)}")

    out(f"\nPer-tissue breakdown:")
    out(f"  {'Tissue':<12} {'NT':>6} {'KT':>6} {'Total':>6}")
    out(f"  {'-'*12} {'-'*6} {'-'*6} {'-'*6}")
    for tissue in sorted(tissue_subtask_counts.keys(), key=lambda x: int(x.split("_")[1])):
        nt = tissue_subtask_counts[tissue].get("NT", 0)
        kt = tissue_subtask_counts[tissue].get("KT", 0)
        total = nt + kt
        out(f"  {tissue:<12} {nt:>6} {kt:>6} {total:>6}")
    total_nt = subtask_counts["NT"]
    total_kt = subtask_counts["KT"]
    out(f"  {'TOTAL':<12} {total_nt:>6} {total_kt:>6} {total_nt + total_kt:>6}")

    # Section 2: Per-Gesture Statistics
    out()
    out("-" * 80)
    out("2. PER-GESTURE STATISTICS")
    out("-" * 80)

    all_gestures_seen = sorted(gesture_counts.keys(), key=lambda g: int(g[1:]))
    total_segments = sum(gesture_counts.values())

    out(f"\nTotal gesture segments: {total_segments}")
    out(f"Unique gestures found: {len(all_gestures_seen)}")
    out(f"Known gestures (10): {', '.join(sorted(KNOWN_GESTURES, key=lambda g: int(g[1:])))}")
    unknown_gestures = set(all_gestures_seen) - KNOWN_GESTURES
    if unknown_gestures:
        out(f"UNKNOWN gestures found: {', '.join(sorted(unknown_gestures))}")
    else:
        out(f"All gestures are within the known set.")

    out()
    out(f"{'Gesture':<8} {'Description':<45} {'Count':>6} {'%':>6} {'Subtasks':<10}")
    out(f"{'-'*8} {'-'*45} {'-'*6} {'-'*6} {'-'*10}")
    for g in all_gestures_seen:
        desc = GESTURE_DESCRIPTIONS.get(g, "???")
        cnt = gesture_counts[g]
        pct = 100.0 * cnt / total_segments
        subs = ", ".join(sorted(gesture_subtasks[g]))
        out(f"{g:<8} {desc:<45} {cnt:>6} {pct:>5.1f}% {subs:<10}")

    out()
    out("Duration statistics (in frames):")
    out()
    out(f"{'Gesture':<8} {'Count':>6} {'Mean':>8} {'Std':>8} {'Min':>6} {'Q25':>6} {'Median':>6} {'Q75':>6} {'Max':>6}")
    out(f"{'-'*8} {'-'*6} {'-'*8} {'-'*8} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
    for g in all_gestures_seen:
        durs = np.array(gesture_durations[g])
        cnt = len(durs)
        mean = np.mean(durs)
        std = np.std(durs)
        mn = np.min(durs)
        q25 = np.percentile(durs, 25)
        med = np.median(durs)
        q75 = np.percentile(durs, 75)
        mx = np.max(durs)
        out(f"{g:<8} {cnt:>6} {mean:>8.1f} {std:>8.1f} {mn:>6} {q25:>6.0f} {med:>6.0f} {q75:>6.0f} {mx:>6}")

    # Per-gesture tissue distribution
    out()
    out("Per-gesture tissue distribution:")
    out()
    out(f"{'Gesture':<8} {'Tissues'}")
    out(f"{'-'*8} {'-'*60}")
    for g in all_gestures_seen:
        tissues = sorted(gesture_tissues[g], key=lambda x: int(x.split("_")[1]))
        out(f"{g:<8} {', '.join(tissues)} ({len(tissues)} tissues)")

    # Per-subtask gesture frequency
    out()
    out("Gesture frequency by subtask:")
    out()
    for subtask in ["NT", "KT"]:
        out(f"  {subtask}:")
        sub_gestures = Counter()
        for s, e, g, st, tissue, fpath in all_segments:
            if st == subtask:
                sub_gestures[g] += 1
        sub_total = sum(sub_gestures.values())
        for g in sorted(sub_gestures.keys(), key=lambda x: int(x[1:])):
            cnt = sub_gestures[g]
            pct = 100.0 * cnt / sub_total if sub_total else 0
            out(f"    {g}: {cnt:>5} ({pct:>5.1f}%)")
        out(f"    Total segments: {sub_total}")
        out()

    # Section 3: Per-Subtask Gesture Sequence Patterns
    out()
    out("-" * 80)
    out("3. PER-SUBTASK GESTURE SEQUENCE PATTERNS")
    out("-" * 80)

    for subtask in ["NT", "KT"]:
        out(f"\n{'='*40}")
        out(f"  {subtask}  -- Gesture Sequence Patterns")
        out(f"{'='*40}")

        sequences = subtask_sequences.get(subtask, [])
        if not sequences:
            out(f"  No sequences found for {subtask}")
            continue

        # Count unique sequences
        seq_counter = Counter(sequences)
        n_unique = len(seq_counter)
        n_total = len(sequences)

        out(f"\n  Total episodes: {n_total}")
        out(f"  Unique sequences: {n_unique}")

        # Show top sequences
        out(f"\n  Most common sequences (top 15):")
        for rank, (seq, count) in enumerate(seq_counter.most_common(15), 1):
            pct = 100.0 * count / n_total
            seq_str = " -> ".join(seq)
            out(f"    {rank:>2}. [{count:>4} episodes, {pct:>5.1f}%] {seq_str}")

        # Sequence length distribution
        seq_lengths = [len(s) for s in sequences]
        out(f"\n  Sequence length distribution:")
        out(f"    Mean: {np.mean(seq_lengths):.1f}, Std: {np.std(seq_lengths):.1f}")
        out(f"    Min: {min(seq_lengths)}, Max: {max(seq_lengths)}")
        len_counter = Counter(seq_lengths)
        out(f"    Length counts:")
        for length in sorted(len_counter.keys()):
            cnt = len_counter[length]
            pct = 100.0 * cnt / n_total
            out(f"      Length {length}: {cnt:>4} episodes ({pct:>5.1f}%)")

        # Transition matrix
        out(f"\n  Gesture transition matrix (row -> col):")
        gestures_in_subtask = sorted(
            set(g for seq in sequences for g in seq), key=lambda x: int(x[1:])
        )
        transition_counts = Counter()
        for seq in sequences:
            for i in range(len(seq) - 1):
                transition_counts[(seq[i], seq[i + 1])] += 1

        # Print header
        header = f"    {'From':<6}"
        for g in gestures_in_subtask:
            header += f" {g:>5}"
        out(header)
        out(f"    {'-'*6}" + f" {'-'*5}" * len(gestures_in_subtask))

        for g_from in gestures_in_subtask:
            row = f"    {g_from:<6}"
            for g_to in gestures_in_subtask:
                cnt = transition_counts.get((g_from, g_to), 0)
                if cnt > 0:
                    row += f" {cnt:>5}"
                else:
                    row += f" {'·':>5}"

            out(row)

        # Common starting and ending gestures
        start_counter = Counter(seq[0] for seq in sequences if seq)
        end_counter = Counter(seq[-1] for seq in sequences if seq)

        out(f"\n  Starting gesture distribution:")
        for g, cnt in start_counter.most_common():
            pct = 100.0 * cnt / n_total
            out(f"    {g}: {cnt:>4} ({pct:>5.1f}%)")

        out(f"\n  Ending gesture distribution:")
        for g, cnt in end_counter.most_common():
            pct = 100.0 * cnt / n_total
            out(f"    {g}: {cnt:>4} ({pct:>5.1f}%)")

    # Section 4: Anomalies
    out()
    out("-" * 80)
    out("4. ANOMALY DETECTION")
    out("-" * 80)

    if all_anomalies:
        # Categorize anomalies
        overlaps = [a for a in all_anomalies if "Overlap" in a]
        gaps = [a for a in all_anomalies if "Gap" in a]
        unknown = [a for a in all_anomalies if "Unknown gesture" in a]
        parse_errors = [a for a in all_anomalies if "Parse error" in a]
        non_positive = [a for a in all_anomalies if "Non-positive" in a]
        non_monotonic = [a for a in all_anomalies if "Non-monotonic" in a]
        non_zero_start = [a for a in all_anomalies if "not 0" in a]
        other = [a for a in all_anomalies if a not in overlaps + gaps + unknown + parse_errors + non_positive + non_monotonic + non_zero_start]

        out(f"\nTotal anomalies found: {len(all_anomalies)}")
        out(f"  Overlapping segments:     {len(overlaps)}")
        out(f"  Gaps between segments:    {len(gaps)}")
        out(f"  Unknown gestures:         {len(unknown)}")
        out(f"  Parse errors:             {len(parse_errors)}")
        out(f"  Non-positive durations:   {len(non_positive)}")
        out(f"  Non-monotonic starts:     {len(non_monotonic)}")
        out(f"  Non-zero first frame:     {len(non_zero_start)}")
        if other:
            out(f"  Other:                    {len(other)}")

        for category_name, category_list in [
            ("Overlapping segments", overlaps),
            ("Gaps between segments", gaps),
            ("Unknown gestures", unknown),
            ("Parse errors", parse_errors),
            ("Non-positive durations", non_positive),
            ("Non-monotonic starts", non_monotonic),
            ("Non-zero first frame", non_zero_start),
            ("Other", other),
        ]:
            if category_list:
                out(f"\n  {category_name} ({len(category_list)}):")
                for a in category_list[:50]:  # limit output
                    out(a)
                if len(category_list) > 50:
                    out(f"  ... and {len(category_list) - 50} more")
    else:
        out(f"\nNo anomalies detected! All {len(all_files)} files are clean.")

    # Section 5: Summary Statistics
    out()
    out("-" * 80)
    out("5. SUMMARY")
    out("-" * 80)

    total_frames_labeled = sum(e - s + 1 for s, e, g, st, tissue, fpath in all_segments)
    out(f"\n  Total label files:        {len(all_files)}")
    out(f"  Total gesture segments:   {total_segments}")
    out(f"  Total frames labeled:     {total_frames_labeled:,}")
    out(f"  Unique gestures used:     {len(all_gestures_seen)}")
    out(f"  Mean segments/episode:    {total_segments / len(all_files):.1f}")

    all_durs = np.array([e - s + 1 for s, e, g, st, tissue, fpath in all_segments])
    out(f"  Overall duration stats:   mean={np.mean(all_durs):.1f}, std={np.std(all_durs):.1f}, median={np.median(all_durs):.0f}")

    # Episode total frame counts
    episode_frames = defaultdict(int)
    for s, e, g, st, tissue, fpath in all_segments:
        episode_frames[fpath] = max(episode_frames[fpath], e + 1)
    ep_lens = list(episode_frames.values())
    out(f"  Episode length (frames):  mean={np.mean(ep_lens):.0f}, std={np.std(ep_lens):.0f}, min={min(ep_lens)}, max={max(ep_lens)}")

    # NT vs KT episode lengths
    for subtask in ["NT", "KT"]:
        ep_lens_sub = []
        for fpath, max_frame in episode_frames.items():
            st, _ = classify_subtask(os.path.dirname(fpath))
            if st == subtask:
                ep_lens_sub.append(max_frame)
        if ep_lens_sub:
            out(f"  {subtask} episode length:     mean={np.mean(ep_lens_sub):.0f}, std={np.std(ep_lens_sub):.0f}, min={min(ep_lens_sub)}, max={max(ep_lens_sub)}")

    out()
    out("=" * 80)
    out("END OF REPORT")
    out("=" * 80)

    # Write output
    report = "\n".join(lines)
    print(report)

    with open(OUTPUT_FILE, "w") as f:
        f.write(report + "\n")
    print(f"\n[Saved to {OUTPUT_FILE}]")


if __name__ == "__main__":
    main()
