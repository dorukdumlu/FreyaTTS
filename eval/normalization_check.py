#!/usr/bin/env python3
"""Check how FreyaTTS reads numbers, dates, amounts and symbols in customer-facing Turkish text.

eval/normalization_tr.jsonl holds sentences as a bank or insurer writes them, each with the readings a
careful human agent could give ("spoken"; the first one is the reference reading). Two levels:

  text   normalize() must turn the sentence into one of those readings, using only characters in the
         model's vocabulary. Deterministic, no GPU.
  audio  the normalized text is synthesized, band-limited to 8 kHz and transcribed back with faster-whisper
         (Turkish, beam 5, as in eval/benchmark.py). Number tokens are suppressed while decoding, so Whisper
         spells numbers out and the comparison never depends on how it formats digits. A line is right when
         its number words match one of the readings ("key") and exactly right when every word does
         ("exact"). --oracle also synthesizes the reference reading with normalization off: what the
         acoustic model does when the text is already right.

The pipeline is deterministic for a text, seed and step count, so --t-scales repeats every line with the
predicted duration scaled by a few percent: the same voice, a different draw.

Usage:
  python3 eval/normalization_check.py                                          # text level
  python3 eval/normalization_check.py --audio --oracle --out results/normalization
  python3 eval/normalization_check.py --repo ../FreyaTTS-main --label main     # another checkout's normalize()
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys

JUDGE_SR = 8000
HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------- comparing readings --------------------------

NUMBER_WORDS = ["sıfır", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz", "on", "yirmi",
                "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan", "yüz", "bin", "milyon", "milyar"]
MONTHS = ["ocak", "şubat", "mart", "nisan", "mayıs", "haziran", "temmuz", "ağustos", "eylül", "ekim", "kasım",
          "aralık"]
UNIT_WORDS = ["virgül", "nokta", "buçuk", "yüzde", "lira", "kuruş", "artı"]
KEY_WORDS = set(NUMBER_WORDS) | set(MONTHS) | set(UNIT_WORDS)

_longest_first = sorted(NUMBER_WORDS, key=len, reverse=True)
_NUMBER = re.compile("|".join(_longest_first))
_FUSED = re.compile("(?:" + "|".join(_longest_first) + ")+")
_STEM = re.compile("(" + "|".join(sorted(NUMBER_WORDS + ["dörd"] + MONTHS + ["lira", "kuruş"], key=len,
                                         reverse=True)) + ")(.+)")
_ORDINAL = re.compile(r"[ıiuü]?nc[ıiuü].*")
_CASE = re.compile(r"[dt][ae]n?|y?[aeıiuü]|n?[ıiuü]n|y?l[ae]|[sş]?[ae]r|l[ıiuü]k|s[ıiuü]")


def words(text):
    """Lowercase word list of a reading: punctuation and suffix apostrophes dropped, "onbeş" -> on beş."""
    t = text.replace("İ", "i").replace("I", "ı").lower()
    t = t.replace("â", "a").replace("î", "i").replace("û", "u")
    t = re.sub(r"['’`´]", "", t)  # "beş'te" and "beşte" are the same reading
    t = re.sub(r"[^a-zçğıöşü0-9]+", " ", t)
    out = []
    for w in t.split():
        if w == "tl":
            w = "lira"
        out.extend(_NUMBER.findall(w) if _FUSED.fullmatch(w) else [w])
    return out


def key(tokens):
    """The number, month and unit words of a word list, case endings dropped ("beşte" -> beş); an ordinal
    keeps a mark ("üçüncü" -> üç.) so it never matches the cardinal."""
    out = []
    for w in tokens:
        if w in KEY_WORDS:
            out.append(w)
            continue
        m = _STEM.fullmatch(w)
        if not m:
            continue
        stem = "dört" if m.group(1) == "dörd" else m.group(1)
        if _ORDINAL.fullmatch(m.group(2)):
            out.append(stem + ".")
        elif _CASE.fullmatch(m.group(2)):
            out.append(stem)
    return out


def edit_distance(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (x != y))
    return row[-1]


def judge(heard, readings):
    """Compare a transcript with the accepted readings of a line."""
    hyp = words(heard)
    refs = [words(r) for r in readings]
    best = min(refs, key=lambda r: edit_distance(hyp, r))
    return {"exact": hyp in refs, "key": key(hyp) in [key(r) for r in refs],
            "errors": edit_distance(hyp, best), "ref_words": len(best)}


def text_level(items, normalize, vocab):
    rows = []
    for item in items:
        sent = normalize(item["text"])
        unknown = "".join(sorted({ch for ch in sent if ch not in vocab}))
        reading_ok = words(sent) in [words(r) for r in item["spoken"]]
        rows.append({"id": item["id"], "cat": item["cat"], "text": item["text"], "sent": sent,
                     "reading_ok": reading_ok, "unknown_chars": unknown, "text_ok": reading_ok and not unknown})
    return rows


# ----------------------------- audio round trip ----------------------------

class RoundTrip:
    """Synthesize a text (normalization off) and transcribe it back; results are cached on disk."""

    def __init__(self, repo, device, steps, asr_name, cache_path):
        import torch
        from faster_whisper import WhisperModel
        from freyatts.pipeline import FreyaTTS

        class Shifted(torch.nn.Module):
            # the duration head predicts log(frames): adding log(scale) scales T before the pipeline's clamps
            def __init__(self, inner, shift):
                super().__init__()
                self.inner, self.shift = inner, shift

            def forward(self, x):
                return self.inner(x) + self.shift

        self.Shifted = Shifted
        self.tts = FreyaTTS.from_pretrained(device=device)
        self.steps, self.asr_name = steps, asr_name
        self.asr = WhisperModel(asr_name, device=device, compute_type="float16" if device == "cuda" else "int8")
        tok = self.asr.hf_tokenizer
        end = tok.token_to_id("<|endoftext|>")
        # no digit or currency tokens: "on dört kırk beş", never "14.45" or "1445"
        self.numerals = [i for i in range(end) if re.search(r"[0-9%$£€₺]", tok.decode([i]))]
        self.cache_path = cache_path
        self.cache = json.load(open(cache_path, encoding="utf-8")) if os.path.exists(cache_path) else {}

    def hear(self, sent, scale):
        """Returns (transcript, seconds of audio)."""
        import librosa
        import numpy as np

        k = hashlib.sha1(json.dumps([sent, scale, self.steps, self.tts.seed, self.asr_name]).encode()).hexdigest()
        if k not in self.cache:
            dur = self.tts.model.dur
            if scale != 1.0:
                self.tts.model.dur = self.Shifted(dur, math.log(scale))
            try:
                wav = self.tts._synth(sent, steps=self.steps, do_norm=False)[0]
            finally:
                self.tts.model.dur = dur
            wav8k = librosa.resample(wav.astype(np.float32), orig_sr=self.tts.sample_rate, target_sr=JUDGE_SR)
            wav16k = librosa.resample(wav8k, orig_sr=JUDGE_SR, target_sr=16000)
            segments, _ = self.asr.transcribe(wav16k, language="tr", beam_size=5, temperature=0.0,
                                              condition_on_previous_text=False,
                                              suppress_tokens=[-1] + self.numerals)
            heard = " ".join(seg.text.strip() for seg in segments).strip()
            self.cache[k] = {"sent": sent, "scale": scale, "heard": heard,
                             "seconds": round(len(wav) / self.tts.sample_rate, 2)}
        return self.cache[k]["heard"], self.cache[k]["seconds"]

    def save(self):
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=1)


def audio_level(rows, items, trip, scales, oracle):
    readings = {item["id"]: item["spoken"] for item in items}
    for n, row in enumerate(rows, 1):
        targets = [("audio", row["sent"])] + ([("oracle", readings[row["id"]][0])] if oracle else [])
        for name, sent in targets:
            row[name] = {}
            for scale in scales:
                heard, seconds = trip.hear(sent, scale)
                row[name][str(scale)] = {"heard": heard, "seconds": seconds, **judge(heard, readings[row["id"]])}
        first = row["audio"][str(scales[0])]
        print(f"[{n}/{len(rows)}] {row['id']} key={'ok' if first['key'] else 'NO'} | {row['sent'][:44]!r} -> "
              f"{first['heard'][:44]!r}", flush=True)
        if n % 10 == 0:
            trip.save()
    trip.save()


# ----------------------------- report --------------------------------------

def summarize(rows, scales):
    cats = sorted({r["cat"] for r in rows}, key=lambda c: [r["cat"] for r in rows].index(c))
    table = []
    for cat in cats + ["all"]:
        sel = [r for r in rows if cat in (r["cat"], "all")]
        line = {"cat": cat, "n": len(sel), "reading_ok": sum(r["reading_ok"] for r in sel),
                "in_vocab": sum(not r["unknown_chars"] for r in sel), "text_ok": sum(r["text_ok"] for r in sel)}
        for name in ("audio", "oracle"):
            if name in sel[0]:
                first = [r[name][str(scales[0])] for r in sel]
                every = [r[name][str(s)] for r in sel for s in scales]
                line[name] = {"key": sum(x["key"] for x in first), "exact": sum(x["exact"] for x in first),
                              "key_all_scales": sum(x["key"] for x in every), "runs": len(every),
                              "wer": round(sum(x["errors"] for x in first) / sum(x["ref_words"] for x in first), 3)}
        table.append(line)
    return table


def print_table(table, label, scales):
    audio = "audio" in table[0]
    oracle = "oracle" in table[0]
    head = ["category", "n", "reading ok", "in vocab", "text ok"]
    if audio:
        head += [f"key @T", f"exact @T", f"key, {len(scales)} T-scales", "WER @T"]
    if oracle:
        head += ["oracle key @T", f"oracle key, {len(scales)} T-scales"]
    print(f"\n### {label}\n")
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for t in table:
        cells = [t["cat"], t["n"], t["reading_ok"], t["in_vocab"], t["text_ok"]]
        if audio:
            a = t["audio"]
            cells += [a["key"], a["exact"], f"{a['key_all_scales']}/{a['runs']}", a["wer"]]
        if oracle:
            o = t["oracle"]
            cells += [o["key"], f"{o['key_all_scales']}/{o['runs']}"]
        print("| " + " | ".join(str(c) for c in cells) + " |")


def main():
    ap = argparse.ArgumentParser(description="Number-reading check for FreyaTTS normalization")
    ap.add_argument("--data", default=os.path.join(HERE, "normalization_tr.jsonl"))
    ap.add_argument("--repo", default=os.path.dirname(HERE), help="FreyaTTS checkout whose normalize() is checked")
    ap.add_argument("--label", default="current")
    ap.add_argument("--audio", action="store_true", help="also synthesize and transcribe back (needs a GPU)")
    ap.add_argument("--oracle", action="store_true", help="with --audio: also synthesize the reference reading")
    ap.add_argument("--t-scales", default="1.0", help="duration scales, first one is the headline, e.g. 1.0,0.92,1.08")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--asr", default="large-v3", help="faster-whisper model (eval/benchmark.py uses large-v3)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None, help="directory for <label>.json and the round-trip cache")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    repo = os.path.abspath(args.repo)
    sys.path.insert(0, repo)
    from freyatts.pipeline import normalize

    vocab = json.load(open(os.path.join(repo, "freyatts", "char_vocab.json"), encoding="utf-8"))
    items = [json.loads(line) for line in open(args.data, encoding="utf-8") if line.strip()]
    scales = [float(s) for s in args.t_scales.split(",")]

    rows = text_level(items, normalize, vocab)
    if args.audio:
        out = args.out or "."
        trip = RoundTrip(repo, args.device, args.steps, args.asr, os.path.join(out, "roundtrip_cache.json"))
        audio_level(rows, items, trip, scales, args.oracle)

    table = summarize(rows, scales)
    print_table(table, args.label, scales)
    for r in rows:
        if not r["text_ok"]:
            why = "unknown " + repr(r["unknown_chars"]) if r["reading_ok"] else "reading"
            print(f"  text  {r['id']:9} {why:14} {r['text']!r} -> {r['sent']!r}")
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, f"{args.label}.json"), "w", encoding="utf-8") as f:
            json.dump({"label": args.label, "asr": args.asr, "steps": args.steps, "t_scales": scales,
                       "summary": table, "rows": rows}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
