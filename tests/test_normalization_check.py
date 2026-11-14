import importlib.util
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("normalization_check", os.path.join(ROOT, "eval", "normalization_check.py"))
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


def test_words_ignores_case_punctuation_and_suffix_apostrophes():
    assert check.words("Saat on dört kırk beş'te.") == check.words("saat on dört kırk beşte")
    assert check.words("IŞIK, İyi!") == ["ışık", "iyi"]
    assert check.words("1.000 TL") == ["1", "000", "lira"]


def test_words_splits_fused_number_words_only():
    assert check.words("onbeş yirmibeş ikiyüzelli") == ["on", "beş", "yirmi", "beş", "iki", "yüz", "elli"]
    assert check.words("bina altın onlar") == ["bina", "altın", "onlar"]


def test_key_keeps_numbers_months_units_and_marks_ordinals():
    assert check.key(check.words("Doğum tarihiniz on dört mart bin dokuz yüz seksen sekiz mi?")) == \
        ["on", "dört", "mart", "bin", "dokuz", "yüz", "seksen", "sekiz"]
    assert check.key(check.words("saat dokuz sıfır beşte kapınızda")) == ["dokuz", "sıfır", "beş"]
    assert check.key(check.words("üçüncü taksiti")) == ["üç."]
    assert check.key(check.words("dördüncü kat")) == ["dört."]
    assert check.key(check.words("üç taksiti")) == ["üç"]
    assert check.key(check.words("ofisimiz binanın katında")) == []


def test_judge_accepts_any_listed_reading():
    readings = ["Kartınızın son dört hanesi dört sekiz iki bir.", "Kartınızın son dört hanesi kırk sekiz yirmi bir."]
    assert check.judge("kartınızın son dört hanesi kırk sekiz yirmi bir", readings)["exact"]
    partly = check.judge("Kartınızın son dört tanesi dört sekiz iki bir", readings)
    assert partly["key"] and not partly["exact"] and partly["errors"] == 1
    assert not check.judge("kartınızın son dört hanesi kırk iki", readings)["key"]


def test_every_line_has_a_reading_the_checker_can_match():
    path = os.path.join(ROOT, "eval", "normalization_tr.jsonl")
    items = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    assert len({item["id"] for item in items}) == len(items)
    for item in items:
        assert item["spoken"], item["id"]
        for reading in item["spoken"]:
            assert not any(ch.isdigit() for ch in reading), (item["id"], reading)
            assert check.judge(reading, item["spoken"])["exact"]
