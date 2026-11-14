from freyatts.pipeline import expand_digits


def test_clock_minute_keeps_leading_zero():
    assert expand_digits("Saat 09:05'te") == "Saat dokuz sıfır beş'te"
    assert expand_digits("Saat 14:00'te") == "Saat on dört'te"
    assert expand_digits("Saat 14:45") == "Saat on dört kırk beş"


def test_decimal_comma():
    assert expand_digits("Faiz 3,05 oldu.") == "Faiz üç virgül sıfır beş oldu."
    assert expand_digits("Faiz 3,5 oldu.") == "Faiz üç virgül beş oldu."
    assert expand_digits("Bakiye 1.250,75") == "Bakiye bin iki yüz elli virgül yetmiş beş"


def test_percent():
    assert expand_digits("%15 indirim") == "yüzde on beş indirim"
    assert expand_digits("% 3,5 faiz") == "yüzde üç virgül beş faiz"


def test_unchanged():
    assert expand_digits("1, 2, 3") == "bir, iki, üç"
    assert expand_digits("Poliçe 4012345.") == "Poliçe dört sıfır bir iki üç dört beş."
    assert expand_digits("1.250 kişi") == "bin iki yüz elli kişi"
    assert expand_digits("%") == "%"


from freyatts.pipeline import normalize, spell_ordinal


def test_dates():
    assert expand_digits("14.03.1988") == "on dört Mart bin dokuz yüz seksen sekiz"
    assert expand_digits("Son ödeme 05/12/2026.") == "Son ödeme beş Aralık iki bin yirmi altı."
    assert expand_digits("2026-11-14 tarihinde") == "on dört Kasım iki bin yirmi altı tarihinde"
    assert expand_digits("01.01.2027'de") == "bir Ocak iki bin yirmi yedi'de"
    # not a date: read part by part instead of as one integer
    assert expand_digits("32.13.2026") == "otuz iki nokta on üç nokta iki bin yirmi altı"


def test_dots_are_thousands_only_in_groups_of_three():
    assert expand_digits("1.250.000") == "bir milyon iki yüz elli bin"
    assert expand_digits("3.5") == "üç virgül beş"
    assert expand_digits("2450.75") == "iki bin dört yüz elli virgül yetmiş beş"
    assert expand_digits("1.2.3") == "bir nokta iki nokta üç"


def test_leading_zeros_keep_every_digit():
    assert expand_digits("son dört hanesi 0042") == "son dört hanesi sıfır sıfır dört iki"
    assert expand_digits("007") == "sıfır sıfır yedi"
    assert expand_digits("0") == "sıfır"
    assert expand_digits("0,5") == "sıfır virgül beş"


def test_lira_amounts():
    assert expand_digits("2.450,75 TL") == "iki bin dört yüz elli lira yetmiş beş kuruş"
    assert expand_digits("2450.75 TL") == "iki bin dört yüz elli lira yetmiş beş kuruş"
    assert expand_digits("10,00 TL") == "on lira"
    assert expand_digits("0,50 TL") == "elli kuruş"
    assert expand_digits("0,5 TL") == "elli kuruş"
    assert expand_digits("₺250 iade") == "iki yüz elli lira iade"
    assert expand_digits("49,90₺.") == "kırk dokuz lira doksan kuruş."
    assert expand_digits("1250TL") == "bin iki yüz elli lira"
    assert expand_digits("250 TL'lik") == "iki yüz elli TL'lik"  # the suffix needs "liralık": left alone


def test_percent_after_the_number():
    assert expand_digits("15% arttı") == "yüzde on beş arttı"
    assert expand_digits("%3.5") == "yüzde üç virgül beş"


def test_clock_with_a_dot_needs_context():
    assert expand_digits("saat 14.45'te") == "saat on dört kırk beş'te"
    assert expand_digits("08.30'da açılıyor") == "sekiz otuz'da açılıyor"
    assert expand_digits("Saat 14.45.") == "Saat on dört kırk beş."
    assert expand_digits("oran 14.45 oldu") == "oran on dört virgül kırk beş oldu"
    assert expand_digits("25:00") == "yirmi beş:sıfır sıfır"


def test_phone_numbers():
    spoken = "sıfır sekiz yüz elli, iki yüz yirmi iki, yirmi üç, otuz beş"
    for written in ("0850 222 23 35", "0850-222-23-35", "0(850) 222 23 35", "08502222335"):
        assert expand_digits(written) == spoken, written
    assert expand_digits("0532 000 00 01") == "sıfır beş yüz otuz iki, sıfır sıfır sıfır, sıfır sıfır, sıfır bir"
    assert expand_digits("+90 532 123 45 67") == "artı doksan, beş yüz otuz iki, yüz yirmi üç, kırk beş, altmış yedi"
    assert expand_digits("444 0 333") == "dört yüz kırk dört sıfır üç yüz otuz üç"


def test_card_numbers_and_ibans():
    assert expand_digits("4543 1234 5678 9010.") == \
        "dört beş dört üç, bir iki üç dört, beş altı yedi sekiz, dokuz sıfır bir sıfır."
    iban = "TR bir iki, sıfır sıfır sıfır altı, dört sıfır sıfır sıfır, sıfır sıfır bir bir, iki üç dört beş, " \
           "altı yedi sekiz dokuz, sıfır bir"
    assert expand_digits("TR12 0006 4000 0011 2345 6789 01") == iban
    assert expand_digits("TR120006400000112345678901") == iban
    assert expand_digits("1 2 3 4") == "bir iki üç dört"


def test_ordinals():
    assert spell_ordinal(3) == "üçüncü" and spell_ordinal(12) == "on ikinci" and spell_ordinal(40) == "kırkıncı"
    assert expand_digits("3. taksiti") == "üçüncü taksiti"
    assert expand_digits("3'üncü kat") == "üçüncü kat"
    assert expand_digits("Tutar 250. Sonra arayın.") == "Tutar iki yüz elli. Sonra arayın."
    assert expand_digits("ayın 5'inde") == "ayın beş'inde"


def test_digits_glued_to_letters():
    assert expand_digits("A4 kağıt") == "A dört kağıt"
    assert expand_digits("15'i") == "on beş'i"


def test_normalize_only_emits_characters_the_model_knows():
    import json
    import os

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "freyatts", "char_vocab.json")
    vocab = json.load(open(path, encoding="utf-8"))
    for text in ("Teşekkür ederiz, iyi günler!", "Kartınız (sonu 4821) kapatıldı.", "Lütfen “onaylıyorum” deyin.",
                 "3–5 iş günü", "Aylık ücret 49,90₺."):
        assert all(ch in vocab for ch in normalize(text)), normalize(text)
    assert normalize("Kartınız (sonu 4821) kapatıldı.") == "Kartınız, sonu dört bin sekiz yüz yirmi bir, kapatıldı."
    assert normalize("Merhaba! Ben asistanım.") == "Merhaba. Ben asistanım."
