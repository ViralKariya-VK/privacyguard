"""
Small rule layer that sits on TOP of the transformer predictions ("hybrid" model).
Found these problems when testing the demo by hand:
  1. the model sometimes tags only a piece of a word  (admin_[USERNAME]n)
  2. human-style passwords ("Kiran@2024!") are missed, because in the training data
     passwords are random strings like r]iD1#8
So:
  - snap spans to whole words (letters/digits/underscore)
  - a regex for "password: xyz" / "password is xyz" context
First version of the password rule fired on "password reset" and ate XML tags,
so now it needs a separator AND the value must contain a digit or symbol.
"""
import re

# keyword + (word like "is"/"tried" or a ':' / '=') + the value
PASSWORD_CTX = re.compile(
    r"(?i)\b(?:password|passwd|pwd|passcode)\b\s*"
    r"(?:(?:is|was|tried|used)\b\s*[:=]?|[:=])\s*[\"']?([^\s\"'<]+)")
# profile links: the part after the slash is a username (github.com/Ahuja-Vanshika)
PROFILE_URL = re.compile(
    r"(?i)\b(?:github\.com|linkedin\.com/in|twitter\.com|x\.com|instagram\.com)/([\w.\-]+)")
# quick hack: tool names that the model sometimes calls a NAME (found on a resume: "Pandas")
TECH_TERMS = {"pandas", "plotly", "python", "sql", "tableau", "tensorflow", "keras", "numpy",
              "scikit-learn", "pytorch", "git", "github", "excel", "docker", "spark", "flask",
              "django", "selenium", "opencv", "matlab", "java", "linux", "word", "powerpoint"}
# ID numbers with a fixed format (the training data has few Indian ids, so the model misses part of them)
ID_PATTERNS = [
    re.compile(r"(?<![\d-])\d{4}[ -]\d{4}[ -]\d{4}(?![\d-])"),   # aadhaar
    re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),                       # PAN
    re.compile(r"(?<![\d-])\d{3}-\d{2}-\d{4}(?![\d-])"),          # US ssn
]
ID_WORDS = {"aadhaar", "aadhar", "pan", "ssn", "passport", "uidai"}   # keyword before an id, never a name
TRAILING = ".,;:)]}"             # punctuation at the end that is probably not part of the password


# ------------------------------------------------------------------ format / context rules
# Found with eval_suite/: the model alone misses short emails, 10-digit phones, usernames,
# ISO dates, street addresses and plain names. These patterns are safe because they need a fixed
# format or a keyword right before the value. Each rule is (label, regex, group, replace_labels):
# the span is added and it replaces overlapping model spans of the labels in `replace_labels`
# (None = any label).
EMAIL_RE = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
OBFUSCATED_EMAIL_RE = re.compile(
    r"(?i)\b[\w.+-]+(?:\s+dot\s+[\w+-]+)*\s*(?:\[at\]|\(at\)|\bat\b)\s*[\w-]+(?:\s*(?:\[dot\]|\(dot\)|\bdot\b)\s*[\w-]+)*"
    r"\s*(?:\[dot\]|\(dot\)|\bdot\b)\s*(?:com|org|net|in|io|co|edu|gov)\b")
PHONE_RES = [
    re.compile(r"(?<![\w.])\+\d{1,3}[\s-]?\(?\d{1,5}\)?(?:[\s.-]?\d{2,5}){1,4}(?![\w])"),      # +91 98765 43210, +44 20 7946 0958
    re.compile(r"(?<![\w.])\(\d{3}\)\s?\d{3}[\s.-]\d{4}(?!\d)"),                                # (415) 555-0132
    re.compile(r"(?<![\w.])\d{3}[\s.-]\d{3}[\s.-]\d{4}(?!\d)"),                                  # 415-555-0199 (also looks like an id: gap filler only)
    re.compile(r"(?<!\d)[6-9]\d{4}\s?\d{5}(?!\d)"),                                              # Indian mobile 98765 43210 / 9876543210
    re.compile(r"(?<!\d)0\d{3,4}\s\d{3}\s?\d{3,4}(?!\d)"),                                       # UK 07700 900123
]
DL_RE = re.compile(r"\b[A-Z]{2}\d{2}\s?\d{11}\b")
ISO_DATE_RE = re.compile(r"(?<![\d-])(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])(?![\d-])")
DMY_DATE_RE = re.compile(r"(?<![\d/.-])(?:0?[1-9]|[12]\d|3[01])[/.-](?:0?[1-9]|1[0-2])[/.-](?:19|20)\d{2}(?![\d/.-])")
# username: keyword + value that "looks like" a username (has _ . digit) or comes after = or :
USER_CTX = re.compile(
    r"(?i)\b(?:user(?:name|\s*id)?|login(?:\s*id)?|uid|usr|account)\b\s*(?:[:=]\s*|\s+)([A-Za-z][\w.\-]{2,30})")
CHAT_USER = re.compile(r"(?m)(?:^|\])\s*([A-Za-z][\w.]{1,20}):(?=\s)")
# street address: "221B Baker Street, Springfield, IL 62704", "Flat 402, Sunrise Apartments, Andheri West, Mumbai 400058"
_CAP = r"[A-Z][\w'-]*"
_H = r"[ \t]+"          # same line only: an address never continues on the next line
ADDRESS_RE = re.compile(
    r"\b(?:\d{1,5}[A-Z]?" + _H + _CAP + r"(?:" + _H + _CAP + r"){0,2}" + _H + r"(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Drive|Dr|Boulevard|Blvd|Marg|Nagar|Way|Place)\b"
    r"|(?:Flat|Apt|Apartment|Unit|House|Plot)[ \t]*(?:No\.?[ \t]*)?\w+)"
    r"(?:,[ \t]*" + _CAP + r"(?:" + _H + r"[A-Z0-9][\w'-]*){0,2}){1,3}")
# a capitalised name right after a phrase that introduces a person
NAME_TOKEN = r"[A-Z](?:['’][A-Z])?[a-zÀ-ſ]+(?:-[A-Z][a-zÀ-ſ]+)*"
NAME_TRIGGERS = ["my name is", "this is", "i am", "i'm", "regards", "thanks", "thank you", "sincerely", "best",
                 "dear", "hello", "hi", "hey", "ask", "employee", "customer", "patient", "requester",
                 "attn", "mr", "mrs", "ms", "dr", "emergency contact"]


def build_name_ctx(triggers):
    return re.compile(
        r"(?<![A-Za-z])(?:(?i:" + "|".join(re.escape(t) for t in triggers) + r")\.?[:,]?\s+)"
        r"(" + NAME_TOKEN + r"(?:\s+" + NAME_TOKEN + r"){0,2})(?![\w@])")


NAME_CTX = build_name_ctx(NAME_TRIGGERS)
# resume / signature header: "Kavya Menon | kavya@mail.com | 99887 76655" (name at line start, contact data after it)
NAME_HEADER = re.compile(r"(?m)^[ \t]*(" + NAME_TOKEN + r"(?:\s+" + NAME_TOKEN + r"){1,2})[ \t]*[|,–-][ \t]*(?:\S+@|\+?\d|\(\d)")
# sign-off typed in lower case: "thanks, regards sarah connor"
LOWER_SIGNOFF = re.compile(r"(?m)\b(?:regards|sincerely|cheers|thanks|my name is)[,:]?[ \t]+([a-z]{2,}[ \t]+[a-z]{2,})[ \t]*$")
LOWER_STOP = {"a", "an", "the", "for", "so", "much", "again", "lot", "you", "all", "in", "advance", "guys", "team",
              "everyone", "very", "to", "and", "your", "help", "it", "that", "this", "now", "then", "soon", "sir"}
IPV4_RE = re.compile(r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\d.])")
USER_AT_IP = re.compile(r"(?<![\w.])([A-Za-z][\w.-]*)@(?=\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})")
NOT_NAMES = {"the", "a", "an", "team", "all", "everyone", "there", "sir", "madam", "fine", "sorry", "sure", "ok",
             "okay", "good", "great", "happy", "glad", "here", "not", "just", "writing", "going", "please",
             "thanks", "regards", "support", "hr", "admin", "manager", "customer", "user", "from", "to"}
# words that follow "Dear / Contact / Regards" but are groups, roles or organisations (found on the benchmark:
# "Dear Team", "Contact Person A", "Best regards, Great Britain University Division", "Name: Not Applicable")
NOT_NAME_WORDS = {"team", "person", "participants", "participant", "applicable", "service", "services", "support",
                  "department", "division", "university", "college", "school", "company", "group", "inc", "ltd",
                  "corp", "applicants", "applicant", "everyone", "committee", "board", "staff", "members", "member",
                  "colleagues", "guests", "students", "parents", "friends", "folks", "sir", "madam", "customer",
                  "user", "admin", "manager", "hr", "number", "details", "information", "handbook", "name",
                  "available", "required", "unknown", "none", "anonymous", "virtual", "reality", "great", "britain",
                  "institute", "office", "center", "centre", "organization", "organisation", "sales", "billing"}
ID_FILTER = True
PREFIX_NOT_ID = re.compile(r"(?i)(?:sku|order|invoice|item|ref|ticket)\W*$")


def _put(out, s, e, label, replace=None, swallow=False, keep_larger=False):
    """add a rule span; drop overlapping model spans (only those with a label in `replace`, None = any).
    swallow=True: also drop overlapping spans that are only fragments inside the new span (the model often cuts a
    phone number into pieces with different labels), unless one of them already covers (almost) all of it."""
    if swallow:
        inside = [sp for sp in out if sp[0] < e and s < sp[1] and s <= sp[0] and sp[1] <= e]
        if inside and all((sp[1] - sp[0]) < 0.9 * (e - s) for sp in inside):
            out = [sp for sp in out if sp not in inside]
    if keep_larger and any(sp[2] == label and sp[0] <= s and e <= sp[1] for sp in out):
        return out      # a weaker pattern must not shrink a span the model already found (+91 98765 43210 -> 98765 43210)
    keep = [sp for sp in out if not (sp[0] < e and s < sp[1] and (replace is None or sp[2] in replace))]
    if any(sp[0] < e and s < sp[1] for sp in keep):    # an overlapping span we may not replace -> rule loses
        return out
    keep.append((s, e, label))
    return keep


def _plain_number(text, sp):
    """a short number or a number after sku / order / invoice: a product or order number, not a person's id"""
    value = text[sp[0]:sp[1]]
    if value.replace(" ", "").isdigit() and len(value.replace(" ", "")) <= 5:
        return True
    return bool(PREFIX_NOT_ID.search(text[max(0, sp[0] - 10):sp[0]]) or re.match(r"(?i)(?:sku|order|invoice|item|ref|ticket)\b", value))


def apply_rules(text, out):
    for m in IPV4_RE.finditer(text):
        out = _put(out, m.start(), m.end(), "IP_ADDRESS", {"IP_ADDRESS", "EMAIL", "PHONE", "DATE", "ID_NUMBER", "LOCATION"})
    for m in USER_AT_IP.finditer(text):
        out = _put(out, m.start(1), m.end(1), "USERNAME", {"USERNAME", "EMAIL", "NAME", "LOCATION"})
    for m in EMAIL_RE.finditer(text):
        if IPV4_RE.fullmatch(m.group().split("@", 1)[1]):
            continue
        out = _put(out, m.start(), m.end(), "EMAIL")
    for m in OBFUSCATED_EMAIL_RE.finditer(text):
        out = _put(out, m.start(), m.end(), "EMAIL", {"NAME", "LOCATION", "USERNAME", "DATE"})
    for k, pat in enumerate(PHONE_RES):
        # the "+country code" format is unambiguous; the local formats must not override another label of the model
        replace = {"PHONE", "ID_NUMBER", "LOCATION", "DATE", "NAME"} if k < 2 else {"PHONE"}
        for m in pat.finditer(text):
            if sum(c.isdigit() for c in m.group()) >= 9:
                out = _put(out, m.start(), m.end(), "PHONE", replace, swallow=True, keep_larger=k >= 2)
    for m in DL_RE.finditer(text):
        out = _put(out, m.start(), m.end(), "ID_NUMBER", {"ID_NUMBER", "PHONE", "LOCATION", "NAME"})
    for pat in (DMY_DATE_RE,):     # ISO dates are not forced: the benchmark does not treat them as PII; birth dates are caught in mark_birth_dates
        for m in pat.finditer(text):
            out = _put(out, m.start(), m.end(), "DATE", {"DATE"})
    for m in USER_CTX.finditer(text):
        v = m.group(1)
        if any(c in v for c in "_.0123456789") or text[m.start(1) - 1] in "=:" or text[m.start(1) - 2:m.start(1)].strip() in ("=", ":"):
            out = _put(out, m.start(1), m.end(1), "USERNAME", {"USERNAME", "NAME", "PASSWORD", "LOCATION"})
    for m in CHAT_USER.finditer(text):
        v = m.group(1)
        at_bracket = m.start() < len(text) and text[m.start():m.start(1)].lstrip().startswith("]") or text[m.start()] == "]"
        if at_bracket or "_" in v:
            out = _put(out, m.start(1), m.end(1), "USERNAME", {"USERNAME", "NAME"})
    for m in ADDRESS_RE.finditer(text):
        out = _put(out, m.start(), m.end(), "LOCATION", {"LOCATION", "NAME", "DATE", "ID_NUMBER"})
    for m in NAME_CTX.finditer(text):
        words = m.group(1).split()
        if words[0].lower() in NOT_NAMES or words[0].lower() in TECH_TERMS:
            continue
        if any(w.lower() in NOT_NAME_WORDS for w in words):
            continue
        out = _put(out, m.start(1), m.end(1), "NAME", {"NAME", "LOCATION", "USERNAME"})
    for m in NAME_HEADER.finditer(text):
        if m.group(1).split()[0].lower() not in NOT_NAMES | TECH_TERMS and not any(
                w.lower() in NOT_NAME_WORDS for w in m.group(1).split()):
            out = _put(out, m.start(1), m.end(1), "NAME", {"NAME", "LOCATION", "USERNAME"})
    for m in LOWER_SIGNOFF.finditer(text):
        if not set(m.group(1).split()) & LOWER_STOP:
            out = _put(out, m.start(1), m.end(1), "NAME", {"NAME", "LOCATION", "USERNAME"})
    # short plain numbers after sku / order / invoice words are not ids
    if ID_FILTER:
        out = [sp for sp in out if not (sp[2] == "ID_NUMBER" and _plain_number(text, sp) and not any(
            p.search(text[sp[0]:sp[1]]) for p in ID_PATTERNS))]
    return out


def _is_word_char(c):
    return c.isalnum() or c == "_"


def _snap(text, s, e, label):
    """grow a span left/right until it does not cut a word in half"""
    if label == "NAME":                    # O'Brien, D'Souza: an apostrophe between letters belongs to the name
        if text[s] in "'\u2019":          # the model tagged only the apostrophe
            while s > 0 and text[s - 1].isalpha():
                s -= 1
        if text[e - 1] in "'\u2019":
            while e < len(text) and text[e].isalpha():
                e += 1
        while s > 1 and text[s - 1] in "'\u2019" and text[s - 2].isalpha():
            s -= 2
        while e < len(text) - 1 and text[e] in "'\u2019" and text[e + 1].isalpha() and text[e - 1].isalpha():
            e += 2
    if label in ("NAME", "LOCATION", "DATE", "USERNAME", "ID_NUMBER"):
        while s > 0 and _is_word_char(text[s - 1]) and _is_word_char(text[s]):
            s -= 1
        while e < len(text) and _is_word_char(text[e]) and _is_word_char(text[e - 1]):
            e += 1
    return s, e


IP_CHARS = set("0123456789abcdefABCDEF.:")


def _snap_ip(text, s, e):
    while s > 0 and text[s - 1] in IP_CHARS:
        s -= 1
    while e < len(text) and text[e] in IP_CHARS:
        e += 1
    while e > s and text[e - 1] in ".:":        # sentence punctuation after the ip
        e -= 1
    return s, e


def refine_spans(text, spans):
    out = []
    for s, e, lab in spans:
        s, e = _snap_ip(text, s, e) if lab == "IP_ADDRESS" else _snap(text, s, e, lab)
        out.append((s, e, lab))
    # (after snapping, otherwise "Pan" is not yet "Pandas")
    out = [sp for sp in out if not (sp[2] == "NAME" and text[sp[0]:sp[1]].lower() in TECH_TERMS)
           and text[sp[0]:sp[1]].lower() not in ID_WORDS]      # "Aadhaar" is the keyword, the number is the PII
    out = sorted(set(out))

    rule_spans = []
    for m in PASSWORD_CTX.finditer(text):
        s, e = m.start(1), m.end(1)
        while e > s and text[e - 1] in TRAILING:
            e -= 1
        value = text[s:e]
        # real passwords almost always have a digit or a symbol, plain words like "reset" don't
        if len(value) >= 4 and any(not c.isalpha() for c in value):
            rule_spans.append((s, e, "PASSWORD"))

    out = apply_rules(text, out)
    for pat in ID_PATTERNS:
        for m in pat.finditer(text):
            out = [sp for sp in out if not (sp[0] < m.end() and m.start() < sp[1])]
            out.append((m.start(), m.end(), "ID_NUMBER"))

    for m in PROFILE_URL.finditer(text):
        e = m.end(1)
        while e > m.start(1) and text[e - 1] in ".-":
            e -= 1
        out = [sp for sp in out if not (sp[0] < e and m.start() < sp[1])]   # also drops "github.com" tagged as EMAIL
        out.append((m.start(1), e, "USERNAME"))

    # rule spans replace model spans they overlap with, EXCEPT when the model already found
    # (almost) the same PASSWORD span -> its boundaries are better on the dataset's random passwords.
    # If the model only found a fragment of it (e.g. "Kiran" of "Kiran@2024!") the rule wins.
    for rs, re_, lab in rule_spans:
        covered = max([min(re_, sp[1]) - max(rs, sp[0]) for sp in out if sp[2] == "PASSWORD"] + [0])
        if covered >= 0.8 * (re_ - rs):
            continue
        out = [sp for sp in out if not (sp[0] < re_ and rs < sp[1])]
        out.append((rs, re_, lab))
    return sorted(out)


def uppercase_lines(text, max_words=4):
    """
    The model was trained mostly on 'Title Case' names, so a heading like 'VANSHIKA AHUJA'
    is not recognised. Returns (offset, line.title()) for short ALL-CAPS lines so the caller can
    run the model on the title-cased version too (same length -> offsets stay valid).
    Long caps lines (like a footer with 11 words) are skipped to avoid turning acronyms into names.
    """
    out = []
    pos = 0
    for line in text.split("\n"):
        words = line.split()
        # only letters (no digits/symbols): "CGPA - 9.59 /10" must not be treated as a heading
        if (line.isupper() and 2 <= len(words) <= max_words and all(w.isalpha() for w in words)
                and len(line.title()) == len(line)):
            out.append((pos, line.title()))
        pos += len(line) + 1
    return out


def lowercase_lines(text, max_words=8):
    """Same trick for text typed in lower case ("thanks, regards sarah connor"): the model wants Title Case.
    Returns (offset, line.title()) for short lines without any capital letter."""
    out = []
    pos = 0
    for line in text.split("\n"):
        words = line.split()
        if (2 <= len(words) <= max_words and not any(c.isupper() for c in line)
                and any(c.isalpha() for c in line) and len(line.title()) == len(line)):
            out.append((pos, line.title()))
        pos += len(line) + 1
    return out


# ------------------------------------------------------------------ date of birth (app level)
# A date is only private when it is a birth date. We don't change the model's labels (the benchmark
# only knows DATE), this is applied afterwards in the app / image pipeline.
BIRTH_WORDS = re.compile(r"(?i)\b(?:birth\s*date|birthday|date\s+of\s+birth|born|dob|d\.o\.b\.?)")
BIRTH_DATE = re.compile(
    r"(?i)\b(?:birth\s*date|birthday|date\s+of\s+birth|born(?:\s+on)?|dob|d\.o\.b\.?)[^\d\w]{0,6}"
    r"(?:(?:is|was|on)\b[^\d\w]{0,4})?"
    r"(\d{4}-\d{2}-\d{2}|\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}"
    r"|\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]{3,9},?\s+\d{4}"
    r"|[A-Za-z]{3,9}\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})")


def mark_birth_dates(text, spans):
    """DATE spans that follow 'birth / born / DOB' become DOB; birth dates the model missed are added."""
    out = []
    for s, e, lab in spans:
        if lab == "DATE" and BIRTH_WORDS.search(text[max(0, s - 25):s]):
            lab = "DOB"
        out.append((s, e, lab))
    for m in BIRTH_DATE.finditer(text):
        s, e = m.start(1), m.end(1)
        out = [sp for sp in out if not (sp[0] < e and s < sp[1])]
        out.append((s, e, "DOB"))
    return sorted(out)
