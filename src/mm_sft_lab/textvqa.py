"""TextVQA answer normalization and leave-one-annotator-out VQA accuracy.

Normalization rules and contraction vocabulary follow Meta MMF's
mmf/utils/m4c_evaluators.py (BSD; attribution in MMF-LICENSE.txt).
"""

import re
from collections import Counter

# Preserve upstream exceptions, including keys whose capitalization is significant.
_CONTRACTION_PAIRS = """
aint=ain't arent=aren't cant=can't couldve=could've couldnt=couldn't
couldn'tve=couldn't've couldnt've=couldn't've didnt=didn't doesnt=doesn't dont=don't
hadnt=hadn't hadnt've=hadn't've hadn'tve=hadn't've hasnt=hasn't havent=haven't
hed=he'd hed've=he'd've he'dve=he'd've hes=he's howd=how'd howll=how'll hows=how's
Id've=I'd've I'dve=I'd've Im=I'm Ive=I've isnt=isn't itd=it'd itd've=it'd've
it'dve=it'd've itll=it'll let's=let's maam=ma'am mightnt=mightn't
mightnt've=mightn't've mightn'tve=mightn't've mightve=might've mustnt=mustn't
mustve=must've neednt=needn't notve=not've oclock=o'clock oughtnt=oughtn't
ow's'at='ow's'at 'ows'at='ow's'at 'ow'sat='ow's'at shant=shan't
shed've=she'd've she'dve=she'd've she's=she's shouldve=should've shouldnt=shouldn't
shouldnt've=shouldn't've shouldn'tve=shouldn't've somebody'd=somebodyd
somebodyd've=somebody'd've somebody'dve=somebody'd've somebodyll=somebody'll
somebodys=somebody's someoned=someone'd someoned've=someone'd've
someone'dve=someone'd've someonell=someone'll someones=someone's
somethingd=something'd somethingd've=something'd've something'dve=something'd've
somethingll=something'll thats=that's thered=there'd thered've=there'd've
there'dve=there'd've therere=there're theres=there's theyd=they'd
theyd've=they'd've they'dve=they'd've theyll=they'll theyre=they're theyve=they've
twas='twas wasnt=wasn't wed've=we'd've we'dve=we'd've weve=we've werent=weren't
whatll=what'll whatre=what're whats=what's whatve=what've whens=when's
whered=where'd wheres=where's whereve=where've whod=who'd whod've=who'd've
who'dve=who'd've wholl=who'll whos=who's whove=who've whyll=why'll
whyre=why're whys=why's wont=won't wouldve=would've wouldnt=wouldn't
wouldnt've=wouldn't've wouldn'tve=wouldn't've yall=y'all yall'll=y'all'll
y'allll=y'all'll yall'd've=y'all'd've y'alld've=y'all'd've y'all'dve=y'all'd've
youd=you'd youd've=you'd've you'dve=you'd've youll=you'll youre=you're youve=you've
"""
CONTRACTIONS = dict(pair.split("=", 1) for pair in _CONTRACTION_PAIRS.split())
NUMBERS = {word: str(index) for index, word in enumerate("zero one two three four five six seven eight nine ten".split())}
NUMBERS["none"] = "0"
PUNCTUATION = ';/[]"{}()=+\\_-><@`,?!'


def normalize_answer(value):
    if not isinstance(value, str):
        raise ValueError("TextVQA answers must be strings")
    text = value.lower().replace(",", "").replace("?", "").replace("'s", " 's").strip()
    text = text.replace("\n", " ").replace("\t", " ").strip()
    original = text
    comma_between_digits = re.search(r"(?<=\d),+(?=\d)", original) is not None
    for char in PUNCTUATION:
        text = text.replace(char, "" if char + " " in original or " " + char in original or comma_between_digits else " ")
    # The reference strips periods except those followed by a digit.
    text = re.sub(r"(?!<=\d)(\.)(?!\d)", "", text)
    words = [NUMBERS.get(word, word) for word in text.lower().split() if word not in ("a", "an", "the")]
    return " ".join(CONTRACTIONS.get(word, word) for word in words)


def vqa_accuracy(prediction, answers):
    if not isinstance(answers, list) or len(answers) != 10 or not any(str(a).strip() for a in answers):
        raise ValueError("TextVQA scoring requires ten non-placeholder reference answers")
    normalized = [normalize_answer(answer) for answer in answers]
    target = normalize_answer(prediction)
    matches = normalized.count(target)
    return sum(min(1.0, (matches - int(answer == target)) / 3) for answer in normalized) / 10


def training_answer(answers):
    if not isinstance(answers, list) or len(answers) != 10 or any(not isinstance(a, str) for a in answers):
        raise ValueError("TextVQA training requires ten string annotations")
    # OCR may legitimately ask for "a", "the" or "a+", which the scoring
    # normalizer maps to "". Only raw blank annotations lack a training target.
    usable = [a.strip() for a in answers if a.strip()]
    if not usable:
        raise ValueError("TextVQA sample has no usable target")
    counts = Counter(normalize_answer(a) for a in usable)
    winning = min(counts, key=lambda key: (-counts[key], key))
    originals = Counter(a for a in usable if normalize_answer(a) == winning)
    return min(originals, key=lambda key: (-originals[key], key))
