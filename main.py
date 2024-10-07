#!/usr/bin/env python
"""

    Greynir: Natural language processing for Icelandic

    Generation of synthetic sentence pairs containing rare terms

    Copyright (C) 2021 Miðeind ehf.

    This software is licensed under the MIT License:

        Permission is hereby granted, free of charge, to any person
        obtaining a copy of this software and associated documentation
        files (the "Software"), to deal in the Software without restriction,
        including without limitation the rights to use, copy, modify, merge,
        publish, distribute, sublicense, and/or sell copies of the Software,
        and to permit persons to whom the Software is furnished to do so,
        subject to the following conditions:

        The above copyright notice and this permission notice shall be
        included in all copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
        EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
        MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
        IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
        CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
        TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
        SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


    This package facilitates the creation of corpora containig synthetic sentence
    pairs, in Icelandic and English, induced with a vocabulary of (usually
    rare) terms.

    The creation of such corpora proceeds in two main stages:

    1)  Processing a base corpus of authentic sentence pairs, accompanied by a
        glossary of fairly common Icelandic nouns with their English counterparts.
        This processing generates a parallel template corpus with placeholders
        for nouns on the Icelandic and on the English side.

    2)  Application of the template corpus to a glossary of rare terms,
        so that the terms are inserted into the templates, correctly inflected
        on both the Icelandic and the English side. This produces a
        final synthetic parallel corpus, induced with the vocabulary, that can
        be used as a part of the training input for a neural machine translation
        network.

"""

from __future__ import annotations

from typing import Tuple, Dict, List, FrozenSet, Iterator, Optional, DefaultDict, TextIO

import sys
import argparse
import random
import re
from collections import defaultdict

import inflect  # type: ignore

from tokenizer import TOK, detokenize
from reynir.reynir import Greynir, _Sentence as Sentence, Terminal
from reynir.bindb import BIN_Db


# Type definitions
GlossaryDict = Dict[str, List[Tuple[str, str]]]
TermDict = Dict[str, Tuple[str, str]]

g = Greynir()
db = BIN_Db()
p = inflect.engine()

PLACEHOLDER = "[*]"
MAX_LINES = 1000

# File types for UTF-8 encoded text files
ReadFile = argparse.FileType("r", encoding="utf-8")
WriteFile = argparse.FileType("w", encoding="utf-8")

# Define the command line arguments

parser = argparse.ArgumentParser(
    description="Generates synthetic sentence pairs induced with desired vocabulary"
)

parser.add_argument(
    "infile",
    nargs="?",
    type=ReadFile,
    default=sys.stdin,
    help="Input file (UTF-8 tab-separated sentence pairs)",
)
parser.add_argument(
    "outfile",
    nargs="?",
    type=WriteFile,
    default=sys.stdout,
    help="Output file (UTF-8 tab-separated sentence pairs)",
)

parser.add_argument(
    "--glossary", type=ReadFile, help="Glossary file (UTF-8 tab-separated terms)",
)

parser.add_argument(
    "--verbose", help="Verbose output", action="store_true",
)

parser.add_argument(
    "--count", type=int, help="Instances of each term to generate",
    default=10, action="store",
)

group = parser.add_mutually_exclusive_group()
group.add_argument(
    "--template", help="Create a template from a .tsv source file", action="store_true",
)
group.add_argument(
    "--generate",
    help="Generate a synthetic corpus from a .tsv template file",
    action="store_true",
)


class TermPair:

    """ A wrapper around a single rare Icelandic lemma and its English counterpart,
        having the ability to provide the lemma in various inflectional forms
        as described by a terminal in a parse tree. """

    CASE_VARIANTS: FrozenSet[str] = frozenset(("nf", "þf", "þgf", "ef"))
    NUMBER_VARIANTS: FrozenSet[str] = frozenset(("nf", "þf", "þgf", "ef"))

    def __init__(self, lemma: str, cat: str, en: Tuple[str, str]) -> None:
        # Initialize a rare lemma pair
        self._lemma = lemma
        self._cat = cat
        # Check if this is a composite word, e.g. 'rauð-dvergur'
        _, m = db.lookup_lemma(lemma)
        m = [mm for mm in m if mm.ordfl == cat]
        if len(m) != 1:
            raise ValueError(f"The lemma '{lemma}/{cat}' is ambiguous in BÍN")
        if "-" in m[0].stofn and "-" not in lemma:
            # Composite: note the prefix and use the suffix as the lemma
            self._prefix, self._lemma = m[0].stofn.rsplit("-", maxsplit=1)
        else:
            # Not composite
            self._prefix = ""
        self._en_singular, self._en_plural = en
        self._cache: Dict[FrozenSet[str], Tuple[str, str]] = dict()

    @property
    def gender(self) -> str:
        """ Returns the [Icelandic] gender of this term """
        return self._cat

    def inflect(self, variants: FrozenSet[str]) -> Optional[Tuple[str, str]]:
        """ Return the term pair, inflected to correspond to the variants given """
        form = self._cache.get(variants)
        if form is None:
            # Generate the inflection forms and cache them
            case = set(variants & self.CASE_VARIANTS).pop()
            for m in db.lookup_forms(self._lemma, self._cat, case):
                if "2" in m.beyging or "3" in m.beyging:
                    # Extra, idiosyncratic forms: skip
                    continue
                number = "ft" if "FT" in m.beyging else "et"
                vset = {case, number}
                if "gr" in m.beyging:
                    vset.add("gr")
                # Create the Icelandic form for this variant set
                is_form = self._prefix + m.ordmynd
                # Create the English form for this variant set
                if number == "ft":
                    en_form = self._en_plural
                else:
                    en_form = self._en_singular
                # Cache the variant -> form association
                self._cache[frozenset(vset)] = (is_form, en_form)
            form = self._cache.get(variants)
        return form


class TemplateCollection:

    """ A wrapper for a collection of template sentences that can be used
        to generate synthetic pairs containing rare terms. The templates
        are segregated by gender. """

    def __init__(self) -> None:
        self._templates: DefaultDict[str, List[Template]] = defaultdict(list)

    def append(self, t: Template) -> None:
        """ Add a template to the appropriate list, by gender """
        self._templates[t.gender].append(t)

    def read(self, infile: TextIO) -> None:
        """ Read a collection of templates from the infile into memory """
        for line in infile:
            line = line.strip()
            if not line:
                continue
            en_template, is_template = line.split("\t", maxsplit=1)
            t = Template.load(is_template, en_template)
            self.append(t)

    def generate(self, pair: TermPair, count: int = 1) -> Iterator[Tuple[str, str]]:
        """ Generate sentence pairs by substituting the term pair
            into randomly sampled templates of matching gender """
        templates = self._templates[pair.gender]
        if not templates:
            raise ValueError(f"No template available for the '{pair.gender}' gender")
        for template in random.sample(templates, k=count):
            sub = template.substitute(pair)
            if sub is not None:
                yield sub


class Template:

    """ A sentence pair template that can be used as a basis for substituting
        rare terms, generating new pairs. """

    INTERESTING_VARIANTS: FrozenSet[str] = frozenset(
        ("nf", "þf", "þgf", "ef", "et", "ft", "gr")
    )
    GENDER_VARIANTS: FrozenSet[str] = frozenset(("kk", "kvk", "hk"))
    CASE_VARIANTS: FrozenSet[str] = frozenset(("nf", "þf", "þgf", "ef"))
    NUMBER_VARIANTS: FrozenSet[str] = frozenset(("et", "ft"))

    # Class-wide defaults
    _variants: FrozenSet[str] = frozenset()
    _gender: str = ""
    _case: str = ""
    _number: str = ""
    _is_template: str = ""
    _en_template: str = ""
    _is_all_caps = False
    _is_first_cap = False
    _en_all_caps = False
    _en_first_cap = False

    def __init__(self) -> None:
        """ Initialize an empty Template instance. Template instances should be
            created using the create() or from_pair() classmethods. """
        pass

    @classmethod
    def create(
        cls,
        sent: Sentence,
        terminal: Terminal,
        en_sent: str,
        en_word: str,
        en_word_regex: str,
    ) -> Template:
        """ Create a Template from a parsed Sentence and its English counterpart """
        self = cls()
        # Save the grammatical variants associated with the noun on the Icelandic side
        vset: FrozenSet[str] = frozenset(terminal.variants)
        self._variants = vset & self.INTERESTING_VARIANTS
        # Note the gender of the Icelandic noun in this template
        self._gender = set(vset & self.GENDER_VARIANTS).pop()
        self._case = set(vset & self.CASE_VARIANTS).pop()
        self._number = set(vset & self.NUMBER_VARIANTS).pop()
        # Replace the token corresponding to the Icelandic noun's terminal
        # with a placeholder string, making it easy to do a simple string
        # replace when substituting a rare noun
        toklist = sent.tokens[:]
        txt: str = toklist[terminal.index].txt
        # Note the respective capitalizations on the Icelandic and the English side
        self._is_all_caps = is_all_caps = txt.isupper()
        self._is_first_cap = is_first_cap = txt[0].isupper()
        self._en_all_caps = en_all_caps = en_word.isupper()
        self._en_first_cap = en_first_cap = en_word[0].isupper()
        # Create an Icelandic-side placeholder of the form
        # gender_case_number_article
        is_placeholder = f"{self._gender}_{self._case}_{self._number}"
        if "gr" in vset:
            is_placeholder += "_gr"
        if is_all_caps:
            is_placeholder += "_caps"
        elif is_first_cap:
            is_placeholder += "_cap"
        # TODO: Encode the original casing of the noun
        toklist[terminal.index] = TOK.Word("{0:" + is_placeholder + "}")
        self._is_template = detokenize(toklist)
        # On the English side, simply replace the word by the placeholder string
        # (we've already ascertained that the word only occurs once)
        en_placeholder = "pl" if self._number == "ft" else "sg"
        if en_all_caps:
            en_placeholder += "_caps"
        elif en_first_cap:
            en_placeholder += "_cap"
        self._en_template = re.sub(
            en_word_regex,
            "{0:" + en_placeholder + "}",
            en_sent,
            count=1,
            flags=re.IGNORECASE,
        )
        assert self._en_template != en_sent
        return self

    @classmethod
    def load(cls, is_template: str, en_template: str) -> Template:
        """ Create a Template instance from a pair of strings """
        self = cls()
        vset: FrozenSet[str]
        # Retrieve the variants on the Icelandic side
        m = re.search(r"{([0-9]+):([^\}]+)}", is_template)
        if m is not None:
            g = m.group(1)
            assert g == "0"  # We currently only handle one placeholder per sentence
            g = m.group(2)
            v = g.split("_")
            vset = frozenset(v)
            self._variants = frozenset(vset & self.INTERESTING_VARIANTS)
            # Note the gender of the Icelandic noun in this template
            glist = list(vset & self.GENDER_VARIANTS)
            assert len(glist) == 1
            self._gender = glist[0]
            glist = list(vset & self.CASE_VARIANTS)
            assert len(glist) == 1
            self._case = glist[0]
            glist = list(vset & self.NUMBER_VARIANTS)
            assert len(glist) == 1
            self._number = glist[0]
            self._is_all_caps = "caps" in vset
            self._is_first_cap = "cap" in vset
            self._is_template = re.sub(r"{[^\}]+}", PLACEHOLDER, is_template)
        else:
            self._is_template = is_template
        # Retrieve the variants on the English side
        m = re.search(r"{([0-9]+):([^\}]+)}", en_template)
        if m is not None:
            g = m.group(1)
            assert g == "0"  # We currently only handle one placeholder per sentence
            g = m.group(2)
            v = g.split("_")
            vset = frozenset(v)
            self._en_all_caps = "caps" in vset
            self._en_first_cap = "cap" in vset
            self._en_template = re.sub(r"{[^\}]+}", PLACEHOLDER, en_template)
        else:
            self._en_template = en_template
        return self

    @property
    def gender(self) -> str:
        return self._gender

    def write(self, outfile: TextIO) -> None:
        """ Write the template as text with placeholders to a template file """
        outfile.write(f"{self._en_template}\t{self._is_template}\n")

    def substitute(self, pair: TermPair) -> Optional[Tuple[str, str]]:
        """ Generate a new sentence pair by substituting the term pair
            into the template """
        terms = pair.inflect(self._variants)
        if terms is None:
            # No such inflection is available (usually applies to nouns
            # that are only singular or only plural)
            return None
        is_term, en_term = terms
        # Emulate the capitalization of the original template
        if self._is_all_caps:
            is_term = is_term.upper()
        elif self._is_first_cap:
            is_term = is_term.capitalize()
        if self._en_all_caps:
            en_term = en_term.upper()
        elif self._en_first_cap:
            en_term = en_term.capitalize()
        # Perform the string replacement and return the result
        return (
            self._is_template.replace(PLACEHOLDER, is_term),
            self._en_template.replace(PLACEHOLDER, en_term),
        )


def english_plural(w: str) -> str:
    """ Return the plural form of an English word """
    return p.plural(w)  # type: ignore


class TemplateCollector:

    """ Collects templates from a base TSV file containing English
        and Icelandic sentence pairs, in reference to a glossary of terms
        that can be used as placeholders. """

    def __init__(
        self, infile: TextIO, outfile: TextIO, glossary: TextIO, verbose: bool = False
    ) -> None:
        self._infile = infile
        self._outfile = outfile
        self._glossary = self._read_glossary(glossary)
        self._verbose = verbose

    @staticmethod
    def _read_glossary(infile: TextIO) -> GlossaryDict:
        """ Read the glossary file (typically ./resources/glossary.txt)
            into a dictionary and return it """
        d: GlossaryDict = dict()
        for line in infile:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                # Comment
                continue
            g = line.split(",")
            if len(g) < 2:
                print(f"Malformed line in glossary file: '{line}'", file=sys.stderr)
                continue
            g = [s.strip() for s in g]
            lemma, cat = g[0].split("/")
            _, m = db.lookup_lemma(lemma)
            if all(mm.ordfl != cat for mm in m):
                print(
                    f"Could not find glossary entry '{lemma}/{cat}' in dictionary",
                    file=sys.stderr,
                )
                continue
            if len(g) == 2:
                en_lemmas = [g[1]]
            else:
                en_lemmas = g[1:]
            # Create lists of regexes for each lemma, singular and plural
            en_singular = [r"\b" + w + r"\b" for w in en_lemmas]
            en_plural = [r"\b" + english_plural(w) + r"\b" for w in en_lemmas]
            d[lemma] = list(zip(en_singular, en_plural))
        return d

    def run(self) -> None:
        """ Read a TSV input file with sentence pairs (en, is)
            and collect template sentences from it """
        cnt = 0
        for line in self._infile:
            line = line.strip()
            if not line:
                continue
            en_sent, is_sent = line.split("\t")
            # Attemt to parse the Icelandic side
            sent = g.parse_single(is_sent)
            if sent is None or sent.terminals is None:
                # Doesn't parse: we can't use this as a template
                continue
            # Find noun terminals in the sentence
            for terminal in sent.terminals:
                if terminal.category != "no":
                    continue
                noun = terminal.text
                if " " in noun:
                    # We don't try to use stuff like 'fjármála- og efnahagsráðherra'
                    continue
                # Found a potential noun
                lemma = terminal.lemma
                is_plural = "ft" in terminal.variants
                if self._verbose:
                    print(f"Noun: {noun}, lemma {lemma}")
                en_lemmas = self._glossary.get(lemma)
                if en_lemmas is None:
                    # Our dictionary doesn't contain at least one possible
                    # English correspondence to the Icelandic noun
                    continue

                count = 0
                target_word = ""
                target_word_regex = ""
                for en_singular, en_plural in en_lemmas:
                    if is_plural:
                        en_word_regex = en_plural
                    else:
                        en_word_regex = en_singular
                    matches = re.findall(en_word_regex, en_sent, re.IGNORECASE)
                    if not matches:
                        # This target is not found: continue search
                        continue
                    if len(matches) > 1:
                        # Target appears more than once: give up
                        break
                    count += 1
                    if count > 1:
                        # More than one, and different, targets: give up
                        break
                    # Use the matched word as the target, in its original case
                    target_word = matches[0]
                    target_word_regex = en_word_regex
                else:
                    # No problem found; exactly one meaning matches
                    if count == 1:
                        assert target_word
                        t = Template.create(
                            sent, terminal, en_sent, target_word, target_word_regex
                        )
                        t.write(self._outfile)

            cnt += 1
            if cnt >= MAX_LINES:
                break


def read_terms(infile: TextIO) -> TermDict:
    """ Read the terms file (typically ./resources/terms.txt)
        into a dictionary and return it """
    d: TermDict = dict()
    for line in infile:
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            # Comment
            continue
        # Format is Icelandic-term/category, English-singular [, English-plural]
        g = line.split(",")
        if len(g) < 2:
            print(f"Malformed line in terms file: '{line}'", file=sys.stderr)
            continue
        g = [s.strip() for s in g]
        lemma, cat = g[0].split("/")
        # The Icelandic term should include a gender (kk/kvk/hk)
        assert cat in Template.GENDER_VARIANTS
        if len(g) == 2:
            # Only the singular form of the noun is given: generate the plural
            en_singular, en_plural = g[1], english_plural(g[1])
        else:
            # Both singular and plural forms are given
            en_singular, en_plural = g[1], g[2]
        # Add to the dictionary of terms
        d[lemma + "/" + cat] = (en_singular, en_plural)
    return d


def generate_templates(
    infile: TextIO, outfile: TextIO, glossary: TextIO, verbose: bool
) -> None:
    """ From a base input file with English-Icelandic sentence pairs,
        and a glossary file, generate an output file containing templates
        with placeholders """
    tc = TemplateCollector(infile, outfile, glossary, verbose=verbose)
    tc.run()


def generate_pairs(
    infile: TextIO, outfile: TextIO, glossary: TextIO, count: int, verbose: bool
) -> None:
    """ Generate synthetic sentence pairs in the output file by substituting rare terms
        from the glossary into templates loaded from the input file """
    tc = TemplateCollection()
    tc.read(infile)
    terms = read_terms(glossary)
    for k, v in terms.items():
        lemma, cat = k.split("/")
        tp = TermPair(lemma, cat, v)
        for pair in tc.generate(tp, count):
            is_sent, en_sent = pair
            outfile.write(f"{en_sent}\t{is_sent}\n")


def main() -> int:
    """ Main program: parse the command line arguments
        and perform the desired actions """
    infile: Optional[TextIO] = None
    outfile: Optional[TextIO] = None
    glossary: Optional[TextIO] = None
    try:
        args = parser.parse_args()
        infile = args.infile
        if infile is None:
            print("Unable to read input file", file=sys.stderr)
            return 1
        outfile = args.outfile
        if outfile is None:
            print("Unable to write output file", file=sys.stderr)
            return 1
        glossary = args.glossary
        if glossary is None:
            print("Unable to read glossary file", file=sys.stderr)
            return 1
        if args.template:
            # Generate a template file
            generate_templates(infile, outfile, glossary, args.verbose)
        elif args.generate:
            # Generate a synthetic corpus file
            generate_pairs(infile, outfile, glossary, args.count, args.verbose)
        else:
            print("Please specify either --template or --generate", file=sys.stderr)
            return 1
    finally:
        if infile is not None:
            infile.close()
        if outfile is not None:
            outfile.close()
        if glossary is not None:
            glossary.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
