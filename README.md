
<img src="static/greynir-logo-large.png" alt="Greynir" width="200" height="200" align="right" style="margin-left:20px; margin-bottom: 20px;">

# GreynirTerms

*GreynirTerms* is a utility program for generating synthetic parallel corpora
of English and Icelandic sentences that contain rare terms. These rare terms
are typically specialized vocabularies or glossaries for particular topic domains.

GreynirTerms is used to help Neural Machine Translation (NMT) networks
learn vocabulary that occurs relatively infrequently in authentic parallel corpora.

GreynirTerms works in two distinct phases:

1) An authentic parallel corpus and a glossary of common terms are combined
   to generate a template file.

2) A template file and a dictionary of rare terms are combined to generate
   a synthetic parallel corpus file.

The synthetic parallel corpus can then be mixed into NMT training corpora to
ensure that the network sees and learns the rare terms.

## Generating a template file (phase 1)

In phase 1, to *generate a template file* by using the common terms as
placeholders:

```
python main.py authentic.tsv template.tsv --template --glossary=resources/glossary.txt
```

The first parameter is the input (authentic corpus) file; if left out, `stdin` is read.

The second parameter is the output (template) file; if left out, the output is written to `stdout`.

The `--glossary` parameter identifies a glossary file containing translations
of common nouns. Its format is explained below.

## Generating a synthetic corpus (phase 2)

In phase 2, to *generate a synthetic parallel corpus* containing the rare terms:

```
python main.py template.tsv synthetic.tsv --generate --glossary=resources/terms.txt --count=50
```

The first parameter is the input (template) file generated in phase 1; if left out, `stdin` is read.

The second parameter is the output (synthetic corpus) file; if left out, the output is written to `stdout`.

The `--glossary` parameter identifies a terms file containing the rare terms
(nouns) that should occur in the synthetic corpus, as well as their translations.

The `--count` parameter specifies the number of (randomly sampled) synthetic pairs
that should be generated for each rare term. Since the grammatical forms of
Icelandic nouns can be
up to 16 (4 cases * 2 numbers * 2 definite/indefinite forms), it is advisable to
generate enough examples so that the synthetic corpus contains most or all of
those forms. The default value of `--count` is 10, which is enough for testing
purposes but probably not enough for real use cases.

## File formats

Authentic and synthetic corpus files, as well as template files, are tab-separated UTF-8
text files. Each line contains an English sentence, a tab character, and an Icelandic
sentence, terminated by a newline character.

The common noun glossary file has the following format:

```
icelandic-noun-lemma/category, english-lemma [, alternative-english-lemma]*
```

Example:

```
rannsókn/kvk, investigation, study
```

This entry causes all forms of the lemma _rannsókn_, along with the corresponding
English form (_investigation/investigations/study/studies_), to be eligible as
placeholders in templates, for feminine nouns.

The rare term glossary file has the following format:

```
icelandic-noun-lemma/category, english-lemma-singular [, english-lemma-plural]
```

Example:

```
rauðdvergur/kk, red dwarf, red dwarves
```

If the English lemma is given only in the singular, a plural form is automatically
generated via heuristics. These are pretty good so an explicit plural should only
be required in very irregular cases.

The GreynirTerms software is careful about maintaining correspondence between
genders, cases, numbers, determiners and casing (upper/lower case) within the
sentence pairs in the generated corpus file. The result should thus remain
grammatically correct (as long as the authentic source is correct) on both
the Icelandic and the English side. However, the meaning may of course become
more or less nonsensical as nouns are replaced with different ones.

## Copyright and licensing

GreynirTerms is Copyright (C) 2021 [Miðeind ehf.](https://mideind.is)

Parts of this software are developed under the auspices of the
Icelandic Government's 5-year Language Technology Programme for Icelandic,
which is described
[here](https://www.stjornarradid.is/lisalib/getfile.aspx?itemid=56f6368e-54f0-11e7-941a-005056bc530c>)
(English version [here](<https://clarin.is/media/uploads/mlt-en.pdf>)).

This software is licensed under the **MIT License**:

   *Permission is hereby granted, free of charge, to any person
   obtaining a copy of this software and associated documentation
   files (the "Software"), to deal in the Software without restriction,
   including without limitation the rights to use, copy, modify, merge,
   publish, distribute, sublicense, and/or sell copies of the Software,
   and to permit persons to whom the Software is furnished to do so,
   subject to the following conditions:*

   *The above copyright notice and this permission notice shall be
   included in all copies or substantial portions of the Software.*

   *THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
   IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
   CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
   TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
   SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.*

<img src="https://github.com/mideind/GreynirPackage/blob/master/doc/_static/MideindLogoVert100.png?raw=true" align="right" style="margin-left:20px;" alt="Miðeind ehf.">

If you would like to use this software in ways that are incompatible
with the standard MIT license, [contact Miðeind ehf.](mailto:mideind@mideind.is)
to negotiate custom arrangements.
