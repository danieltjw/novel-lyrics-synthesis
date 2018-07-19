# Novel Lyrics Synthesis

In this project, Recurrent Neural Networks (RNNs) are applied to sequence modeling and Natural Language Processing (NLP) tasks. Character-level language models were trained on 100 song lyrics and then used to generate new lyrics. The quality of the generated lyrics were evaluated using 3 metrics—ability to form valid words, emulate the original sentence structure (frequency distribution of sentence length) and similarity (BLEU score).

## Summary of findings

- The latest CuDNN accelerated GRUs brought about a 7.3x / 6.7x speed up in training time compared to non-CuDNN GRU implementation 1 / 2 respectively.
- Perplexity has a moderate negative correlation (r=-0.632, DF=15; P<0.01) with the valid words percentage (1st metric). This indicates that the 1st metric measures some variability that is independent from perplexity. 
- Lower perplexity does not guarantee a better language model—at least for the task of forming valid words (1st metric).
- GRU networks with shorter sequence length (10, 20) are not able to properly emulate the sentence length of the original corpus. These networks' sentences have a significantly higher standard deviation (23.7, 16.6) compared to the original corpus (11.8). The mean sentence length of across different models did not vary greatly. This shows that while shorter models can adequately learn the frequency of newline chars, they struggle with emulating the structure (frequency distribution) of the original lyrics.

## Results

![](docs/hyper-parameter_search_results.png)

_Note: GRU network sequence length: 50, Batch size: 128_

---

![](docs/heatmap_valid_word_per.png)

_Note: GRU network sequence length: 50, Batch size: 128_

## Metrics

### __1. Valid Words percentage__

T*h*is m*e*tric eva*l*uates the model’s ability to generate non-gibberish words. The *p*ercentage of valid words over all words generated is calculated. Words are defined as strings of characters delimited by space or newline characters.

[SCOWL (Spell Checker Oriented Word Lists)](https://github.com/en-wl/wordlist) was found to be the best dictionary word list for the task. Comparatively, the [NLTK (Natural Language Toolkit) word list](https://www.nltk.org/book/ch02.html#wordlist-corpora) was found to be inadequate as only 72% of the lyrics’ words were found within it, compared to the 97% in the SCOWL.

A reduced SCOWL dictionary (scowl size: 50) was used instead of the default (scowl size: 60) as it was found to contain less unsuitable words. For example, the word ‘ot’ was found in the latter but not in the former.

Furthermore, words that were 1-2 letters long (412 in total) were manually screened so as to remove words that are not common in lyrics. This helped reduce the number of false positives that might have resulted in an overly exaggerated score.

`Dictset (Valid Words) = Words in lyrics + Words in SCOWL`

### __2. Sentence Length__

The sentence length metr*i*cs measures the ability of the model to emulate the sentence length of the original corpus. The sentence length is based on the distance between newlines (*‘*\n’) characters, inclusive of the last ending newline. Sentences of length 0, such as those used in paragraph breaks (‘\\*n*\\*n*’), are not considered.

### __3. Sentence BLEU__

The [sentence BLEU (BiLingual Evaluation Understudy) score](http://www.nltk.org/api/nltk.translate.html#nltk.translate.bleu_score.sentence_bleu) implementation in NLTK (N*a*tural *L*anguage ToolK*i*t) will be used as an e*v*aluation m*e*tric on how similar the generated sentences are compared to the existing lyrics. BLEU score has been a mainstay in the assessment of machine translation tasks ([Papineni, Roukos, & Ward, 2002](https://www.aclweb.org/anthology/P02-1040.pdf)). A preferred score would be one in the goldilocks zone—not too high which would indicate it being too similar and boring, but also not to low which may be too unfamiliar. 

As sentence-level BLEU score will be used instead of corpus-level one, a [smoothing function](https://www.nltk.org/api/nltk.translate.html#nltk.translate.bleu_score.SmoothingFunction.method3) ([Chen & Cherry, 2014](http://acl2014.org/acl2014/W14-33/pdf/W14-3346.pdf)) was added to address null n-gram count.

## Data pre-processing

The lyrics included in the corpus should reflect an artist's style. These steps were taken to decide which lyrics would make up the corpus. Personal judgment was used in the last step.

1. Remove cover songs
2. Remove songs where artist has no writing credits
3. Remove songs not in official releases (Albums, EPs, Singles)
4. Remove song collaborations which did not fit artist’s style

The 100 songs which would form the corpus were sourced from [https://genius.com](https://genius.com). As those lyrics were transcribed and annotated by volunteers, the quality varied greatly. 

Song section annotation [Verse], [Chorus], [Bridge] was time-consuming to check and correct—requiring listening to each section interpreting them in the context of the song.

How sentences were partitioned was also quite arbitrary. Lyrics, unlike other texts, do not have a convention on when to start a new sentence. My preference was on breaking sentences according to the musical phrasing rather than grammatically.

The lyrics were standardised using text replacements and regular expressions:

- Quotes: Remove single, double and standard { " } and non-standard { “ ” }
- Punctuations: Remove { ! }, { ? }, { ‽ }, { : }, { … }, { — }
- Punctuations: Replace semicolon { ; } → comma { , } 
- Abbreviations: { 2 am } → { 2 a.m. }
- Colloquialism: { in' } →  { ing }
- Numbers: { 42 } → { forty-two }
- Embellishments: Remove vocalisations (often in brackets): {(ah-aah-aah)} → {}
- Embellishments: Remove spoken words that are non-musical in nature
- Extra spaces at end of sentences: {" \n"} → {"\n"}
- Words joined by commas without delimiters { apart,now } → { apart, now }

These lyrical patterns were left intact:
- Contractions: { 'bout }, { 'cause }. These reduces the syllables of words and have an impact musically.

Once the corpus was compiled and standardised, it had to be broken down into snippets compatible with the RNNs network sequence length. Instead of simply concatenating all the lyrics, attention was paid to ensure the lyrics from adjacent songs will not bleed into each other. This is achieved be vectorising the data song-wise.

## Benchmark

The benchmark model selected was based on the well known [char-rnn](https://github.com/karpathy/char-rnn) project. It has been widely used and would be a good baseline comparison, with a caveat being that only the default hyper-parameters were used and not the implementation.

![](docs/benchmark.png)
_Note: LSTM / GRU network sequence length: 50, Batch size: 128_

The best model performs better at 92.45% compared to the benchmark’s 90.52%. Interestingly, the perplexity of the benchmark model is lower at 2.47 compared to the best model’s 2.7. It seems a lower perplexity does not guarantee a better language model—at least for the task of forming valid words (1st metric).

## Implementations

- Early stopping during model training to optimise computational resources
- Custom hyper-parameter search that enabled network sequence length variation
- Automated workflow that streamlined the training and evaluation of multiple models
- Additional features, such as the spreadsheet reports, facilitated a more organised research process

## Source code
Jupyter notebook: [nls.ipynb](https://github.com/danieltjw/novel-lyrics-synthesis/blob/master/nls.ipynb)

