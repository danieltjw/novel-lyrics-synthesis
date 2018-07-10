# Novel Lyrics Synthesis

In this project, Recurrent Neural Networks (RNNs) are applied to sequence modeling and Natural Language Processing (NLP) tasks. Character-level language models were trained on 100 song lyrics and then used to generate new lyrics. The quality of the generated lyrics were evaluated using 3 metrics—ability to form valid words, emulate the original sentence structure (frequency distribution of sentence length) and similarity (BLEU score).

## Summary of findings:

- The latest CuDNN accelerated GRUs brought about a 7.3x / 6.7x speed up in training time compared to non-CuDNN GRU implementation 1 / 2 respectively.
- Perplexity has a moderate negative correlation (r=-0.632, DF=15; P<0.01) with the valid words percentage (1st metric). This indicates that the 1st metric measures some variability that is independent from perplexity. 
- Lower perplexity does not guarantee a better language model—at least for the task of forming valid words (1st metric).
- GRU networks with shorter sequence length (10, 20) are not able to properly emulate the sentence length of the original corpus. These networks' sentences have a significantly higher standard deviation (23.7, 16.6) compared to the original corpus (11.8). The mean sentence length of across different models did not vary greatly. This shows that while shorter models can adequately learn the frequency of newline chars, they struggle with emulating the structure (frequency distribution) of the original lyrics.


## Results:

![](docs/hyper-parameter_search_results.png)
_Note: GRU network sequence length = 50, Batch_size = 128_

---

![](docs/heatmap_valid_word_per.png)

_Note: GRU network sequence length = 50, Batch_size = 128_

## Implementations:

- A custom hyper-parameter search was implemented to allow for network sequence length variation.
- Two evaluation metrics (valid words and sentence length) were designed and implemented, and used to draw insights on hyper-parameters' impact on learning.
- An automated workflow was developed that allow multiple models to be trained and evaluated. 
- Additional features, such as the spreadsheet reports, were integrated into the code to make the research process more organised.
