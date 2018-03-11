This is the 1.0 distribution of the The Multi-genre NLI (MultiNLI) Corpus.

License information and a detailed description of the corpus are included in the accompanying PDF.

If you use this corpus, please cite the attached data description paper.

@unpublished{williams2017broad,
	Author = {Williams, Adina and Nangia, Nikita and Bowman, Samuel R.},
	note = {arXiv preprint 1704.05426},
	Title = {A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference},
	Year = {2017},
	url = {http://arxiv.org/abs/1704.05426}}

Project page: https://www.nyu.edu/projects/bowman/multinli/


Release Notes
-------------

1.0:
- Replaces values in pairID and promptID fields. PromptID values are now shared across examples
  that were collected using the same prompt, as was originally intended, and pairID values are
  simply promptID values with an extra letter indicating the specific field in the prompt that was
  used. If you do not use these fields, this release is equivalent to 0.9.
