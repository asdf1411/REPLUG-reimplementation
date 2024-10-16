# REPLUG-reimplementation
This repository is the reimplementation based on the research paper **REPLUG: Retrieval-Augmented Black-Box Language Models**, and its original implementation at [REPLUG](https://github.com/swj0419/REPLUG).

To run this code, please replace the corresponding files in the original repository with the ones provided in this repository and add the file `lm_dataformat.py`, which does not exist in the original implementation.

For the question-answering tasks, you can generate the embeddings&index for the corpus first using `generate_passage_embeddings.py` from the original implementation. Then, run `save_logprob_data.py` to save the retrieved results in `cache_dict`.

After collecting all the retrieved documents for each training dataset, running `replug_lsr.py` to train the retriever, remember to set the configuration in `present_configs` and provide the OPENAI_API_KEY in `replug_lsr.py` before training.
