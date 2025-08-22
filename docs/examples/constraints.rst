.. _constraints:

================
With constraints
================

.. warning::
    This is an advanced feature that should be used with care.
    In most cases, it is not necessary to use constraints.

In the majority of applications, the ABX evaluation is fully specified using only the ON, BY, and ACROSS conditions.
However, in some cases, because of the specificites of the dataset or the hierarchy of attributes, it can be
necessary to filter the triplets further.

The :py:class:`.Subsampler` can already be used to limit the number of :py:class:`.Cell` in a :py:class:`.Task`, but
it only operates at the cell level, not at the triplet level. It can subsample the cells, and remove some based
on the categories used for the ON, BY, and ACROSS conditions. But there is no constrained filtering inside each cell.

To achieve this finer filtering of triplets, the :py:class:`.Score` class accepts a :py:type:`.Constraints` argument.
:py:type:`.Constraints` are lists of polars expressions that are applied to each triplet before computing the ABX
score of the individual cell. The expressions should involve the labels of the triplets contained in ``.Dataset.labels``,
suffixed by ``_a``, ``_b``, and ``_x``. This is a powerful and general mechanism, and it can be used to do any kind of filtering.

For example, let's say we are interested in accent discrimination from sentence embeddings.
The dataset is described by the following labels:

.. csv-table::
    :header: sentence, accent, speaker, path

    Hello world, american, A1, /path/1.pt
    Hello world, british, B1, /path/2.pt
    We went to the park yesterday, french, F1, /path/3.pt
    ...

We want to understand whether our embeddings can discriminate accents when the semantic content is the same.
Therefore, we compute the ABX score ON accent, BY sentence.
However, we want to ensure that the speakers are different in each triplet to avoid the trivial cases where the same
speaker has uttered both stimuli A and X. We achieve this by constraining all speakers in a triplet to be different:

.. code-block:: python

    import polars as pl
    import torch
    from fastabx import Dataset, Task, Score

    labels = pl.read_csv("labels.csv")
    embeddings = torch.stack([torch.load(path) for path in labels["path"]]) # Shape (len(labels), dim)

    constraints =  [
        pl.col("speaker_a").ne(pl.col("speaker_x"))
        & pl.col("speaker_a").ne(pl.col("speaker_b"))
        & pl.col("speaker_x").ne(pl.col("speaker_b"))
    ]

    dataset = Dataset.from_numpy(embeddings, labels)
    task = Task(dataset, on="accent", by=["sentence"])
    score = Score(task, "angular", constraints=constraints)
    print(score.collapse(levels=["sentence"]))


In this particular example, we provide :py:func:`.constraints_all_different` to directly build the constraints.

.. code-block:: python

    from fastabx.constraints import constraints_all_different

    constraints = constraints_all_different("speaker")