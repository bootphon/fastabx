Evaluating the discrete tokens of Spirit LM
===========================================

Install the spiritlm repository

.. code-block:: console

    $ git clone git@github.com:facebookresearch/spiritlm.git
    $ cd spiritlm
    $ pip install .

Make sure to complete their form to access and download the checkpoints.
Then, run the following code to evaluate the discrete tokens of Spirit LM

.. code-block:: python

    from pathlib import Path

    import torch
    from fastabx import zerospeech_abx
    from spiritlm.speech_tokenizer import spiritlm_base

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = spiritlm_base()


    def maker(x: Path) -> torch.Tensor:
        units = tokenizer.encode_units(str(x))["hubert"]
        return torch.tensor([int(u) for u in units.split(" ")], dtype=torch.int32)


    abx = zerospeech_abx(
        "./triphone-dev-clean.item",
        "./dev-clean",
        frequency=25,
        distance="identical",
        feature_maker=maker,
        extension=".wav",
    )
    print(abx)

