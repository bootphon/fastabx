Usage with other libraries
===========================

In this tutorial, we will show how to compute the Triphone ABX error rate
within speaker of the 11th layer of HuBERT base, on the dev-clean subset of LibriSpeech.
This will show how to use fastabx with various libraries.
The only thing to adapt is the feature extraction part, everything else is handled by ``zerospeech_abx``.

In the following examples, the wav files are in the ``dev-clean`` directory and the item file is ``triphone-dev-clean.item``.


With torchaudio
---------------

.. code-block:: python

    import torch
    import torchaudio
    from fastabx import zerospeech_abx

    layer = 11
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = torchaudio.pipelines.HUBERT_base
    model = bundle.get_model().to(device)


    def maker(path: str) -> torch.Tensor:
        x, sr = torchaudio.load(str(path))
        assert sr == bundle.sample_rate
        features, _ = model.extract_features(x.to(device))
        return features[layer - 1]


    abx = zerospeech_abx(
        "./triphone-dev-clean.item",
        "./dev-clean",
        feature_maker=maker,
        extension=".wav",
    )
    print(abx)


With S3PRL
----------

.. code-block:: python

    import torch
    import torchaudio
    from fastabx import zerospeech_abx
    from s3prl.nn import S3PRLUpstream


    layer, sample_rate = 11, 16_000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = S3PRLUpstream("hubert").eval().to(device)
    assert sample_rate // model.downsample_rates[layer] == 50


    def maker(path: str) -> torch.Tensor:
        x, sr = torchaudio.load(str(path))
        assert sr == sample_rate
        return model(x.to(device))[layer]


    abx = zerospeech_abx(
        "./triphone-dev-clean.item",
        "./dev-clean",
        feature_maker=maker,
        extension=".wav",
    )
    print(abx)

