.. _slicing:

================
Slicing features
================

To compute phoneme or triphone based ABX, we need phone-level alignments.
Those are described in :doc:`item files </items>`, like the following

.. code-block:: text

   #file onset offset #phone prev-phone next-phone speaker
   6295-244435-0009 0.2925 0.4725 IH L NG 6295
   6295-244435-0009 0.3725 0.5325 NG IH K 6295
   6295-244435-0009 0.4325 0.5725 K NG AH 6295
   ...
   2902-9006-0005 0.3725 0.6925 UW JH L 2902
   2902-9006-0005 0.5125 0.7525 L UW IY 2902
   2902-9006-0005 0.5925 0.7925 IY L AH 2902
   ...

We compute the representations using the full audio file, and we then slice to only get the frames
that correspond to the unit of interest. Since the frames are downsampled, there is a decision to make
on exactly which frame to keep and which to remove.

Let :math:`t_\text{on}, t_\text{off}` the times of start and end of the triphone or phoneme considered, with :math:`t_\text{on} < t_\text{off}`.
This corresponds to the columns "onset" and "offset" of the item file.

Let :math:`\Delta t` the constant time step between consecutive features, 20 ms for example.
We follow ABXpy, and consider that the discrete times associated to the features
are :math:`t_i = \frac{\Delta t}{2} + \Delta t \times i`.

We define the set of frames indices to select :math:`I` as

.. math::
    I = \left\{ i \mid \onset \leq t_i \leq \offset \right\},

We have, for any :math:`i \in \mathbb{N}`,

.. math::
	i \in I \Leftrightarrow \begin{cases}
	    i \geq  \frac{\onset}{\Delta t} - \frac{1}{2} \\
	    i \leq \frac{\offset}{\Delta t} - \frac{1}{2}
	    \end{cases}.

Therefore, the beginning and end indices (both included) are:

.. math::
	\begin{aligned}
	    i_\text{start} & = \min(I) = \left\lceil \frac{\onset}{\Delta t} - \frac{1}{2} \right\rceil, \\
	    i_\text{end} & = \max(I) = \left\lfloor \frac{\offset}{\Delta t} - \frac{1}{2} \right\rfloor.
	\end{aligned}

In Libri-Light, because the features were sliced with :code:`features[i_start : i_end]` instead of :code:`features[i_start : i_end + 1]`,
the last included index was :math:`i_\text{end} - 1 = \left\lfloor \frac{\offset}{\Delta t} - \frac{1}{2} \right\rfloor - 1`
(see `here <https://github.com/facebookresearch/libri-light/blob/3fb5006a39e6f9e86daf3e5e52bc87630f3cdf3e/eval/ABX_src/abx_iterators.py#L178-L189>`_).

Timestamp validation
====================

For fixed-rate slicing, ``frequency`` must be a positive finite ``int``, decimal string, or
:class:`decimal.Decimal`. Use a string such as ``"49.95"`` for a fractional rate. Item onsets and offsets must be
finite, non-null, and satisfy ``0 <= onset <= offset``.

With ``Dataset.from_item_with_times``, each feature frame must have exactly one finite timestamp in a
one-dimensional array. Both interval boundaries are inclusive: frames with ``onset <= timestamp <= offset``
are selected. Timestamp count must equal feature-frame count. Timestamp values are rounded to the greater decimal
precision of the item onset and offset columns before comparison, so both metadata boundaries contribute to the
comparison precision.
