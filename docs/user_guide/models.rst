9. Models
*********

The following models are implemented and shipped with pyadlml as is. Some models
may need additional library's installed beforehand. Pyadlml doesn't include those
library's in order to not be bloated.

HMM: Hidden Markov Model
========================

TODO

RNN: Recurrent Neural Net
=========================

TODO

NODE: Neural Ordinary Differential Equation
===========================================

Definition
~~~~~~~~~~

Examples
~~~~~~~~

first install

.. code-block::

    $ pip install torchdiff-eq

Then

.. code:: python

    from pyadlml.models import NODE

    NODE().fit()