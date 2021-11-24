
.. image:: https://user-images.githubusercontent.com/26142730/128845891-26bdedfe-a442-45eb-b2de-b3672fd729a2.png
  :width: 50 %
  :align: center

*HTV-Learn* is a framework that uses the Hessian-Schatten Total-Variation semi-norm as a regularizer in supervised learning as well as a measure of model complexity.

The aim of this repository is to facilitate the reproduction of the results reported in the research papers:

* "Learning of Continuous and Piecewise-Linear Functions with Hessian Total-Variation Regularization" (submitted);

* "Measuring Complexity of Learning Schemes Using Hessian-Schatten Total-Variation" (submitted).

.. contents:: **Table of Contents**
    :depth: 2


Requirements
============

* numpy >= 1.10
* django >= 3.2.7
* scipy >= 1.7.1
* torch >= 1.9.0
* matplotlib >= 3.4.3
* plotly >= 5.3.1
* cvxopt >= 1.2.6
* odl >= 0.7.0

The code was developed and tested on a x86_64 Linux system.

Installation
============

To install the package, we first create an environment with python 3.8:

.. code-block:: bash

    >> conda create -y -n htv python=3.8
    >> source activate htv

Then, we clone the repository:

.. code-block:: bash

    >> git clone https://github.com/joaquimcampos/HTV-Learn
    >> cd HTV-Learn

.. role:: bash(code)
   :language: bash

Finally, we install the requirements via the command:

.. code-block:: bash

  >> pip install --upgrade -r requirements.txt

.. role:: bash(code)
   :language: bash

Reproducing results
===================

The models shown in the [Campos2020]_ are saved under the `models/ <https://github.com/joaquimcampos/HTV-Learn/tree/master/models>`_ folder.
We can plot a model and its associated dataset via the command:

.. code-block:: bash

    >> ./scripts/plot_model.py [model]

To reproduce the results from scratch, we can run the scripts matching the pattern :bash:`./scripts/run_*.py`
(e.g. :bash:`./scripts/run_face_htv.py`). To see the running options, add :bash:`--help` to this command.

Developers
==========

*HTV-Learn* is developed by the `Biomedical Imaging Group <http://bigwww.epfl.ch/>`_,
`École Polytéchnique Fédérale de Lausanne <https://www.epfl.ch/en/>`_, Switzerland.

Original author:

-   **Joaquim Campos** (joaquimcampos15@hotmail.com)

References
==========

.. [Campos2020] J. Campos, S. Aziznejad, M. Unser, "Learning of Continuous and Piecewise-Linear Functions with Hessian Total-Variation Regularization" (submitted)

.. [Aziznejad2020] S. Aziznejad, J. Campos, M. Unser, "Measuring Complexity of Learning Schemes Using Hessian-Schatten Total-Variation" (submitted)

License
=======

The code is released under the terms of the `MIT License <https://github.com/joaquimcampos/HTVLearn/blob/master/LICENSE>`_

Acknowledgements
================

This work was supported in part by the Swiss National Science Foundation under Grant 200020_184646 / 1 and in part by the European Research Council (ERC)
under Grant 692726-GlobalBioIm.

Logo
----

The logo rights belong to © Ben Foster 2021.
You can check his website `here <https://benfostersculpture.com/>`_.
