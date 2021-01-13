.. pyadlml documentation master file, created by
   sphinx-quickstart on Sun Sep 20 15:52:57 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=============================================
Activities of Daily Living - Machine Learning
=============================================

TODO insert readme introduction

References

- want to collect your own data: https://github.com/tcsvn/activity-assistant
- notebooks: https://github.com/tcsvn/pyadlml/notebooks

.. warning::
   Project is under heavy development. Things are going to change and break. I am
   currently writing the documentation, meaning this is a draft rather than a document
   that should be taken seriously. So take everything written here with a huge
   grain of salt because it is subject to change.

Modules
=======

.. raw:: html

   <div class="row">
       <div class="col-sm-4 col-sm-offset-1">
          <a class="reference internal" href="modules/robust.html">
             <h2>pyadlml.dataset</h2>
          </a>
          <p>
            Provides easy access to datasets that can be used as benchmarks.
          </p>
       </div>
       <div class="col-sm-4 col-sm-offset-1">
          <a class="reference internal" href="modules/survival.html">
             <h2>pyadlml.preprocessing</h2>
          </a>
          <p>
            Some tools for preprocessing, such as discrete time encoder and
            tools for feature generation.
          </p>
       </div>
   </div>
   <div class="row">
       <div class="col-sm-4 col-sm-offset-1">
          <a class="reference internal" href="modules/robust.html">
             <h2>pyadlml.plot</h2>
          </a>
          <p>
            Numerous visualizations for activities, devices and their interaction.
          </p>
       </div>
       <div class="col-sm-4 col-sm-offset-1">
          <a class="reference internal" href="modules/survival.html">
             <h2>pyadlml.models</h2>
          </a>
          <p>
            Wrapper around common models to provide a seamless start for
            newcomers.
          </p>
       </div>
   </div>


Overview
========

.. toctree::
   :maxdepth: 2

   user_guide.rst
   dataset.rst
   api.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
