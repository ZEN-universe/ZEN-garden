.. _api_reference.api_reference:

API Reference
==============

:Release: |version|
:Date: |today|

Overview
--------

The diagram below shows the high-level sequence of a ZEN-garden run. 
Solid arrows are the run sequence; 
dashed arrows close a loop back to its first step.
Each node names the class or function that performs that stage.

.. include:: /files/references/_generated/workflow_diagram.rst

.. note::

    This diagram is generated at documentation-build time from
    ``@workflow_step`` markers on the corresponding functions (see
    :py:mod:`zen_garden.workflow_step` and
    ``docs/_ext/workflow_diagram.py``). To add, move, or reword a stage,
    edit the marker on the function that performs it rather than this page.


ZEN-garden
----------

.. toctree::
   :maxdepth: 1

   ../api/general
   ../api/input
   ../api/model
   ../api/postprocess
   ../api/plugin_system









