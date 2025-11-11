.. _documentation.documentation:

Documentation Guide
=====================

Overview
--------

The documentation, located in the ``docs`` folder, is written in 
`reStructuredText (rst) <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_ 
and is built using `Sphinx <https://www.sphinx-doc.org/en/master/>`_. 
All necessary packages are installed when installing the requirements for the 
developer environment (:ref:`dev_install.dev_install`). 

The ``docs`` folder has the following file structure:

.. code-block:: text

    docs/
    |--_build/
    |--_ext/
    |--dataset_examples/
    |--files/...
    |   |--folder/file1.rst
    |   `--folder/file2.rst
    |--conf.py  
    |--index.rst
    `--make.bat


Each of these elements is described in detail below:

* ``_build`` - a folder containing the html files of the documentation. 
  This folder does not get pushed to the GitHub repository and is only present
  after the Sphinx documentation is built locally (see 
  :ref:`documentation.build_locally`).

* ``_ext`` - a folder containing manually programmed extensions for
  the Sphinx package. For example, this folder contains the ``docsttring.py`` 
  extension, which allows users to insert text from docstrings of methods or 
  classes into the documentation (see :ref:`documentation.docstrings`).

* ``dataset_examples`` - a folder containing the input files for the 
  :ref:`dataset examples <dataset_examples.dataset_examples>`. Developers can 
  change these files to update the dataset examples. The dataset 
  examples get automatically uploaded to Zenodo in every major or minor release 
  of ZEN-garden. A minor release can be created if ``#minor`` is 
  included in the pull request header.

* ``files`` - a folder containing the reStructuredText files that make up the 
  documentation. All text for the documentation is located here. The files
  are organized in sub-folders by section of the documentation.

* ``conf.py`` - a python configuration file for the Sphinx documentation. This
  file can be used to activate Sphinx extensions or tell Sphinx to ignore 
  certain files. 

* ``index.rst`` - a reStructuredText file which sets the structure of the 
  documentation. It servers as a table of contents for the documentation, 
  telling Sphinx which rst files to read and where to place them in the 
  website.

* ``make.bat`` - a batch file which controls how Sphinx is called from the 
  command line. It defines the ``make`` commands used for building the 
  documentation locally.  This file can be modified to, for instance, alter 
  the behavior of  the ``make clean`` command.


.. _documentation.build_locally:

Building the documentation locally
----------------------------------

Always build the documentation locally before pushing it to the ZEN-garden
repository. Local builds allow developers to preview the documentation before 
publication. To build the documentation locally, use the following steps:

1. Open a terminal window and :ref:`activate the ZEN-garden environment
   <installation.activate>`.

2. Navigate to the docs folder of the ZEN-garden installation:

   .. code::

       cd <path-to-ZEN-garden>\docs

3. Remove all previous builds and clear any cached files from Sphinx. To do
   this, run the command shown below. This step should repeated each time
   the documentation is rebuilt.

   .. code::

       make clean

4. Build the local documentation using the command:

   .. code::

       make html

5. The documentation is then available in the ``docs/build/html`` folder. Open the 
   ``index.html`` file in your browser to preview the documentation. Repeat steps
   3 and 4 each time you want to update the build.


Documentation guidelines
------------------------

* Always update the documentation each time changes are made to the ZEN-garden
  code. The documentation should be clear and contain all the necessary 
  information to understand new features or bug fixes.
* Wrap all lines at 80 characters.
* New paragraphs are marked with one blank line and new sections are marked 
  with two blank lines.
* All sections must have a manual label if referenced. The label must follow the 
  format ``<file_name>.<section_name>``. The first part of the label, 
  ``<file_name>`` should be the same for all labels within the same restructured
  text file. This makes it easy to identify what file the referenced section is 
  written in. It also prevents duplicate labels when multiple sections have the 
  same name.
* Always use "make clean" before "make html" to ensure that you receive all 
  warnings and error messages when building the documentation. Never push
  documentation which produces warning messages.
* Avoid redundant, duplicate information in the docstrings and the documentation. 
  If the docstrings and the documentation require similar content, consider
  using the :ref:`docstring extenstion <documentation.docstrings>`. This 
  extension copies text from the docstring and inserts into the documentation.
  This helps avoid duplicate text and thus helps ensure that all elements of 
  the documentation are consistent and up-to-date.
* All figures are stored in the ``docs\files\figures`` folder. Within this folder
  figures can be further organized into sub-folders based on the section of the
  documentation in which they intended for. 


Addition new pages to the documentation
---------------------------------------

The following steps allow new pages to be added to the documentation:

1. Add a new restructured text file in the ``docs\files`` folder. Files in 
   this folder are typically grouped into subfolders by section of the 
   docmentation (e.g. Quick Start, Developer Guide, etc.). Example:
   ``docs\files\quick_start\<new_file_name>.rst``

2. Insert the desired text in the restructured text file. The following template
   can be used to create the new file:

   .. code-block:: rst
    
      .. _page_label.page_label:

      ###########
      Page Title
      ###########

      Insert description here. You can also reference a 
      :ref:`Section <page_label.section_label>`, a 
      :ref:`Subsection <page_label.subsection_label>`, or a figures 
      (:numref:`page_label.figure_label`). Bold lettering can be written like 
      **this**, italics like *this*, and files or variables like ``this``. 
      Finally, links can be included as follows:
      `ZEN-garden Github <https://github.com/ZEN-universe/ZEN-garden>`_


      .. _page_label.section_label:

      Section Name
      =============

      This section contains a code block of python code:

      .. code:: python

        import numpy as np
        print("Hello World")

      It also contains a figure:

      .. _page_label.figure_label:

      .. figure:: ../figures/tutorials/figure_name.png
          :figwidth: 550 pt
          :align: center
        
          Caption goes here.


      .. _page_label.subsection_label:

      Subsection Name
      ---------------

      This subsection contains a numbered list:

      1. First item
      2. Second item

      The following is creates a bulleted list:

      * Item 
      * Another item


3. Add the path of new file to the ``docs\index.rst`` file. The file paths are
   expressed relative to the ``docs`` folder. The location of the
   file path determines where in the documentation the new page is inserted. For
   example, the following syntax inserts the new file in the Quick Start 
   section between the page on installation and running models:  

   .. code-block:: rst

       .. toctree::
   
           :maxdepth: 1
           :caption: Quick Start

           files/quick_start/installation
           files/quick_start/<new_file_name>.rst
           files/quick_start/running_models

.. _documentation.docstring:

Docstrings 
----------

All modules, functions, classes, and methods should contain 
well-written docstrings which describe their structure and function.
ZEN-garden uses google-style docstrings that can be interpreted by Sphinx's 
Napoleon extension. Templates for docstrings of classes and methods are 
provided below. These should serve as a starting point for developers. Elements
which are not required can be deleted from the template upon implementation. 

Template for a class docstring:

.. code:: text

    """
    One-line summary of the class.

    One or more paragraphs describing the high-level structure and function
    of the class. The following information should be included:

    (1) what is the overall function of the class, i.e. why was it created?
    (2) how is the class set up? What are the structures and basic principles 
        of the class? This description should allow new developers to be able 
        to scroll through the class and understand the basic function and setup.
    (3) What features does the class have? This description should enable new 
        developers to identify whether a class is relevant for their programming
        task.
    (4) what are the most important attributes of the class? Do any rules govern
        these attributes? These can also be listed under the attributes section 
        below.
    (5) what are the most important methods of the class? Do these follow any 
        specific logic?

    Attributes:
        attr1 (type): Description of `attr1`. Make sure to properly indent each 
            new line by four spaces when using multi-line descriptions.
        attr2 (type): Description of `attr2`.

    Args:
        param1 (type): Description of `param1`. Parameters include any inputs
            to the __init__ function.
        param2 (type, optional): Description of `param2`. Defaults to None.

    Raises:
        ValueError: Outline conditions under which error is raised.
        RuntimeError: Outline conditions under which error is raised.

    See Also:
        RelatedClass: List all inherited classes here and give a brief
            explanation why.
        relatedfunction(): List all related functions here and give a 
            brief explanation of the relationship.

    Example:
        Example usage of the class.

        >>> obj = ClassName(param1, param2)
        >>> obj.method1("value")

    Todo:
        - List items here.
        - These will not appear in the online documentation
    """

Template for a method docstring:

.. code:: text
    
    """
    One-line summary of the method.

    One or more paragraphs describing the method’s purpose, behavior, and any 
    important implementation details or side effects. This description 
    should enable users to understand what the method does and when to use it.

    Args:
        param1 (type): Description of `param1`. Make sure to properly indent 
            each new line by four spaces when including  multi-line 
            descriptions.
        param2 (type, optional): Description of `param2`. Defaults to None.

    Returns:
        return_type: Description of the return value.

    Raises:
        SomeError: Description of conditions when this exception is raised.
        AnotherError: Description for another exception, if applicable.

    Examples:
        Basic usage example:

        >>> obj = ClassName()
        >>> result = obj.method_name(param1, param2)
        >>> print(result)

    Todo:
        - List any planned improvements or refactoring tasks here.
        - These will not show up in the documentation.
    """

.. _documentation.tips_and_tricks:

Tips and Tricks 
---------------

.. _documentation.autodocs:

Autodocs
^^^^^^^^

ZEN-garden uses the Sphynx Autodocs to create documentation for 
all modules, classes, and methods. The automatic documentation is created by the 
``autosummary`` directives in the the following files (which are called 
by ``<path-to-ZEN-garden>/files/references/api_reference.rst`` ):

- ``<path-to-ZEN-garden>/files/api/general.rst``
- ``<path-to-ZEN-garden>/files/api/model.rst``
- ``<path-to-ZEN-garden>/files/api/preprocess.rst``
- ``<path-to-ZEN-garden>/files/api/postprocess.rst``

The ``autosummary`` command automatically produces documentation for 
modules and sub-modules based on the docstrings in the code. In addition, it 
creates reStructuredText files for each module that describe how the modules 
are documented. These automatically generated files are written to the 
``<path-to-ZEN-garden>/files/api/generated`` folder at the time when the 
documentation is built.

In rare occasions, the formatting of modules and submodules created by 
``autosummary`` needs to be adjusted manually. Sphinx has a variety of options 
which allow users to control the behavior of the Autodocs. To override the 
documentation produced by the generated reStructuredText files, follow these 
steps:

1. Find the reStructuredText file for the module you would like to fix in the
   ``<path-to-ZEN-garden>/files/api/generated`` folder. Copy and paste this
   file into the ``<path-to-ZEN-garden>/files/api/modules`` folder. Unlike the
   ``generated`` folder, the ``modules`` folder gets pushed to GitHub and is
   designed to hold all manually written module documentations.

2. Make all required changes to the file (the one in the ``modules`` folder).

3. Point Sphinx to the newly created documentation file. Suppose, for example,
   the manual documentation was written for the module ``manual_module``. Then,
   replace the old ``autosummary`` directive.

   Replace::

       .​. autosummary::
          :toctree: generated

          zen_garden.manual_module
          zen_garden.other_module

   with::

       .​. autosummary::
          :toctree: modules

          zen_garden.model.manual_module

       .​. autosummary::
          :toctree: generated

          zen_garden.model.other_module


.. _documentation.docstrings:

Use text from Docstrings in the Documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The docstring extension allows text from the main body of a docstring to be 
inserted into the documentation text. The main body of the docstring is 
everything except the summary line and information about parameters, arguments, 
or return values (prefaced by 'Args:', 'Returns:', 'Raises:', etc.). 
The docstring extension can help reduce redundant text and avoid inconsistencies
in the documentation. 

To activate the docstring extension, insert the ``docstring_class`` or 
``docstring_method`` directive into an reStructuredText files. These directives 
are followed by a module name or method name, respectively. Two examples below 
show how to insert the docstrings for the ``ConversionTechnologyRules`` class 
and its method ``constraint_capacity_factor_conversion``.


.. code::

    .. docstring_class:: zen_garden.model.objects.technology.conversion_technology.ConversionTechnologyRules

    .. docstring_method:: zen_garden.model.objects.technology.conversion_technology.ConversionTechnologyRules.constraint_capacity_factor_conversion
