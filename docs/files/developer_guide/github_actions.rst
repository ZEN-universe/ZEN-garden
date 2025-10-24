.. _github_actions.github_actions:

GitHub Actions
==============

ZEN-garden uses GitHub Actions to automate the testing, version bumping, and 
release of the code. GitHub Actions are specified in the
``.github\workflows\`` folder of the ZEN-garden repository. Each action is defined
in a YAML file and consists of (*1*) a list of events which trigger the action
and (*2*)* a series of jobs that must be completed when the action is triggered.
The jobs are usually defined via a a bash script. Detailed information on the
syntax of GitHub Actions can be found in the `GitHub Actions documentation
<https://docs.github.com/en/actions>`_.

ZEN-garden includes three actions that are triggered during different parts 
of the GitHub workflow. These three actions are specified by the following three
files:

* ``pytest_with_conda.yml``
* ``bump_and_release.yml``
* ``pytest_for_datasets_with_conda.yml``


Tests: ``pytest_with_conda.yml``
--------------------------------

The action is triggered whenever a pull request to the main branch of ZEN-garden 
is created or modified. It also automatically runs at 04:30 on the 
first day of every month. The action builds the ZEN-garden environment, 
installs ZEN-garden, and then runs the ZEN-garden test suite. The branch
protection rules require that all jobs in this action must pass in order for 
a pull request to be eligible for a merge. 

Bump and Release: ``pytest_with_conda.yml``
-------------------------------------------

This action is triggered whenever a push is made to main branch of ZEN-garden
(i.e., a pull request is merged). This action performs the following steps:

1. Bump the version of ZEN-garden. The version bumping can be controlled by
   hashtags in the merge commit message, as described in the section on :ref:`
   merging pull requests <contributing.merge>`. To bypass the branch protection 
   rules, the version change is first committed to a new branch and submitted 
   as a pull request. Once the tests pass, this pull requests automatically gets
   merged into the main branch once the tests pass. 

2. Create a new tag on the most recent commit of the main branch. The new
   tag matches the updated version of ZEN-garden.

3. Release the latest version of ZEN-garden to PyPI.

.. note::
   The bump-version step does does not trigger another instance of the GitHub Action
   because its merge is performed using the ``secrets.GITHUB_TOKEN`` token. This token
   cannot trigger new action workflows.

.. warning::

   The bump-version step requires a personal access token (PAT) for the pull 
   request. The ``secrets.GITHUB_TOKEN`` cannot be used since the pull request
   must trigger the tests (since these must pass). **The personal access token
   requires read and write permissions for "pull request" and "content", and it
   must be recreated every year.**


Dataset Tests: ``pytest_for_datasets_with_conda.yml``
-----------------------------------------------------

This action is automatically triggered at 04:30 on the first day of each month.
It tests that example datasets to ensure they they run properly.

