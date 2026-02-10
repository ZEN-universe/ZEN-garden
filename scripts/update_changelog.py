"""Automated versioning and changelog update for CI workflows.

This module is intended to be executed as a script within a continuous
integration (CI) environment. It parses pull request metadata and content
to determine the required semantic version bump, updates the project
version in ``pyproject.toml``, and generates a corresponding changelog
entry.

The script expects pull request information to be provided via environment
variables and files typically populated by GitHub Actions, including:
- ``PR_AUTHOR``: the pull request author's username
- ``PR_NUMBER``: the pull request number
- ``pr_body.txt``: a file containing the pull request body text

Based on a structured "Detailed list of changes" section in the PR body,
the script determines whether a major, minor, patch, or no version bump
is required. It then updates ``CHANGELOG.md`` and exports derived values
(such as version numbers, branch name, and commit title) back to the CI
environment for use in subsequent workflow steps.
"""

import os
import re
import tomllib  # Python 3.11+
from datetime import date, datetime
from pathlib import Path
from typing import Tuple


def extract_pr_info(pr_body_file: Path) -> Tuple[str, str, str]:
    """Extract pull request metadata and body text.

    Reads the pull request author and number from environment variables
    and loads the PR body from a file.

    Args:
        pr_body_file: Path to a file containing the pull request body.

    Returns:
        A tuple containing:
        - PR author username (string).
        - PR number (string).
        - Full PR body text (string).

    Raises:
        RuntimeError: If required environment variables are not set.
        FileNotFoundError: If the PR body file does not exist.
    """
    # read PR author and number from environment variables
    pr_author = os.environ.get("PR_AUTHOR")
    pr_number = os.environ.get("PR_NUMBER")

    if not pr_author:
        raise RuntimeError("PR_AUTHOR environment variable is not set")
    if not pr_number:
        raise RuntimeError("PR_NUMBER environment variable is not set")

    # Read PR body
    with open(pr_body_file, "r") as f:
        pr_body = f.read()

    return pr_author, pr_number, pr_body


def get_zen_garden_version(pyproject_toml_file: Path) -> str:
    """Read the project version from pyproject.toml.

    Args:
        pyproject_toml_file: Path to the pyproject.toml file.

    Returns:
        The project version string (e.g. "1.2.3").
    """
    # get ZEN-garden version
    with open(pyproject_toml_file, "rb") as f:
        data = tomllib.load(f)
    version = data["project"]["version"]

    return version


def parse_changes_from_pr_body(pr_body: str, pr_number: str, pr_author: str) -> dict:
    """Parse pull request body and categorize changes.

    Expects a section in the pull request titled "Detailed list of changes" with
    entries formatted as:
        - type: description

    Supported types are: feat, fix, docs, chore, breaking.

    Args:
        pr_body (str): Full pull request body text.
        pr_number (str): Pull request number.
        pr_author (str): Pull request author.

    Returns:
        dict: A dictionary mapping change types to lists of descriptions.

    Raises:
        ValueError: If the required section is missing, empty,
            or contains unrecognized change types.
    """
    # get pull request information
    pr_info = (
        f"[[🔀 PR #{pr_number}]"
        f"(https://github.com/ZEN-universe/ZEN-garden/pull/{pr_number}) "
        f"@{pr_author}]"
    )
    # Extract "Detailed list of changes" section
    match = re.search(
        r"## *?Detailed list of changes *?\n(.*?)(\n## |$)", pr_body, re.DOTALL
    )
    if not match:
        raise ValueError(
            "PR body does not have a section labeled `Detailed list of changes`"
        )

    changes_section = match.group(1)

    # Parse each line: expect "- type: description"
    # Format: "Section Tytle": {"Keyword": keyword, "changes": changes}
    categorized_changes = {
        "feat": {"title": "New Features ✨", "changes": []},
        "fix": {"title": "Bug Fixes 🐛", "changes": []},
        "docs": {"title": "Documentation Changes 📝", "changes": []},
        "chore": {"title": "Maintenance Tasks 🧹", "changes": []},
        "breaking": {"title": "BREAKING CHANGES ⚠️", "changes": []},
    }
    for line in changes_section.splitlines():
        line = line.strip()
        m = re.match(r"-\s*(\w+)\s*:\s*(.+)", line)  # search correct format
        if m:
            change_type = m.group(1).lower()
            description = m.group(2).strip() + f" {pr_info}"
            if change_type in categorized_changes:
                categorized_changes[change_type]["changes"].append(description)
            else:
                raise ValueError(f"Unrecognized change type {change_type} in PR body")

    # Check if any valid changes
    if not any([item["changes"] for item in categorized_changes.values()]):
        raise ValueError("Detailed list of changes are empty or could not be processed")

    return categorized_changes


def determine_bump_type(categorized_changes: dict) -> str:
    """Determine the semantic version bump required.

    Bump precedence:
        breaking > feat > fix > none

    Args:
        categorized_changes: Parsed PR changes grouped by type.

    Returns:
        str: The version bump type determined semantically.
    """
    # Determine semantic version bump
    if categorized_changes["breaking"]["changes"]:
        semver_bump = "major"
    elif categorized_changes["feat"]["changes"]:
        semver_bump = "minor"
    elif categorized_changes["fix"]["changes"]:
        semver_bump = "patch"
    else:
        semver_bump = "none"

    return semver_bump


def bumpversion(semver_bump: str, pyproject_toml_file: Path):
    """Bump the project version in pyproject.toml.

    Updates the version according to semantic versioning rules and
    writes the new version back to the file.

    Args:
        semver_bump (str): Semantic version bump type.
        pyproject_toml_file (Path): Path to the pyproject.toml file.

    Returns:
        A tuple containing:
        - New version (str).
        - Old version (str).
    """
    # Extract current version
    version = get_zen_garden_version(pyproject_toml_file)
    major, minor, patch = map(int, version.split("."))

    # Deduce new version
    if semver_bump == "major":
        major_new = major + 1
        minor_new = 0
        patch_new = 0
    elif semver_bump == "minor":
        major_new = major
        minor_new = minor + 1
        patch_new = 0
    elif semver_bump == "patch":
        major_new = major
        minor_new = minor
        patch_new = patch + 1
    else:
        major_new = major
        minor_new = minor
        patch_new = patch

    new_version = f"{major_new}.{minor_new}.{patch_new}"

    # Bump version in pyproject.toml
    # Replace version line textually (preserves formatting)
    content = pyproject_toml_file.read_text(encoding="utf-8")
    content = content.replace(
        f'version = "{version}"',
        f'version = "{new_version}"',
        1,
    )

    # Save new pyproject.toml
    pyproject_toml_file.write_text(content, encoding="utf-8")

    return new_version, version


def parse_changelog(changelog_file: Path) -> Tuple[str, str, str]:
    """Parse an existing changelog into header and body.

    The header is everything before the first subsection. Unversioned changes
    are anything with inside the section whose title includes
    "[Unversioned Changes]". The existing changelog consits of the remainder
    of the file.

    Args:
        changelog_file (Path): Path to the CHANGELOG.md file.

    Returns:
        A tuple containing:
        - Changelog header/preface (str).
        - Unversioned changes (str)
        - Existing changelog entries from past versions(str).

    Raises:
        ValueError: If the changelog does not match the expected format.
    """
    # Read existing changelog (if exists)
    if os.path.exists(changelog_file):
        with open(changelog_file, "r", encoding="utf-8") as f:
            changelog = f.read()
    else:
        raise FileNotFoundError(f"Could not fine the changelog file: {changelog_file}")

    # Extract header:
    match = re.search("(#.*?)(\n##.*$)", changelog, re.DOTALL)
    if match:
        header = match.group(1)
        changelog_existing = match.group(2)[1:]
    else:
        raise ValueError("CHANGELOG.md format is invalid. Could not find preface.")

    # Extract unversioned changes:
    match = re.search(
        r"(##\s*\[Unversioned Changes\][^\n]*\n)"  # group 1: header
        r"(.*?)"  # group 2: section body
        r"(\n##(?!#).*|\Z)",
        changelog_existing,
        re.DOTALL,
    )
    if match:
        unversioned_changes = match.group(2)
        changelog_existing = match.group(3)
    else:
        unversioned_changes = ""

    return header, unversioned_changes, changelog_existing


def parse_unversioned_changes(
    unversioned_changes: str, categorized_changes: dict
) -> dict:
    """Extracts bullet-point changes from an unversioned changelog section.

    This function scans a markdown-formatted changelog string for sections
    matching the titles defined in `categorized_changes`. For each matching
    section, it collects list items (lines starting with `-`) and appends them
    to the corresponding category's `"changes"` list.

    The function mutates and returns the provided `categorized_changes` dict.

    Args:
        unversioned_changes (str): Markdown text containing unversioned
            changelog entries organized under `### <title>` headings.
        categorized_changes (dict): Mapping of change types to metadata.
            Each value must contain:
            - "title" (str): Section title to search for.
            - "changes" (list): List to append extracted change entries to.

    Returns:
        dict: The updated `categorized_changes` dictionary with extracted
        change entries added to each category's `"changes"` list.
    """
    for change_type, item in categorized_changes.items():
        title = item["title"]
        match = re.search(
            rf"###\s*{re.escape(title)}\s*\n(.*?)(?=\n## |\n### |\Z)",
            unversioned_changes,
            re.DOTALL,
        )
        if not match:
            continue

        section = match.group(1)
        for line in section.splitlines():
            line = line.strip()
            m = re.match(r"-\s*(.*)", line)  # search correct format
            if m:  # remove empty lines
                categorized_changes[change_type]["changes"].append(m.group(1))

    return categorized_changes


def update_changelog(
    header: str,
    categorized_changes: dict,
    changelog_existing: str,
    semver_bump: str,
    new_version: str,
) -> str:
    """Generate an updated changelog with the current PR changes.

    Adds a new version section if version is bumped by a major, minor, or
    patch. Otherwise adds a new section of unversioned changes. The sections
    contain the categorized change entries from the PR metadata as well as
    any unversioned changes from previous pull requests.

    Args:
        header (str): Changelog header/preface.
        categorized_changes (dict): Parsed PR changes grouped by type.
        changelog_existing (str): Existing changelog content.
        semver_bump (str): Semantic version bump type.
        pr_number (str): Pull request number.
        pr_author (str): Pull request author username.
        new_version (str): The new version of ZEN-garden

    Returns:
        str: The full updated changelog text.
    """
    # Append changes to CHANGELOG.md --------------------------------------
    # initialize changelog additions
    changelog_addition = ""

    # add version number if bump
    if semver_bump != "none":
        changelog_addition += f"\n## [v{new_version}] - {date.today()} \n"
    else:
        changelog_addition += f"\n## [Unversioned Changes] - {date.today()} \n"

    for items in categorized_changes.values():
        if items["changes"]:
            title = items["title"]
            changelog_addition += f"\n### {title}\n"
            for change in items["changes"]:
                changelog_addition += f"- {change}\n"

    return (
        header.strip("\n")
        + "\n\n"
        + changelog_addition.strip("\n")
        + "\n\n"
        + changelog_existing.strip("\n")
    )


def suggest_branch_name(new_version):
    """Suggest a branch name for the version bump.

    The branch name includes the new version and a timestamp.

    Args:
        new_version (str): New version string.

    Returns:
        str: A suggested branch name.
    """
    branch_name = f"v{new_version}-{datetime.now().strftime('%d.%m.%Y-%H.%M.%S')}"

    return branch_name


def suggest_commit_title(semver_bump, old_version, new_version):
    """Suggest a commit title for the changelog/version update.

    Args:
        semver_bump (str): Semantic version bump type.
        old_version (str): Previous version string.
        new_version (str): New version string.

    Returns:
        str: A human-readable commit title.
    """
    if semver_bump != "none":
        commit_title = f"Bump version: v{old_version} → v{new_version}"
    else:
        commit_title = f"Update changelog: v{new_version}"

    return commit_title


def main():
    """Main entry point for the version bump and changelog update process.

    This function automates the process of determining a semantic version bump
    (major, minor, or patch) based on the changes in a pull request, updating
    the version in the pyproject.toml file, and generating the necessary
    changelog entries. Additionally, it writes important metadata (new version,
    branch name, and commit title) to the GitHub Actions environment file for
    use in subsequent steps.

    The following steps are performed:
        1. Extract pull request information (author, number, and body).
        2. Parse the "Detailed list of changes" section of the PR body.
        3. Determine the semantic version bump type (major, minor, or patch).
        4. Update the version in the pyproject.toml file.
        5. Parse the existing changelog and update it with new changes.
        6. Suggest a commit title and branch name based on the version bump.
        7. Write the updated changelog back to the file.
        8. Append version and metadata to the GitHub Actions environment file.

    Args:
        None

    Returns:
        None

    Raises:
        RuntimeError: If required environment variables (e.g., PR_AUTHOR,
            PR_NUMBER) are not set.
        ValueError: If the PR body format is invalid or missing the required
            section.
        FileNotFoundError: If necessary files (e.g., pyproject.toml,
            pr_body.txt) are missing.
    """
    changelog_file = Path("CHANGELOG.md")
    pr_body_file = Path("pr_body.txt")  # saved in github action
    pyproject_toml_file = Path("pyproject.toml")

    pr_author, pr_number, pr_body = extract_pr_info(pr_body_file)
    categorized_changes = parse_changes_from_pr_body(pr_body, pr_number, pr_author)

    semver_bump = determine_bump_type(categorized_changes)

    new_version, old_version = bumpversion(semver_bump, pyproject_toml_file)

    header, unversioned_changes, changelog_existing = parse_changelog(changelog_file)

    categorized_changes = parse_unversioned_changes(
        unversioned_changes, categorized_changes
    )

    changelog = update_changelog(
        header, categorized_changes, changelog_existing, semver_bump, new_version
    )

    commit_title = suggest_commit_title(semver_bump, old_version, new_version)
    branch_name = suggest_branch_name(new_version)

    # Save new changelog file
    with open(changelog_file, "w", encoding="utf-8") as f:
        f.write(changelog)

    # Write to the GitHub Actions environment file
    with open(os.environ["GITHUB_ENV"], "a") as env_file:
        env_file.write(f"NEW_VERSION={new_version}\n")
        env_file.write(f"OLD_VERSION={old_version}\n")
        env_file.write(f"BRANCH_NAME={branch_name}\n")
        env_file.write(f"COMMIT_TITLE={commit_title}\n")
        env_file.write(f"SEMVER_BUMP={semver_bump}\n")


if __name__ == "__main__":
    main()
