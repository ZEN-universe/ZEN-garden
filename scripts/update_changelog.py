import os
import re
from datetime import date
from importlib.metadata import version as get_version
import tomllib  # Python 3.11+
from pathlib import Path

def extract_pr_info(pr_body_file: Path):

    # read PR author and number from environment variables
    pr_author = os.environ.get("PR_AUTHOR")
    pr_number = os.environ.get("PR_NUMBER")

    if not pr_author:
        raise RuntimeError("PR_AUTHOR environment variable is not set")
    if not pr_number:
        raise RuntimeError("PR_NUMBER environment variable is not set")
    
    ## Read PR body
    with open(pr_body_file, "r") as f:
        pr_body = f.read()

    return pr_author, pr_number, pr_body


def get_zen_garden_version(pyproject_toml_file: Path):
    
    ## get ZEN-garden version
    with open("pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    version = data["project"]["version"]

    return version

def parse_changes_from_pr_body(pr_body: str):
    
    # Extract "Detailed list of changes" section
    match = re.search(
        r"## *?Detailed list of changes *?\n(.*?)(\n## |$)", pr_body, re.DOTALL
    )
    if not match:
        raise ValueError("PR body does not have a section labeled `Detailed list of changes`")

    changes_section = match.group(1)

    # Parse each line: expect "- type: description"
    categorized_changes = {
        "feat": [],
        "fix": [],
        "docs": [],
        "chore": [],
        "breaking": []
    }

    for line in changes_section.splitlines():
        line = line.strip()
        m = re.match(r"-\s*(\w+)\s*:\s*(.+)", line) # search correct format
        if m:
            change_type = m.group(1).lower()
            description = m.group(2).strip()
            if change_type in categorized_changes:
                categorized_changes[change_type].append(description)
            else:
                raise ValueError(f"Unrecognized change type {change_type} in PR body")

    # Check if any valid changes
    if not any(categorized_changes.values()):
        raise ValueError("Detailed list of changes are empty or could not be processed")

    return categorized_changes


def determine_bump_type(categorized_changes):
    
    # Determine semantic version bump
    if categorized_changes["breaking"]:
        semver_bump = "major"
    elif categorized_changes["feat"]:
        semver_bump = "minor"
    elif categorized_changes["fix"]:
        semver_bump = "patch"
    else:
        semver_bump = "none"

    return semver_bump

def bumpversion(semver_bump: str, pyproject_toml_file: Path):
    
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

def parse_changelog(changelog_file: Path):
    # Read existing changelog (if exists)
    if os.path.exists(changelog_file):
        with open(changelog_file, "r", encoding="utf-8") as f:
            changelog = f.read()

    # Remove preface:
    match = re.search("(#.*?)(\n##.*$)", changelog, re.DOTALL)
    header = match.group(1)
    changelog_existing = match.group(2)

    return header, changelog_existing

def update_changelog(header, categorized_changes, changelog_existing, 
                     semver_bump, pr_number, pr_author):
    # Append changes to CHANGELOG.md --------------------------------------
    pr_info = f"[[🔀 PR #{pr_number}](https://github.com/ZEN-universe/ZEN-garden/pull/{pr_number}) @{pr_author}]"

    # initialize changelog additions
    changelog_addition = ""

    # add version number if bump
    if semver_bump != "none":
        changelog_addition += f"\n## [v{new_version}] - {date.today()} \n"
    else:
        changelog_addition += f"\n## [Unversioned Changes] - {date.today()} \n"

    type_labels = {
        "feat": "New Features ✨",
        "fix": "Bug Fixes 🐛",
        "docs": "Documentation Changes 📝",
        "chore": "Maintainance Tasks 🧹",
        "breaking": "BREAKING CHANGES ⚠️💥"
    }

    for change_type, items in categorized_changes.items():
        if items:
            changelog_addition += f"\n### {type_labels[change_type]}\n"
            for item in items:
                changelog_addition += f"- {item} {pr_info}\n"
    
    return header + changelog_addition + changelog_existing

if __name__ == "__main__":

    changelog_file = Path("CHANGELOG.md")
    pr_body_file = Path("pr_body.txt") # saved in github action
    pyproject_toml_file = Path("pyproject.toml")

    pr_author, pr_number, pr_body = extract_pr_info(pr_body_file)
    categorized_changes = parse_changes_from_pr_body(pr_body)

    semver_bump = determine_bump_type(categorized_changes)

    new_version, old_version = bumpversion(semver_bump, pyproject_toml_file)

    header, changelog_existing = parse_changelog(changelog_file)

    changelog = update_changelog(header, categorized_changes, changelog_existing, semver_bump, pr_number, pr_author)

    # Save new changelog file
    with open(changelog_file, "w", encoding="utf-8") as f:
        f.write(changelog)

    # Write to the GitHub Actions environment file
    # with open(os.environ["GITHUB_ENV"], "a") as env_file:
    #     env_file.write(f"NEW_VERSION={new_version}\n")
