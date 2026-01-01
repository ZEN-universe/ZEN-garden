import os
import re
from datetime import date
from importlib.metadata import version as get_version


#pr_body_file = os.environ.get("PR_BODY_FILE", "pr_body.txt")
pr_body = os.getenv("PR_BODY")
print(pr_body)
zen_garden_version = "v" + get_version("zen_garden")
changelog_file = "CHANGELOG.md"

# # Read PR body
# pr_body_file = "./.github/pull_request_template.md"
# with open(pr_body_file, "r") as f:
#     pr_body = f.read()

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

# Determine semantic version bump
if categorized_changes["breaking"]:
    semver_bump = "major"
elif categorized_changes["feat"]:
    semver_bump = "minor"
elif categorized_changes["fix"]:
    semver_bump = "patch"
else:
    semver_bump = "none"

# Read existing changelog (if exists)
if os.path.exists(changelog_file):
    with open(changelog_file, "r", encoding="utf-8") as f:
        changelog = f.read()

# Remove preface:
match = re.search("(#.*?)(\n##.*$)", changelog, re.DOTALL)
header = match.group(1)
changelog_existing = match.group(2)

# Append changes to CHANGELOG.md --------------------------------------

# initialize changelog additions
changelog_addition = ""

# add version number if bump
if semver_bump != "none":
    changelog_addition += f"\n## {zen_garden_version}\n"

# add date 
changelog_addition += f"\n### [{date.today()}]\n"

type_labels = {
    "feat": "Feature",
    "fix": "Fix",
    "docs": "Documenataion",
    "chore": "Chores",
    "breaking": "BREAKING CHANGES"
}

for change_type, items in categorized_changes.items():
    if items:
        changelog_addition += f"\n#### {type_labels[change_type]}\n"
        for item in items:
            changelog_addition += f"- {item}\n"

# Save new changelog file
with open(changelog_file, "w", encoding="utf-8") as f:
    f.write(header + changelog_addition + changelog_existing)

# Write to the GitHub Actions environment file
# with open(os.environ["GITHUB_ENV"], "a") as env_file:
#     env_file.write(f"NEW_VERSION={new_version}\n")
