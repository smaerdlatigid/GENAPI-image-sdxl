#!/usr/bin/env python3

import json
import os
import subprocess

def add_safe_directory(path):
    """Add a directory to Git's safe.directory configuration."""
    subprocess.run(["git", "config", "--global", "--add", "safe.directory", path])

json_file = "custom_nodes.json"
comfy_dir = "ComfyUI"
custom_nodes_dir = f"{comfy_dir}/custom_nodes/"

# check if the comfy directory exists
if not os.path.isdir(custom_nodes_dir):
    raise FileNotFoundError(
        f"Directory {custom_nodes_dir} does not exist. Make sure you have the correct path to the ComfyUI directory."
    )

# Read the JSON file containing the repositories and commit hashes
with open(json_file, "r") as file:
    repos = json.load(file)

# Disable the detached head warning
subprocess.run(["git", "config", "--global", "advice.detachedHead", "false"])

# Loop over each repository in the list
for repo in repos:
    repo_url = repo["repo"]
    commit_hash = repo["commit"]
    repo_name = os.path.basename(repo_url.replace(".git", ""))

    # Check if the repository directory already exists
    repo_path = os.path.join(custom_nodes_dir, repo_name)
    absolute_repo_path = os.path.abspath(repo_path)

    # Add the repository path to safe.directory
    add_safe_directory(absolute_repo_path)

    if not os.path.isdir(repo_path):
        # Clone the repository into the destination directory
        print(
            f"Cloning {repo_url} into {repo_path} and checking out to commit {commit_hash}"
        )
        subprocess.run(["git", "clone", "--recursive", repo_url, repo_path])

        # Store the current directory and change to the repository's directory
        current_dir = os.getcwd()
        os.chdir(repo_path)
        subprocess.run(["git", "checkout", commit_hash])
        subprocess.run(["git", "submodule", "update", "--init", "--recursive"])

        # Change back to the original directory after operations
        os.chdir(current_dir)

    # If the repository already exists, check if the commit hash is the same
    else:
        current_dir = os.getcwd()
        os.chdir(repo_path)

        current_commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        )
        print(f"Custom node installed for {repo_name} at {current_commit[:7]}")

        if current_commit[:7] != commit_hash[:7]:
            response = input(
                f"Do you want to update {repo_name}? Current ({current_commit[:7]}) is different from ({commit_hash[:7]}) (y/n): "
            )
            if response.lower() == "y":
                print(f"Checking out to commit {commit_hash}")
                subprocess.run(["git", "fetch"])
                subprocess.run(["git", "checkout", commit_hash])
                subprocess.run(["git", "submodule", "update", "--init", "--recursive"])
            else:
                print("Skipping checkout, keeping current commit")

        os.chdir(current_dir)

# Copy custom node config files to the correct directory
config_files = {
    "was_suite_config": {
        "src": "custom_node_configs/was_suite_config.json",
        "dest": os.path.join(custom_nodes_dir, "was-node-suite-comfyui/"),
    },
    "rgthree_config": {
        "src": "custom_node_configs/rgthree_config.json",
        "dest": os.path.join(custom_nodes_dir, "rgthree-comfy/"),
    },
    "comfy_settings": {
        "src": "custom_node_configs/comfy.settings.json",
        "dest": os.path.join(comfy_dir, "user", "default"),
    },
}

if "comfy_settings" in config_files:
    paths = config_files["comfy_settings"]
    if not os.path.exists(paths["dest"]):
        os.makedirs(paths["dest"])

for config_file, paths in config_files.items():
    if (
        os.path.isfile(paths["src"])
        and os.path.isdir(paths["dest"])
        and not os.path.exists(
            os.path.join(paths["dest"], os.path.basename(paths["src"]))
        )
    ):
        print(f"Copying {config_file} to {paths['dest']}")
        subprocess.run(["cp", paths["src"], paths["dest"]])