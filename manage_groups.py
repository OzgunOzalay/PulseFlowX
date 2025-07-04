#!/usr/bin/env python3
"""
Utility script to manage subject group assignments.
"""

import json
import argparse
from pathlib import Path

def load_groups():
    """Load current group assignments."""
    group_file = Path("subject_groups.json")
    if group_file.exists():
        with open(group_file, 'r') as f:
            return json.load(f)
    else:
        return {"groups": {"AUD": [], "HC": []}, "metadata": {}}

def save_groups(data):
    """Save group assignments to file."""
    with open("subject_groups.json", 'w') as f:
        json.dump(data, f, indent=2)

def list_subjects():
    """List all subjects in Data directory."""
    data_dir = Path("Data")
    if not data_dir.exists():
        print("Data directory not found!")
        return []
    
    subjects = [d.name for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('sub-')]
    return sorted(subjects)

def show_current_groups():
    """Display current group assignments."""
    data = load_groups()
    groups = data.get("groups", {})
    
    print("=== CURRENT GROUP ASSIGNMENTS ===")
    for group_name, subjects in groups.items():
        print(f"\n{group_name} ({len(subjects)} subjects):")
        for subject in subjects:
            print(f"  - {subject}")
    
    # Show unassigned subjects
    all_subjects = set(list_subjects())
    assigned_subjects = set()
    for subjects in groups.values():
        assigned_subjects.update(subjects)
    
    unassigned = all_subjects - assigned_subjects
    if unassigned:
        print(f"\nUnassigned subjects ({len(unassigned)}):")
        for subject in sorted(unassigned):
            print(f"  - {subject}")

def add_subject_to_group(subject, group):
    """Add a subject to a specific group."""
    data = load_groups()
    groups = data.get("groups", {})
    
    if group not in groups:
        print(f"Error: Group '{group}' not found. Available groups: {list(groups.keys())}")
        return
    
    if subject not in groups[group]:
        groups[group].append(subject)
        groups[group].sort()  # Keep sorted
        save_groups(data)
        print(f"Added {subject} to {group} group")
    else:
        print(f"{subject} is already in {group} group")

def remove_subject_from_group(subject, group):
    """Remove a subject from a specific group."""
    data = load_groups()
    groups = data.get("groups", {})
    
    if group not in groups:
        print(f"Error: Group '{group}' not found")
        return
    
    if subject in groups[group]:
        groups[group].remove(subject)
        save_groups(data)
        print(f"Removed {subject} from {group} group")
    else:
        print(f"{subject} is not in {group} group")

def main():
    parser = argparse.ArgumentParser(description="Manage subject group assignments")
    parser.add_argument("--list", action="store_true", help="List current group assignments")
    parser.add_argument("--subjects", action="store_true", help="List all available subjects")
    parser.add_argument("--add", nargs=2, metavar=("SUBJECT", "GROUP"), help="Add subject to group")
    parser.add_argument("--remove", nargs=2, metavar=("SUBJECT", "GROUP"), help="Remove subject from group")
    
    args = parser.parse_args()
    
    if args.list:
        show_current_groups()
    elif args.subjects:
        subjects = list_subjects()
        print("Available subjects:")
        for subject in subjects:
            print(f"  - {subject}")
    elif args.add:
        subject, group = args.add
        add_subject_to_group(subject, group)
    elif args.remove:
        subject, group = args.remove
        remove_subject_from_group(subject, group)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 