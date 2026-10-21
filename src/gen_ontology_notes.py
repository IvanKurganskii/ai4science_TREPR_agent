#!/usr/bin/env python3
"""Generate ontology class notes from ontology.yaml."""
import yaml
import os

YAML_PATH = "/mnt/c/Users/kurga/OneDrive/Документы/YSDA/AI4Science/my_project/ontology.yaml"
NOTES_DIR = "/mnt/c/Users/kurga/OneDrive/Документы/YSDA/AI4Science/my_project/ontology"

with open(YAML_PATH, "r") as f:
    data = yaml.safe_load(f)

classes = data.get("classes", {})
properties = data.get("properties", {})

# Build reverse mapping: for each class, find properties where it's in domain
class_properties = {}
for cls_name in classes:
    class_properties[cls_name] = []

for prop_name, prop_info in properties.items():
    domain = prop_info.get("domain", [])
    if isinstance(domain, str):
        domain = [domain]
    for d in domain:
        if d in class_properties:
            class_properties[d].append((prop_name, prop_info.get("range", "")))

for cls_name, cls_info in classes.items():
    parent = cls_info.get("parent", "null")
    definition = cls_info.get("definition", "")
    
    props = class_properties.get(cls_name, [])
    relations_text = ""
    if props:
        relations_text = "\n".join(f"- {p[0]}: {p[1]}" for p in props)
    else:
        relations_text = "(нет отношений)"
    
    # Связи (подкласс)
    if parent and parent != "null":
        links = f"- подклассОт: [[{parent}]]"
    else:
        links = ""
    
    content = f"""---
kind: класс
---
# {cls_name}

{definition}

Отношения от этого класса: {relations_text}.

## Связи
{links}
"""
    filepath = os.path.join(NOTES_DIR, f"{cls_name}.md")
    with open(filepath, "w") as f:
        f.write(content)

print(f"Created {len(classes)} ontology class notes.")