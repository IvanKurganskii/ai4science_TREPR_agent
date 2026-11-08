#!/usr/bin/env python3
"""Generate selected ontology class cards, or compare them without writing.

Attributes document ontology.yaml. Printing a datatype or unit does not implement
a numeric, physical, or semantic validator for instance notes.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

try:
    import yaml
except ImportError:
    yaml = None


def names(value, field: str) -> list[str]:
    if value is None or value == "null":
        return []
    result = [value] if isinstance(value, str) else value
    if not isinstance(result, list) or any(not isinstance(item, str) or not item for item in result):
        raise ValueError(f"{field}: ожидается имя класса или список имён")
    return result


def render_cards(data: dict, selected: list[str] | None = None) -> dict[str, str]:
    """Return generated text only; filesystem changes happen in main()."""
    if not isinstance(data, dict) or not isinstance(data.get("classes"), dict):
        raise ValueError("ontology.yaml: нужен словарь classes")
    classes = data["classes"]
    for name, info in classes.items():
        if (not isinstance(name, str) or not name or name in (".", "..")
                or any(c in name for c in '/\\:<>"|?*') or any(ord(c) < 32 for c in name)
                or name.rstrip(" .") != name or not isinstance(info, dict)):
            raise ValueError(f"Недопустимое имя или описание класса: {name!r}")
        reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
        if name.split(".", 1)[0].upper() in reserved:
            raise ValueError(f"Имя класса зарезервировано Windows: {name}")
    selected = list(dict.fromkeys(selected)) if selected else list(classes)
    unknown = [name for name in selected if name not in classes]
    if unknown:
        raise ValueError("Неизвестные классы: " + ", ".join(unknown))

    parents = {name: names(info.get("parent"), f"{name}.parent") for name, info in classes.items()}
    ancestors: dict[str, set[str]] = {}

    def collect(name: str, active: set[str]) -> set[str]:
        if name in ancestors:
            return ancestors[name]
        if name in active:
            raise ValueError(f"Цикл наследования: {name}")
        active = active | {name}
        found = {name}
        for parent in parents[name]:
            if parent not in classes:
                raise ValueError(f"{name}: неизвестный надкласс {parent}")
            found.update(collect(parent, active))
        ancestors[name] = found
        return found

    for name in classes:
        collect(name, set())

    sections: dict[str, list[tuple[str, dict, set[str]]]] = {}
    for section in ("properties", "attributes"):
        values = data.get(section, {})
        if not isinstance(values, dict):
            raise ValueError(f"{section}: нужен словарь")
        sections[section] = []
        for name, info in values.items():
            if not isinstance(name, str) or not isinstance(info, dict):
                raise ValueError(f"{section}: некорректное свойство {name!r}")
            domain = set(names(info.get("domain"), f"{section}.{name}.domain"))
            missing = domain - classes.keys()
            if missing:
                raise ValueError(f"{section}.{name}: неизвестные классы domain: {', '.join(sorted(missing))}")
            if section == "attributes" and "datatype" in info and info["datatype"] not in ("number", "integer", "string"):
                raise ValueError(f"attributes.{name}: datatype должен быть number, integer или string")
            if section == "attributes" and "values" in info and not isinstance(info["values"], list):
                raise ValueError(f"attributes.{name}.values: нужен список")
            sections[section].append((name, info, domain))

    cards = {}
    for class_name in selected:
        relations, attributes = [], []
        for name, info, domain in sections["properties"]:
            if domain & ancestors[class_name]:
                target = info.get("range", "")
                targets = target if isinstance(target, list) else [target]
                relations.append(f"- {name}: {', '.join(str(item) for item in targets)}")
        for name, info, domain in sections["attributes"]:
            if not domain & ancestors[class_name]:
                continue
            details = []
            if "values" in info:
                details.append("значения: " + ", ".join(str(item) for item in info["values"]))
            if "datatype" in info:
                details.append("тип данных: " + info["datatype"])
            if "unit" in info:
                details.append("единица: " + str(info["unit"]))
            attributes.append(f"- {name}: {'; '.join(details) if details else '(ограничения не заданы)'}")
        relations_text = "\n".join(relations) if relations else "(нет отношений)"
        attributes_text = "\n".join(attributes) if attributes else "(нет атрибутов)"
        links = "\n".join(f"- подклассОт: [[{parent}]]" for parent in parents[class_name])
        definition = str(classes[class_name].get("definition", "")).strip()
        cards[class_name] = (
            f"---\nkind: класс\n---\n# {class_name}\n\n{definition}\n\n"
            f"Отношения от этого класса: {relations_text}.\n\n"
            f"Атрибуты: {attributes_text}.\n\n## Связи\n{links}\n"
        )
    return cards


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Корень проекта; по умолчанию текущая папка, при внешнем запуске указывай явно")
    parser.add_argument("--class", dest="classes", action="append", help="Класс для генерации; можно указать несколько раз, без опции — все классы")
    parser.add_argument("--check", action="store_true", help="Только проверить выбранные карточки, ничего не записывать")
    args = parser.parse_args(argv)
    if yaml is None:
        print("Нужен PyYAML из существующей среды проекта.", file=sys.stderr)
        return 2
    try:
        root = args.root.resolve(strict=True)
        source = root / "ontology.yaml"
        if source.is_symlink() or source.resolve().parent != root:
            raise ValueError("ontology.yaml должен быть обычным файлом внутри проекта")
        data = yaml.safe_load(source.read_text(encoding="utf-8-sig"))
        cards = render_cards(data, args.classes)
        directory = root / "ontology"
        if directory.is_symlink() or directory.resolve().parent != root:
            raise ValueError("ontology/ должен находиться внутри проекта и не быть ссылкой")
        pending = []
        for name, expected in cards.items():
            path = directory / f"{name}.md"
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError(f"Карточка не должна быть ссылкой за пределы проекта: {name}")
            current = path.read_text(encoding="utf-8-sig") if path.is_file() else None
            if current != expected:
                pending.append((path, expected, current is None))
        if args.check:
            for path, _, missing in pending:
                print(f"{'Отсутствует' if missing else 'Отличается'}: {path.relative_to(root)}")
            print(f"Проверено: {len(cards)}; замечаний: {len(pending)}.")
            return 1 if pending else 0
        if pending:
            directory.mkdir(exist_ok=True)
        for path, expected, _ in pending:
            path.write_text(expected, encoding="utf-8", newline="\n")
        print(f"Карточек: {len(cards)}; создано/обновлено: {len(pending)}; без изменений: {len(cards) - len(pending)}.")
        return 0
    except (OSError, ValueError, RecursionError, yaml.YAMLError) as error:
        print(f"Не удалось обработать онтологию: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
