"""Read-only structural check of one completed or paused project request.

Checks files and ontology links, not scientific truth, test execution, or whether
an acceptance quote actually came from a person. Pass every new/edited graph note
with --note; unrelated legacy tasks and notes are not validated recursively.
An analytical result format is a declaration, not proof of analytical work.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import sys

try:
    import yaml
except ImportError:
    yaml = None

MAX_TEXT_BYTES = 2 * 1024 * 1024
MAX_DOCUMENTS = 2000
MAX_ERRORS = 20
WIKI = re.compile(r"\[\[([^\]]+)\]\]")
DONE = {"выполнена", "принята"}
STATUSES = {"открыта", "в работе", "выполнена", "заблокирована", "принята"}


@dataclass
class Document:
    path: Path
    front: dict
    text: str
    title: str
    links: list[tuple[str, str]]
    link_errors: list[str]


class WorkflowCheck:
    def __init__(self, root: Path):
        self.root = root.resolve(strict=True)
        self.errors: list[str] = []
        self.documents: dict[Path, Document] = {}
        self.invalid: dict[Path, str] = {}
        self.names: dict[str, set[Path]] = {}
        self.classes: dict = {}
        self.properties: dict = {}

    def error(self, message: str) -> None:
        if message not in self.errors:
            self.errors.append(message)

    def safe_path(self, value: str | Path) -> Path:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Путь должен быть относительным и внутри проекта: {value}")
        target = self.root / path
        for item in (target, *target.parents):
            if item == self.root:
                break
            if item.is_symlink():
                raise ValueError(f"Символические ссылки не разрешены: {value}")
        if not target.resolve().is_relative_to(self.root):
            raise ValueError(f"Путь выходит из проекта: {value}")
        return target

    def read_text(self, path: Path) -> str:
        self.safe_path(path.relative_to(self.root))
        if not path.is_file():
            raise ValueError(f"Нет файла: {path.relative_to(self.root)}")
        if path.stat().st_size > MAX_TEXT_BYTES:
            raise ValueError(f"Слишком большой текстовый файл: {path.relative_to(self.root)}")
        return path.read_text(encoding="utf-8-sig")

    @staticmethod
    def clean_link(value: str) -> str:
        return value.split("|", 1)[0].split("#", 1)[0].strip()

    @staticmethod
    def section(text: str, title: str) -> str:
        match = re.search(r"^##\s+" + re.escape(title) + r"\s*$", text, re.M)
        if not match:
            return ""
        rest = text[match.end():]
        following = re.search(r"^#{1,2}\s+", rest, re.M)
        return rest[:following.start() if following else len(rest)].strip()

    def load(self, path: Path) -> Document:
        if path in self.documents:
            return self.documents[path]
        if path in self.invalid:
            raise ValueError(self.invalid[path])
        try:
            text = self.read_text(path)
            match = re.match(r"\A---\s*\n(.*?)\n---\s*(?:\n|$)", text, re.S)
            if not match:
                raise ValueError("Нет YAML frontmatter")
            front = yaml.safe_load(match.group(1))
            if not isinstance(front, dict):
                raise ValueError("Некорректный YAML frontmatter")
            title_match = re.search(r"^#\s+(.+)$", text, re.M)
            title = title_match.group(1).strip() if title_match else path.stem
            links, bad = [], []
            for line in self.section(text, "Связи").splitlines():
                line = line.strip()
                if not line or line.startswith("<!--"):
                    continue
                relation = re.match(r"^-\s+([^:]+):\s*(.+)$", line)
                targets = WIKI.findall(relation.group(2)) if relation else []
                if not relation or not targets:
                    bad.append("Неверная строка в разделе Связи: " + line[:100])
                    continue
                links.extend((relation.group(1).strip(), self.clean_link(target)) for target in targets)
            doc = Document(path, front, text, title, links, bad)
            self.documents[path] = doc
            return doc
        except (OSError, UnicodeError, ValueError, yaml.YAMLError) as exc:
            message = f"{path.relative_to(self.root)}: {exc}"
            self.invalid[path] = message
            raise ValueError(message) from exc

    def index(self) -> None:
        ontology = yaml.safe_load(self.read_text(self.safe_path("ontology.yaml")))
        if not isinstance(ontology, dict) or not isinstance(ontology.get("classes"), dict):
            raise ValueError("ontology.yaml: нет словаря classes")
        self.classes = ontology["classes"]
        self.properties = ontology.get("properties", {})
        if not isinstance(self.properties, dict):
            raise ValueError("ontology.yaml: properties должен быть словарём")
        count = 0
        for folder in ("tasks", "notes", "reports", "ontology"):
            base = self.root / folder
            if not base.exists():
                continue
            self.safe_path(folder)
            for path in base.rglob("*.md"):
                count += 1
                if count > MAX_DOCUMENTS:
                    raise ValueError(f"Более {MAX_DOCUMENTS} заметок: сузь/раздели хранилище")
                relative = path.relative_to(self.root)
                try:
                    self.safe_path(relative)
                    doc = self.load(path)
                except ValueError:
                    continue  # Unrelated legacy errors do not block this request.
                aliases = doc.front.get("aliases", [])
                if isinstance(aliases, str):
                    aliases = [aliases]
                if not isinstance(aliases, list):
                    aliases = []
                for name in (path.stem, *aliases):
                    if isinstance(name, str):
                        self.names.setdefault(name, set()).add(path)

    def resolve(self, link: str) -> Path:
        link = self.clean_link(link)
        direct = self.safe_path(link)
        candidates = [direct]
        if direct.suffix.lower() != ".md":
            candidates.append(Path(str(direct) + ".md"))
        existing = []
        for candidate in candidates:
            checked = self.safe_path(candidate.relative_to(self.root))
            if checked.is_file():
                existing.append(checked)
        if len(existing) > 1:
            raise ValueError(f"Неоднозначная ссылка [[{link}]]")
        if existing:
            return existing[0]
        matches = self.names.get(link.removesuffix(".md"), set())
        if len(matches) != 1:
            raise ValueError(f"{'Неоднозначная' if matches else 'Не найдена'} ссылка [[{link}]]")
        return next(iter(matches))

    def is_a(self, child: str, parent: str) -> bool:
        if not isinstance(child, str) or not isinstance(parent, str):
            return False
        pending, seen = [child], set()
        while pending:
            item = pending.pop()
            if item == parent:
                return True
            if item in seen:
                continue
            seen.add(item)
            entry = self.classes.get(item, {})
            parents = entry.get("parent") if isinstance(entry, dict) else None
            pending.extend(p for p in (parents if isinstance(parents, list) else [parents]) if isinstance(p, str))
        return False

    def accepts(self, actual: str, expected) -> bool:
        expected = expected if isinstance(expected, list) else [expected]
        return any(isinstance(name, str) and self.is_a(actual, name) for name in expected)

    def target_type(self, path: Path) -> str:
        if path.suffix.lower() == ".md" and path.parts[len(self.root.parts)] in ("tasks", "notes", "reports", "ontology"):
            return self.load(path).front.get("type", "")
        return "Файл"

    def validate_document(self, doc: Document) -> None:
        label = str(doc.path.relative_to(self.root))
        for message in doc.link_errors:
            self.error(f"{label}: {message}")
        class_note = doc.front.get("kind") == "класс"
        source_type = doc.path.stem if class_note else doc.front.get("type")
        if not isinstance(source_type, str) or source_type not in self.classes:
            self.error(f"{label}: неизвестный класс {source_type!r}")
            source_type = ""
        assertions = [target for name, target in doc.links if name == "тип"]
        if not class_note and assertions != [source_type]:
            self.error(f"{label}: нужна одна связь тип: [[{source_type}]]")
        if source_type:
            card = self.safe_path(f"ontology/{source_type}.md")
            if not card.is_file():
                self.error(f"{label}: отсутствует карточка класса ontology/{source_type}.md")
        if class_note:
            entry = self.classes.get(source_type, {})
            parent = entry.get("parent") if isinstance(entry, dict) else None
            expected = parent if isinstance(parent, list) else [parent] if parent else []
            actual = [target for relation, target in doc.links if relation == "подклассОт"]
            if sorted(actual) != sorted(expected):
                self.error(f"{label}: набор подклассОт не совпадает с ontology.yaml")
        for relation, target in doc.links:
            if relation == "тип":
                continue
            if relation == "подклассОт" and class_note:
                entry = self.classes.get(source_type, {})
                parent = entry.get("parent") if isinstance(entry, dict) else None
                parents = parent if isinstance(parent, list) else [parent]
                if target not in parents:
                    self.error(f"{label}: подклассОт не совпадает с ontology.yaml")
                continue
            prop = self.properties.get(relation)
            if not isinstance(prop, dict):
                self.error(f"{label}: неизвестное отношение {relation}")
                continue
            if not self.accepts(source_type, prop.get("domain")):
                self.error(f"{label}: класс {source_type} не входит в domain отношения {relation}")
            try:
                path = self.resolve(target)
                actual = self.target_type(path)
                if not self.accepts(actual, prop.get("range")):
                    self.error(f"{label}: цель [[{target}]] типа {actual!r} не входит в range отношения {relation}")
            except ValueError as exc:
                self.error(f"{label}: {exc}")

    def run(self, task: str, *, min_subtasks: int = 0, notes: list[str] | None = None) -> list[str]:
        try:
            self.index()
            root_path = self.safe_path(task)
            if not root_path.is_relative_to(self.root / "tasks") or root_path.suffix != ".md":
                raise ValueError("--task должен указывать на Markdown-файл внутри tasks/")
            root_doc = self.load(root_path)
            children: dict[Path, set[Path]] = {}
            for path, doc in list(self.documents.items()):
                if not path.is_relative_to(self.root / "tasks"):
                    continue
                for relation, target in doc.links:
                    if relation == "подзадачаОт":
                        try:
                            children.setdefault(self.resolve(target), set()).add(path)
                        except ValueError:
                            pass
            # A broken frontmatter must not hide a child whose body still has a
            # valid parent link. Unrelated malformed legacy tasks remain outside
            # scope, and no untrusted target is read before path validation.
            for path in self.invalid:
                if not path.is_relative_to(self.root / "tasks"):
                    continue
                try:
                    links = self.section(self.read_text(path), "Связи")
                    for match in re.finditer(r"^-\s*подзадачаОт\s*:\s*(.+)$", links, re.M):
                        for target in WIKI.findall(match.group(1)):
                            try:
                                children.setdefault(self.resolve(target), set()).add(path)
                            except ValueError:
                                pass
                except ValueError:
                    continue
            selected, pending = set(), [root_path]
            while pending:
                path = pending.pop()
                if path in selected:
                    continue
                selected.add(path)
                pending.extend(children.get(path, set()))
            colors = {}
            stack = [(root_path, False)]
            while stack:
                path, leaving = stack.pop()
                if leaving:
                    colors[path] = 2
                    continue
                if colors.get(path) == 1:
                    self.error("Цикл подзадачаОт в текущей задаче")
                    continue
                if colors.get(path) == 2:
                    continue
                colors[path] = 1
                stack.append((path, True))
                stack.extend((child, False) for child in children.get(path, set()))
            if len(children.get(root_path, set())) < min_subtasks:
                self.error(f"Нужно не менее {min_subtasks} прямых подзадач текущей задачи")
            root_status = root_doc.front.get("status")
            if not isinstance(root_status, str):
                root_status = ""
            if root_status not in DONE | {"заблокирована"}:
                self.error("Текущая задача перед финальным ответом должна быть выполнена или заблокирована")
            complete = root_status in DONE
            for path in selected:
                doc = self.load(path)
                self.validate_document(doc)
                label = str(path.relative_to(self.root))
                for field in ("Готово, когда", "Не входит"):
                    if not re.search(r"^[ \t]*(?:-[ \t]*)?" + re.escape(field) + r"[ \t]*:[ \t]*\S[^\n]*$", doc.text, re.M):
                        self.error(f"{label}: нужно непустое поле {field}: ...")
                if path == root_path and not re.search(r"^[ \t]*(?:-[ \t]*)?Запрос пользователя[ \t]*:[ \t]*\S[^\n]*$", doc.text, re.M):
                    self.error(f"{label}: нужно непустое поле Запрос пользователя: ...")
                if children.get(path):
                    plan = self.section(doc.text, "План")
                    if not plan:
                        self.error(f"{label}: для родительской задачи нужен непустой раздел План")
                    else:
                        rows = [re.sub(r"^[ \t]*(?:[-*+][ \t]+|\d+[.)][ \t]+)", "", line).strip() for line in plan.splitlines() if line.strip()]
                        for child in children[path]:
                            child_doc = self.load(child)
                            choices = (child.name, child.stem, child_doc.title, child.relative_to(self.root).as_posix())
                            if not any(row == choice or any(row.startswith(choice + separator) for separator in (" — ", " - ", ": ")) for row in rows for choice in choices):
                                self.error(f"{label}: План не содержит подзадачу {child.name}")
                if not self.is_a(doc.front.get("type"), "Задача"):
                    self.error(f"{label}: это не задача")
                status = doc.front.get("status")
                if not isinstance(status, str):
                    status = ""
                if status not in STATUSES:
                    self.error(f"{label}: неизвестный статус {status!r}")
                if "result_kind" in doc.front:
                    result_kind = doc.front["result_kind"]
                    if not isinstance(result_kind, str) or result_kind not in ("анализ", "артефакт"):
                        self.error(f"{label}: result_kind должен быть анализ или артефакт")
                if complete and status not in DONE:
                    self.error(f"{label}: потомок выполненной задачи ещё не выполнен")
                if status in DONE:
                    for child in children.get(path, set()):
                        child_status = self.load(child).front.get("status")
                        if not isinstance(child_status, str) or child_status not in DONE:
                            self.error(f"{label}: выполненный родитель имеет невыполненную подзадачу {child.name}")
                if status == "в работе":
                    self.error(f"{label}: начатая подзадача оставлена в работе перед финальным ответом")
                if status in DONE | {"заблокирована"}:
                    outcome = self.section(doc.text, "Итог")
                    if not outcome:
                        self.error(f"{label}: нет непустого раздела Итог")
                    if status == "заблокирована" and not re.search(r"Причина\s*:\s*\S", outcome, re.I):
                        self.error(f"{label}: для блокировки нужна строка Причина: ... в Итоге")
                    if status == "принята" and not re.search(r"Принято пользователем\s*:\s*[«\"']\s*[^»\"']+\s*[»\"']", outcome):
                        self.error(f"{label}: для принятия нужна цитата Принято пользователем: «...»")
                if path in children.get(path, set()):
                    self.error(f"{label}: задача не может быть своей подзадачей")
            # Scope is explicit edited notes + notes closing selected tasks. Link
            # targets are read for type/existence only, never recursively checked.
            graph = set()
            explicit_notes = set()
            for name in notes or []:
                explicit_notes.add(self.safe_path(name))
            graph.update(explicit_notes)
            closures: dict[Path, set[Path]] = {}
            for path, doc in list(self.documents.items()):
                if path.is_relative_to(self.root / "tasks"):
                    continue
                for relation, target in doc.links:
                    if relation not in ("решает", "выполняет"):
                        continue
                    try:
                        resolved = self.resolve(target)
                    except ValueError:
                        continue
                    if resolved in selected:
                        graph.add(path)
                        closures.setdefault(resolved, set()).add(path)
            for path in selected:
                doc = self.load(path)
                if isinstance(doc.front.get("status"), str) and doc.front["status"] in DONE and not children.get(path):
                    if doc.front.get("result_kind") == "анализ":
                        for title in ("Результат анализа", "Проверка анализа"):
                            if not self.section(doc.text, title):
                                self.error(f"{path.relative_to(self.root)}: для аналитического результата нужен непустой раздел {title}")
                    elif not closures.get(path):
                        self.error(f"{path.relative_to(self.root)}: выполненную конечную задачу не закрывает Проверка/Эксперимент/Отчёт")
            for path in graph:
                if path.suffix != ".md":
                    self.error(f"--note должен указывать на Markdown: {path.relative_to(self.root)}")
                else:
                    self.validate_document(self.load(path))
            self.validate_diary(root_path, explicit_notes)
        except (OSError, ValueError, UnicodeError, yaml.YAMLError) as exc:
            self.error(str(exc))
        return self.errors

    def validate_diary(self, task: Path, explicit_notes: set[Path]) -> None:
        text = self.read_text(self.safe_path("diary.md"))
        headings = list(re.finditer(r"^##\s+.+$", text, re.M))
        if not headings:
            self.error("diary.md: нет записи с заголовком ##")
            return
        latest = text[headings[-1].end():]
        fields = {}
        for match in re.finditer(r"^[ \t]*-[ \t]*(Задача|Сделано|Проверка|Граф|Дальше)[ \t]*:[ \t]*(.*?)[ \t]*$", latest, re.M):
            fields[match.group(1)] = match.group(2)
        for key in ("Задача", "Сделано", "Проверка", "Граф", "Дальше"):
            if not fields.get(key, "").strip():
                self.error(f"diary.md: в последней записи нет непустого поля {key}")
        targets = WIKI.findall(fields.get("Задача", ""))
        matches = False
        for target in targets:
            try:
                matches |= self.resolve(target) == task
            except ValueError:
                pass
        if not matches:
            self.error("diary.md: последняя запись должна ссылаться на текущую задачу в поле Задача")
        graph_targets = set()
        for target in WIKI.findall(fields.get("Граф", "")):
            try:
                path = self.resolve(target)
                graph_targets.add(path)
                if path.suffix == ".md" and path.parts[len(self.root.parts)] in ("tasks", "notes", "reports", "ontology"):
                    doc = self.load(path)
                    actual_type = doc.path.stem if doc.front.get("kind") == "класс" else doc.front.get("type")
                else:
                    actual_type = "Файл"
                if not isinstance(actual_type, str) or actual_type not in self.classes:
                    self.error(f"diary.md: неизвестный класс цели Граф [[{target}]]")
            except ValueError as exc:
                self.error(f"diary.md: Граф: {exc}")
        for path in explicit_notes - graph_targets:
            self.error(f"diary.md: поле Граф не перечисляет изменённую заметку {path.relative_to(self.root)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Корень проверяемого проекта; по умолчанию текущая папка")
    parser.add_argument("--task", required=True, help="Текущая корневая задача tasks/<имя>.md")
    parser.add_argument("--min-subtasks", type=int, default=0)
    parser.add_argument("--note", action="append", default=[], help="Каждая новая/изменённая заметка текущей работы")
    args = parser.parse_args(argv)
    if args.min_subtasks < 0:
        parser.error("--min-subtasks не может быть отрицательным")
    if yaml is None:
        print("Нужен PyYAML из существующей среды проекта; пакеты автоматически не устанавливаются.", file=sys.stderr)
        return 2
    try:
        check = WorkflowCheck(args.root)
        errors = check.run(args.task, min_subtasks=args.min_subtasks, notes=args.note)
    except (OSError, ValueError) as exc:
        print(f"Не удалось открыть проект: {exc}", file=sys.stderr)
        return 2
    if errors:
        print(f"Протокол не завершён: {len(errors)} замечаний.")
        for error in errors[:MAX_ERRORS]:
            print("- " + " ".join(error.split())[:200])
        if len(errors) > MAX_ERRORS:
            print(f"Ещё замечаний: {len(errors) - MAX_ERRORS}.")
        return 1
    print("Структурная проверка текущей задачи, связей и дневника пройдена.")
    print("Научная корректность, фактическое выполнение команд и выбор аналитического формата этой проверкой не подтверждаются.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
