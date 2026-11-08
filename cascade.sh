#!/usr/bin/env bash
# Каскадное сжатие статьи учебным агентом: L1 (по частям) -> L2 (тезисы) -> L3 (одна строка).
# Каждая часть сжимается отдельным запуском `ai4science run`, то есть в своём окне контекста;
# следующий уровень читает только файлы предыдущего.
#
# Запуск из корня проекта (macOS, Linux, WSL):
#     bash cascade.sh paper/paper.pdf
# Результат:
#     memory/<статья>/<статья>.txt    текст; страницы разделены символом \f
#     memory/<статья>/parts/part_NN.md части по WORDS_PER_PART слов (по умолчанию 1500), с метками [с. N];
#                                    если есть memory/<статья>/sections/*.md (ручные разделы),
#                                    сжимаются они
#     memory/<статья>/L1/<часть>.md   сжатие каждой части
#     memory/<статья>/L1.md, L2.md, L3.md
# <статья> — имя переданного PDF без пути и расширения .pdf.
# Повторный запуск пропускает уже сжатые части этой статьи.
# Чтобы начать заново, удалите parts/ и L1/ только в папке нужной статьи.
# Сверку в memory/<статья>/L2.md заполняет человек.
set -euo pipefail

PDF="${1:?укажите PDF статьи: bash cascade.sh paper/paper.pdf}"
WORDS_PER_PART="${WORDS_PER_PART:-1500}"
export AI4SCIENCE_WEB_DISABLED=1   # поиск в сети не нужен; Lean отключает флаг --no-lean

PDF_NAME="${PDF##*/}"
ARTICLE="${PDF_NAME%.[pP][dD][fF]}"
case "$ARTICLE" in
  ''|.|..) echo "Не удалось определить имя статьи по пути: $PDF" >&2; exit 1 ;;
esac
WORK_DIR="memory/$ARTICLE"
DEFAULT_PARTS="$WORK_DIR/parts"
SECTIONS="$WORK_DIR/sections"
L1_DIR="$WORK_DIR/L1"
L1_SUMMARY="$WORK_DIR/L1.md"
L2_SUMMARY="$WORK_DIR/L2.md"
L3_SUMMARY="$WORK_DIR/L3.md"
mkdir -p "$L1_DIR"

# 1. Текст статьи по страницам
TXT="$WORK_DIR/$ARTICLE.txt"
if command -v pdftotext >/dev/null 2>&1; then
  pdftotext "$PDF" "$TXT"
else
  python3 - "$PDF" "$TXT" <<'PY'
import sys
from pypdf import PdfReader  # python3 -m pip install pypdf
pages = [p.extract_text() or "" for p in PdfReader(sys.argv[1]).pages]
open(sys.argv[2], "w", encoding="utf-8").write("\f".join(pages))
PY
fi

# 2. Части: ручные разделы этой статьи или страницы, в сумме около WORDS_PER_PART слов
if ls "$SECTIONS"/*.md >/dev/null 2>&1; then
  PARTS="$SECTIONS"
else
  PARTS="$DEFAULT_PARTS"
  mkdir -p "$PARTS"
fi
if [ "$PARTS" = "$DEFAULT_PARTS" ] && ! ls "$PARTS"/part_*.md >/dev/null 2>&1; then
  python3 - "$TXT" "$WORDS_PER_PART" "$PARTS" <<'PY'
import sys
from pathlib import Path
pages = open(sys.argv[1], encoding="utf-8").read().split("\f")
limit = int(sys.argv[2])
parts_dir = Path(sys.argv[3])
parts, cur, n = [], [], 0
for i, text in enumerate(pages, start=1):
    if not text.strip():
        continue
    cur.append((i, text))
    n += len(text.split())
    if n >= limit:
        parts.append(cur)
        cur, n = [], 0
if cur:
    parts.append(cur)
for k, part in enumerate(parts, start=1):
    first, last = part[0][0], part[-1][0]
    with (parts_dir / f"part_{k:02d}.md").open("w", encoding="utf-8") as f:
        f.write(f"# Часть {k}: с. {first}–{last}\n\n")
        for i, text in part:
            f.write(f"[с. {i}]\n{text.strip()}\n\n")
print(f"частей: {len(parts)}")
PY
fi

# 3. L1: каждая часть — в отдельном запуске агента
for f in "$PARTS"/*.md; do
  out="$L1_DIR/$(basename "$f")"
  if [ -s "$out" ]; then echo "уже есть: $out"; continue; fi
  ai4science --no-lean run "Прочитай файл \"$f\" — часть научной статьи; метки [с. N] — номера страниц. \
Сожми эту часть до 100–150 слов по-русски: главное утверждение, метод, результат. \
Числа, формулы и названия — дословно; после каждого числа и утверждения — страница в виде (с. N). \
Первая строка — заголовок '## <тема части> (с. a–b)'. Запиши результат в файл \"$out\" и больше ничего не делай."
  [ -s "$out" ] || echo "ВНИМАНИЕ: агент не записал $out — запустите скрипт ещё раз"
done

{
  echo "# L1 — реферат по частям"
  echo
  echo "Статья: <авторы, «название», издание, год; DOI или arXiv с номером версии>."
  echo "Страницы — по файлу $PDF."
  echo
  for f in "$PARTS"/*.md; do cat "$L1_DIR/$(basename "$f")" 2>/dev/null; echo; done
} > "$L1_SUMMARY"

# 4. L2: тезисы по L1, в новом окне
ai4science --no-lean run "Прочитай \"$L1_SUMMARY\". Составь от 5 до 10 тезисов всей статьи по-русски. \
Каждый тезис — одно проверяемое утверждение: результат с числом, условие, ограничение; \
в конце тезиса — страница (с. N) из L1. \
Сфокусируйся на параметрах ВРЭПР-спектров, описании спиновых плотностей, временах процессов, описании донора,\
акцептора и тезисах о переносе и разделении заряда.\
Запиши в \"$L2_SUMMARY\": первая строка '# L2 — тезисы', \
пустая строка, затем нумерованный список. Больше ничего в файл не пиши."

if [ -s "$L2_SUMMARY" ] && ! grep -q '^## Сверка' "$L2_SUMMARY"; then
  cat >> "$L2_SUMMARY" <<'EOF'

## Сверка с оригиналом

Не меньше трёх тезисов сверяются по PDF, не по L1: страница, дословная цитата, итог.
Итог — «подтверждено»; «расходится: …; исправлено» — что именно расходилось;
«нет в тексте; тезис удалён».

| № | Тезис | Стр. | Цитата из оригинала | Итог |
|---|---|---|---|---|
| 1 | <номер тезиса> | <N> | "<дословная цитата>" | <итог> |
| 2 | <номер тезиса> | <N> | "<дословная цитата>" | <итог> |
| 3 | <номер тезиса> | <N> | "<дословная цитата>" | <итог> |
EOF
fi

# 5. L3: одна строка по L2, в новом окне
ai4science --no-lean run "Прочитай \"$L2_SUMMARY\", только нумерованный список тезисов. \
Сформулируй главный результат статьи одной фразой до 30 слов по-русски. \
Сфокусируйся на параметрах ВРЭПР-спектров, описании спиновых плотностей, временах процессов, описании донора,\
акцептора и тезисах о переносе и разделении заряда.\
Запиши в \"$L3_SUMMARY\": первая строка '# L3 — одна строка', пустая строка, затем эта фраза. Больше ничего не пиши."

echo
echo "Слов: статья $(wc -w < "$TXT"), L1 $(wc -w < "$L1_SUMMARY"), L2 $(wc -w < "$L2_SUMMARY"), L3 $(wc -w < "$L3_SUMMARY")"
echo "Дальше: заполните строку «Статья:» в $L1_SUMMARY и таблицу сверки в $L2_SUMMARY."
