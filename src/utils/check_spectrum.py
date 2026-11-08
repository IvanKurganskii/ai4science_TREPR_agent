#!/usr/bin/env python3
"""Проверка спектров на существование: не все нули, нет NaN/Inf."""

import sys
import os
import csv
import math
from pathlib import Path


def check_spectrum_file(filepath: str) -> dict:
    """Проверить один CSV-файл спектра.
    
    Возвращает словарь с результатами проверки.
    """
    result = {
        "file": filepath,
        "exists": os.path.isfile(filepath),
        "size_bytes": 0,
        "has_comments": False,
        "comment_lines": 0,
        "data_rows": 0,
        "data_cols": 0,
        "all_zero": False,
        "has_nan": False,
        "has_inf": False,
        "nonzero_count": 0,
        "total_count": 0,
        "nonzero_fraction": 0.0,
        "min_val": None,
        "max_val": None,
        "abs_max_val": None,
        "status": "unknown",
        "errors": []
    }
    
    if not result["exists"]:
        result["status"] = "missing"
        result["errors"].append("Файл не найден")
        return result
    
    result["size_bytes"] = os.path.getsize(filepath)
    
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        result["status"] = "error"
        result["errors"].append(f"Ошибка чтения: {e}")
        return result
    
    if len(lines) == 0:
        result["status"] = "empty"
        result["errors"].append("Файл пуст")
        return result
    
    # Подсчёт comment-строк
    comment_lines = 0
    data_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#'):
            comment_lines += 1
        elif stripped:
            data_lines.append(stripped)
    
    result["has_comments"] = comment_lines > 0
    result["comment_lines"] = comment_lines
    result["data_rows"] = len(data_lines)
    
    if len(data_lines) == 0:
        result["status"] = "no_data"
        result["errors"].append("Нет строк данных")
        return result
    
    # Парсинг данных
    values = []
    for i, line in enumerate(data_lines):
        try:
            reader = csv.reader([line], delimiter=',')
            row = next(reader)
            row_values = []
            for val in row:
                val = val.strip()
                if val == '':
                    continue
                num = float(val)
                row_values.append(num)
            values.extend(row_values)
        except (ValueError, StopIteration) as e:
            result["errors"].append(f"Ошибка парсинга строки {i+1}: {e}")
    
    result["total_count"] = len(values)
    
    if len(values) == 0:
        result["status"] = "no_numeric_data"
        result["errors"].append("Нет числовых данных")
        return result
    
    # Определение структуры: 2D или 1D
    # 2D: несколько строк с одинаковым числом колонок
    # 1D (slice): одна строка
    if len(data_lines) > 1:
        # Проверяем одинаковую длину строк
        row_lengths = []
        for line in data_lines:
            reader = csv.reader([line], delimiter=',')
            try:
                row = next(reader)
                row_lengths.append(len([v for v in row if v.strip()]))
            except StopIteration:
                row_lengths.append(0)
        
        if len(set(row_lengths)) == 1 and row_lengths[0] > 1:
            result["data_cols"] = row_lengths[0]
            result["is_2d"] = True
        else:
            result["is_2d"] = False
    else:
        result["is_2d"] = False
    
    # Проверки
    nonzero = sum(1 for v in values if v != 0.0)
    nan_count = sum(1 for v in values if math.isnan(v))
    inf_count = sum(1 for v in values if math.isinf(v))
    
    result["all_zero"] = (nonzero == 0)
    result["has_nan"] = (nan_count > 0)
    result["has_inf"] = (inf_count > 0)
    result["nonzero_count"] = nonzero
    result["nonzero_fraction"] = nonzero / len(values) if len(values) > 0 else 0.0
    result["min_val"] = min(values)
    result["max_val"] = max(values)
    result["abs_max_val"] = max(abs(v) for v in values)
    
    # Определение статуса
    if result["all_zero"]:
        result["status"] = "all_zero"
        result["errors"].append("Все значения равны нулю")
    if result["has_nan"]:
        result["status"] = "has_nan"
        result["errors"].append(f"Найдено {nan_count} NaN значений")
    if result["has_inf"]:
        result["status"] = "has_inf"
        result["errors"].append(f"Найдено {inf_count} Inf значений")
    
    if result["status"] == "unknown":
        result["status"] = "ok"
    
    return result


def check_directory(dirpath: str, recursive: bool = False) -> list:
    """Проверить все CSV-файлы в директории."""
    results = []
    csv_files = []
    
    if recursive:
        for root, dirs, files in os.walk(dirpath):
            for f in files:
                if f.endswith('.csv'):
                    csv_files.append(os.path.join(root, f))
    else:
        for f in os.listdir(dirpath):
            if f.endswith('.csv'):
                csv_files.append(os.path.join(dirpath, f))
    
    for filepath in sorted(csv_files):
        res = check_spectrum_file(filepath)
        results.append(res)
    
    return results


def print_summary(results: list):
    """Вывести краткую сводку."""
    total = len(results)
    ok = sum(1 for r in results if r["status"] == "ok")
    all_zero = sum(1 for r in results if r["status"] == "all_zero")
    has_nan = sum(1 for r in results if r["status"] == "has_nan")
    has_inf = sum(1 for r in results if r["status"] == "has_inf")
    missing = sum(1 for r in results if r["status"] == "missing")
    other = total - ok - all_zero - has_nan - has_inf - missing
    
    print(f"\n{'='*70}")
    print(f"Проверка спектров: {total} файлов")
    print(f"  OK: {ok}")
    print(f"  Все нули: {all_zero}")
    print(f"  NaN: {has_nan}")
    print(f"  Inf: {has_inf}")
    print(f"  Отсутствуют: {missing}")
    print(f"  Ошибки: {other}")
    print(f"{'='*70}")
    
    for r in results:
        status_icon = "✓" if r["status"] == "ok" else "✗"
        print(f"  {status_icon} {os.path.basename(r['file'])}: {r['status']} "
              f"(размер={r['size_bytes']}B, строки={r['data_rows']}, "
              f"значений={r['total_count']}, nonzero={r['nonzero_fraction']:.1%}, "
              f"abs_max={r['abs_max_val']:.2e}")
        if r["errors"]:
            for err in r["errors"]:
                print(f"    → {err}")


def main():
    if len(sys.argv) < 2:
        print("Использование: python check_spectrum.py <путь> [--recursive]")
        print("  путь — файл или директория")
        sys.exit(1)
    
    path = sys.argv[1]
    recursive = "--recursive" in sys.argv
    
    if os.path.isfile(path):
        results = [check_spectrum_file(path)]
    elif os.path.isdir(path):
        results = check_directory(path, recursive=recursive)
    else:
        print(f"Ошибка: путь '{path}' не существует")
        sys.exit(2)
    
    print_summary(results)
    
    # Выход с кодом 0 если все OK, иначе 1
    all_ok = all(r["status"] == "ok" for r in results)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
