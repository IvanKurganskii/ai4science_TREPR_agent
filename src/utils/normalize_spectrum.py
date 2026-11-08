#!/usr/bin/env python3
"""Нормировка спектров-срезов и кинетик.

Методы нормировки:
- max: на максимальное значение интенсивности
- abs_max: на абсолютный максимум интенсивности
- range: на разность между максимумом и минимумом
- integral: на интеграл спектра (правило трапеций)

Формат входных файлов:
- Строки с # — комментарии (сохраняются)
- Строка 1: магнитное поле (через запятую)
- Строка 2: интенсивность (через запятую)

Выходной файл:
- Те же комментарии
- Строка 1: магнитное поле (без изменений)
- Строка 2: нормированная интенсивность
- Добавляется строка с методом нормировки
"""

import sys
import os
import csv
import math
from pathlib import Path


def read_spectrum(filepath: str) -> tuple:
    """Прочитать CSV-файл спектра.
    
    Возвращает:
        comments: список строк комментариев
        field: список значений магнитного поля
        intensity: список значений интенсивности
    """
    comments = []
    data_lines = []
    
    with open(filepath, 'r') as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith('#'):
                comments.append(stripped)
            elif stripped:
                data_lines.append(stripped)
    
    if len(data_lines) < 2:
        raise ValueError(f"Файл {filepath} содержит менее 2 строк данных")
    
    # Парсинг магнитного поля (строка 1)
    reader = csv.reader([data_lines[0]], delimiter=',')
    field = [float(val) for val in next(reader) if val.strip()]
    
    # Парсинг интенсивности (строка 2)
    reader = csv.reader([data_lines[1]], delimiter=',')
    intensity = [float(val) for val in next(reader) if val.strip()]
    
    if len(field) != len(intensity):
        raise ValueError(f"Длина поля ({len(field)}) и интенсивности ({len(intensity)}) не совпадают")
    
    return comments, field, intensity


def normalize_max(intensity: list) -> list:
    """Нормировка на максимальное значение."""
    max_val = max(intensity)
    if max_val == 0:
        raise ValueError("Все значения интенсивности равны нулю")
    return [v / max_val for v in intensity]


def normalize_abs_max(intensity: list) -> list:
    """Нормировка на абсолютный максимум."""
    abs_max = max(abs(v) for v in intensity)
    if abs_max == 0:
        raise ValueError("Все значения интенсивности равны нулю")
    return [v / abs_max for v in intensity]


def normalize_range(intensity: list) -> list:
    """Нормировка на разность между максимумом и минимумом."""
    min_val = min(intensity)
    max_val = max(intensity)
    range_val = max_val - min_val
    if range_val == 0:
        raise ValueError("Все значения интенсивности одинаковы")
    return [(v - min_val) / range_val for v in intensity]


def normalize_integral(intensity: list) -> list:
    """Нормировка на интеграл спектра (правило трапеций)."""
    # Интеграл по правилу трапеций
    integral = 0.0
    for i in range(1, len(intensity)):
        dx = 1.0  # шаг по полю не используется для нормировки
        integral += (intensity[i] + intensity[i-1]) * dx / 2.0
    
    if integral == 0:
        raise ValueError("Интеграл спектра равен нулю")
    return [v / integral for v in intensity]


def write_spectrum(filepath: str, comments: list, field: list, 
                   intensity: list, method: str, output_dir: str = None):
    """Записать нормированный спектр.
    
    Args:
        filepath: путь к исходному файлу
        comments: список комментариев
        field: магнитное поле
        intensity: нормированная интенсивность
        method: метод нормировки
        output_dir: директория для выходного файла (по умолчанию — рядом с исходным)
    """
    # Определяем выходной путь
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.basename(filepath)
        name, ext = os.path.splitext(base)
        output_path = os.path.join(output_dir, f"{name}_normalized_{method}{ext}")
    else:
        output_path = filepath.replace('.csv', f'_normalized_{method}.csv')
    
    with open(output_path, 'w') as f:
        # Записываем комментарии
        for comment in comments:
            f.write(f"{comment}\n")
        
        # Добавляем строку с методом
        f.write(f"# Normalization method: {method}\n")
        
        # Записываем магнитное поле
        f.write(','.join(f"{v:.7f}" for v in field) + '\n')
        
        # Записываем нормированную интенсивность
        f.write(','.join(f"{v:.10e}" for v in intensity) + '\n')
    
    return output_path


def normalize_file(filepath: str, method: str, output_dir: str = None) -> str:
    """Нормировать один файл спектра.
    
    Args:
        filepath: путь к файлу
        method: метод нормировки (max, abs_max, range, integral)
        output_dir: директория для выходного файла
    
    Returns:
        путь к нормированному файлу
    """
    if method not in ['max', 'abs_max', 'range', 'integral']:
        raise ValueError(f"Неизвестный метод нормировки: {method}. Доступные: max, abs_max, range, integral")
    
    comments, field, intensity = read_spectrum(filepath)
    
    # Применяем нормировку
    if method == 'max':
        normalized = normalize_max(intensity)
    elif method == 'abs_max':
        normalized = normalize_abs_max(intensity)
    elif method == 'range':
        normalized = normalize_range(intensity)
    elif method == 'integral':
        normalized = normalize_integral(intensity)
    
    # Записываем результат
    output_path = write_spectrum(filepath, comments, field, normalized, method, output_dir)
    
    return output_path


def normalize_directory(dirpath: str, method: str, output_dir: str = None, 
                       recursive: bool = False) -> list:
    """Нормировать все CSV-файлы в директории.
    
    Args:
        dirpath: путь к директории
        method: метод нормировки
        output_dir: директория для выходных файлов
        recursive: рекурсивный поиск
    
    Returns:
        список путей к нормированным файлам
    """
    csv_files = []
    
    if recursive:
        for root, dirs, files in os.walk(dirpath):
            for f in files:
                if f.endswith('.csv') and 'normalized' not in f:
                    csv_files.append(os.path.join(root, f))
    else:
        for f in os.listdir(dirpath):
            if f.endswith('.csv') and 'normalized' not in f:
                csv_files.append(os.path.join(dirpath, f))
    
    results = []
    for filepath in sorted(csv_files):
        try:
            output_path = normalize_file(filepath, method, output_dir)
            results.append((filepath, output_path, "ok"))
            print(f"  ✓ {os.path.basename(filepath)} → {os.path.basename(output_path)}")
        except Exception as e:
            results.append((filepath, None, f"error: {e}"))
            print(f"  ✗ {os.path.basename(filepath)}: {e}")
    
    return results


def print_summary(results: list):
    """Вывести краткую сводку."""
    total = len(results)
    ok = sum(1 for _, _, status in results if status == "ok")
    errors = total - ok
    
    print(f"\n{'='*70}")
    print(f"Нормировка: {total} файлов, {ok} успешно, {errors} с ошибками")
    print(f"{'='*70}")


def main():
    if len(sys.argv) < 3:
        print("Использование:")
        print("  python normalize_spectrum.py <путь> <метод> [--output-dir <dir>] [--recursive]")
        print("  методы: max, abs_max, range, integral")
        print("  путь — файл или директория")
        sys.exit(1)
    
    path = sys.argv[1]
    method = sys.argv[2]
    output_dir = None
    recursive = False
    
    i = 3
    while i < len(sys.argv):
        if sys.argv[i] == '--output-dir' and i + 1 < len(sys.argv):
            output_dir = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--recursive':
            recursive = True
            i += 1
        else:
            i += 1
    
    if os.path.isfile(path):
        output_path = normalize_file(path, method, output_dir)
        print(f"Нормирован: {os.path.basename(path)} → {os.path.basename(output_path)}")
        sys.exit(0)
    elif os.path.isdir(path):
        results = normalize_directory(path, method, output_dir, recursive)
        print_summary(results)
        all_ok = all(status == "ok" for _, _, status in results)
        sys.exit(0 if all_ok else 1)
    else:
        print(f"Ошибка: путь '{path}' не существует")
        sys.exit(2)


if __name__ == "__main__":
    main()
