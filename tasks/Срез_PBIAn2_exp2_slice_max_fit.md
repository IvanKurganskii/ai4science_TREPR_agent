---
type: Задача
status: заблокирована
---
# Срез PBIAn2_exp2_slice_max — фит спиновых параметров

Запрос пользователя: «Выбери произвольный срез и зафитируй его»

Готово, когда: получен FIT среза PBIAn2_exp2_slice_max с хорошим согласием (K <= K_MAX относительно шума родительского 2D-измерения); параметры g, D, E, населённости сохранены в results/fits/1d/; проверки check_fit_1d.py, check_refit_1d.py, check_fit_perturbation_1d.py, check_fit_table.py и check_fit_draw_1d.py пройдены (код 0, passed)
Не входит: другие срезы, 2d-моделирование

## Параметры фита (зафиксированы до подбора)
- Срез: results/spectra/slices/PBIAn2_exp2_slice_max_b7827b4c8c4d4df68cb6ecd7efd1f5d4.csv
- Частота СВЧ: 9.73843874 GHz (из data/raw_spectra/pars_PBIAn.json, AN2_300719s1_exp2_smallStep)
- Температура: 79.963 K
- Модель: триплет (S=1), high_j, изотропный g
- Базис населённостей: xyz (фиксирован; совместим с MaRS Cobyla в check_refit_1d.py)
- ham_strain: 0 (фиксирован; MaRS Cobyla не поддерживает ham_strain)
- Свободные параметры: g_iso, D, E, gauss, p_x, p_y (p_z = 1 - p_x - p_y) → N_FREE = 6
- Границы: g_iso [2.0, 2.12]; D [500, 3000] МГц; E [0, 400] МГц при E <= D/3; gauss [1e-5, 5e-4] T
- Шумовое окно: 0-500 нс
- Нормирование: max (по максимуму интенсивности)
- K_MAX: определяется по качеству данных после первого фита
- Допуск улучшения: improvement_atol=0.05, improvement_rtol=0.05

## План
1. Запустить поиск границ и экстремумов (signal_bounds, find_peaks)
2. Определить стартовые параметры (базис, D/E, g_iso)
3. Запустить TPE-поиск (глобальный)
4. Запустить COBYLA‑доводку от лучших параметров TPE
5. Проверить качество согласия через check_fit_1d.py – **пройдено** (K=65173, rmse=0.038)
6. Выполнить контрольную доводку через check_refit_1d.py – **не удалось** (ошибка: Unknown or invalid problem fields)
7. Проверить Пирсон и контроль D/E через check_fit_perturbation_1d.py – **не выполнено** (требуется корректный problem JSON)
8. Проверить параметры через check_fit_table.py – **не выполнено**
9. Отрисовать наложение фита/эксперимента и проверить через check_fit_draw_1d.py – **не выполнено**
10. Сохранить результаты в results/fits/1d/ и results/tables/1d-fit_results.csv

## Проблемы
- Отсутствует корректный `problem_1d.json` для проверки `check_refit_1d.py`. Создан файл `results/fits/1d/PBIAn2_exp2_slice_max_2026-11-09/problem_1d.json` с параметрами из `fit_config.json`. Требуется повторный запуск `check_refit_1d.py` и последующих проверок.

## Связи
- тип: [[Задача]]
- подзадачаОт: [[Моделирование 1d спектров PBIAn]]
