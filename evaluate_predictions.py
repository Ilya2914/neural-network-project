import argparse
import csv
from collections import Counter
from html import escape
from pathlib import Path


ENCODINGS = ("utf-8-sig", "utf-8", "cp1251")
W, H = 1600, 900
BLUE = "#4472C4"
ORANGE = "#ED7D31"
GREEN = "#70AD47"
RED = "#C00000"
DARK = "#44546A"
TEXT = "#263238"
MUTED = "#6B7280"
GRID = "#E7E6E6"


def read_csv(path: str, encoding: str | None) -> tuple[list[str], list[dict[str, str]]]:
    encodings = (encoding,) if encoding else ENCODINGS
    last_error = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                return list(reader.fieldnames or []), list(reader)
        except UnicodeDecodeError as error:
            last_error = error
    raise ValueError(f"Cannot read {path} with encodings {encodings}") from last_error


def pick(columns: list[str], names: list[str], file_name: str) -> str:
    for name in names:
        if name in columns:
            return name
    raise ValueError(f"Cannot find {names} in {file_name}. Columns: {columns}")


def normalize_id(value: str) -> str:
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    return str(int(number)) if number.is_integer() else text


def repair_text(value: str) -> str:
    text = str(value)
    try:
        return text.encode("cp1251").decode("utf-8")
    except UnicodeError:
        return text


def load_categories(path: str | None, encoding: str | None) -> dict[str, str]:
    if not path or not Path(path).exists():
        return {}
    columns, rows = read_csv(path, encoding)
    id_col = pick(columns, ["category_id", "id"], path)
    name_col = pick(columns, ["name", "category_name"], path)
    return {normalize_id(row[id_col]): repair_text(row[name_col]) for row in rows}


def short_name(category_id: str, categories: dict[str, str], limit: int = 42) -> str:
    name = categories.get(category_id, f"Категория {category_id}")
    parts = [part.strip() for part in name.split("|") if part.strip()]
    compact = " / ".join(parts[-2:]) if len(parts) >= 2 else name
    text = f"{category_id}. {compact}"
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def fmt_int(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, score


def evaluate(
    truth_rows: list[dict[str, str]],
    pred_rows: list[dict[str, str]],
    truth_id_col: str,
    pred_id_col: str,
    truth_label_col: str,
    pred_label_col: str,
) -> dict[str, object]:
    pred_by_id = {str(row[pred_id_col]).strip(): row for row in pred_rows}
    support: Counter[str] = Counter()
    predicted: Counter[str] = Counter()
    correct_by_class: Counter[str] = Counter()
    errors_by_class: Counter[str] = Counter()
    confusion: Counter[tuple[str, str]] = Counter()
    correct = 0
    matched = 0

    for row in truth_rows:
        item_id = str(row[truth_id_col]).strip()
        pred_row = pred_by_id.get(item_id)
        if pred_row is None:
            continue

        true_label = normalize_id(row[truth_label_col])
        pred_label = normalize_id(pred_row[pred_label_col])
        matched += 1
        support[true_label] += 1
        predicted[pred_label] += 1

        if true_label == pred_label:
            correct += 1
            correct_by_class[true_label] += 1
        else:
            errors_by_class[true_label] += 1
            confusion[(true_label, pred_label)] += 1

    if matched == 0:
        raise ValueError("No matching item_id values between truth and predictions.")

    per_class = {}
    macro_f1 = 0.0
    weighted_f1 = 0.0
    labels = sorted(set(support) | set(predicted), key=lambda x: int(x) if x.isdigit() else x)

    for label in labels:
        tp = correct_by_class[label]
        fp = predicted[label] - tp
        fn = support[label] - tp
        precision, recall, score = f1(tp, fp, fn)
        class_support = support[label]
        errors = errors_by_class[label]
        macro_f1 += score
        weighted_f1 += score * class_support
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": score,
            "support": class_support,
            "errors": errors,
            "error_rate": errors / class_support if class_support else 0.0,
        }

    return {
        "matched": matched,
        "correct": correct,
        "wrong": matched - correct,
        "accuracy": correct / matched,
        "f1_macro": macro_f1 / len(labels),
        "f1_weighted": weighted_f1 / matched,
        "per_class": per_class,
        "confusion": confusion,
    }


def t(text: str, x: int, y: int, size: int, color: str = TEXT, weight: int = 400, anchor: str = "start") -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{color}" '
        f'text-anchor="{anchor}">{escape(text)}</text>'
    )


def rect(x: int, y: int, w: int | float, h: int, color: str, radius: int = 8, stroke: str | None = None) -> str:
    border = f' stroke="{stroke}"' if stroke else ""
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{color}"{border}/>'


def save_svg(path: Path, body: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">\n'
            f'{rect(0, 0, W, H, "#FFFFFF", 0)}\n'
            + "\n".join(body)
            + "\n</svg>\n"
        )


def card(title: str, value: str, x: int, y: int, color: str) -> list[str]:
    return [
        rect(x, y, 320, 128, "#F8FAFC", 8, GRID),
        rect(x, y, 10, 128, color, 5),
        t(title, x + 34, y + 42, 22, MUTED),
        t(value, x + 34, y + 98, 42, DARK, 700),
    ]


def chart_title(title: str, subtitle: str) -> list[str]:
    return [
        t(title, 90, 92, 52, DARK, 700),
        t(subtitle, 92, 140, 25, MUTED),
    ]


def render_summary(path: Path, metrics: dict[str, object]) -> None:
    rows = [
        ("Accuracy", float(metrics["accuracy"]), BLUE),
        ("F1 macro", float(metrics["f1_macro"]), ORANGE),
        ("F1 weighted", float(metrics["f1_weighted"]), GREEN),
    ]
    body = chart_title("Качество модели", "Сравнение test_predictions.csv с готовыми ответами")
    body += card("Всего объектов", fmt_int(int(metrics["matched"])), 90, 205, BLUE)
    body += card("Верно", fmt_int(int(metrics["correct"])), 440, 205, GREEN)
    body += card("Ошибки", fmt_int(int(metrics["wrong"])), 790, 205, ORANGE)
    body += card("Классов", str(len(metrics["per_class"])), 1140, 205, "#A5A5A5")

    for i, (label, value, color) in enumerate(rows):
        y = 440 + i * 105
        body += [
            t(label, 90, y + 38, 30, TEXT, 700),
            rect(310, y, 880, 56, GRID),
            rect(310, y, round(880 * value, 1), 56, color),
            t(fmt_pct(value), 1235, y + 39, 34, color, 700),
        ]
    save_svg(path, body)


def render_correct_wrong(path: Path, metrics: dict[str, object]) -> None:
    total = int(metrics["matched"])
    correct = int(metrics["correct"])
    wrong = int(metrics["wrong"])
    correct_share = correct / total
    x, y, w, h = 130, 435, 1340, 96
    body = chart_title("Верные ответы и ошибки", "Итоговая доля правильных и неправильных классификаций")
    body += [
        t(fmt_pct(correct_share), 270, 315, 82, GREEN, 700),
        t("верно", 305, 365, 32, MUTED, 700),
        t(fmt_pct(1 - correct_share), 1035, 315, 82, ORANGE, 700),
        t("ошибок", 1065, 365, 32, MUTED, 700),
        rect(x, y, w, h, GRID, 10),
        rect(x, y, round(w * correct_share, 1), h, GREEN, 10),
        rect(round(x + w * correct_share), y, round(w * (1 - correct_share), 1), h, ORANGE, 10),
        t(f"{fmt_int(correct)} правильных", x, y + 170, 32, GREEN, 700),
        t(f"{fmt_int(wrong)} ошибок", x + w, y + 170, 32, ORANGE, 700, "end"),
        t(f"Всего сравнено: {fmt_int(total)} объявлений", 90, 805, 28, MUTED),
    ]
    save_svg(path, body)


def render_bar_chart(
    path: Path,
    title: str,
    subtitle: str,
    rows: list[tuple[str, float, str]],
    color: str,
    value_is_percent: bool,
) -> None:
    max_value = max((value for _, value, _ in rows), default=1.0)
    bar_x, value_x = 725, 1365
    bar_w, row_h = 540, 74
    body = chart_title(title, subtitle)

    for i, (label, value, note) in enumerate(rows):
        y = 205 + i * row_h
        scaled = value if value_is_percent else value / max_value
        shown = fmt_pct(value) if value_is_percent else fmt_int(int(value))
        body += [
            t(label, 90, y + 26, 23, TEXT, 700),
            t(note, 90, y + 55, 18, MUTED),
            rect(bar_x, y + 6, bar_w, 34, GRID),
            rect(bar_x, y + 6, round(bar_w * scaled, 1), 34, color),
            t(shown, value_x, y + 32, 24, color, 700),
        ]
    save_svg(path, body)


def render_confusion(path: Path, metrics: dict[str, object], categories: dict[str, str], top_n: int) -> None:
    pairs = metrics["confusion"].most_common(top_n)
    max_count = max((count for _, count in pairs), default=1)
    body = chart_title("Самые частые пары ошибок", "Истинная категория слева, предсказанная справа")

    for i, ((true_id, pred_id), count) in enumerate(pairs):
        y = 205 + i * 74
        body += [
            t(short_name(true_id, categories, 36), 90, y + 25, 21, TEXT, 700),
            t("->", 590, y + 25, 22, MUTED, 700),
            t(short_name(pred_id, categories, 36), 650, y + 25, 21, TEXT, 700),
            rect(1120, y + 5, 250, 34, GRID),
            rect(1120, y + 5, round(250 * count / max_count, 1), 34, RED),
            t(fmt_int(count), 1410, y + 31, 24, RED, 700),
        ]
    save_svg(path, body)


def make_charts(charts_dir: Path, metrics: dict[str, object], categories: dict[str, str], top_n: int) -> list[Path]:
    per_class = metrics["per_class"]
    worst_f1 = sorted(per_class.items(), key=lambda item: item[1]["f1"])[:top_n]
    most_errors = sorted(per_class.items(), key=lambda item: item[1]["errors"], reverse=True)[:top_n]

    outputs = [
        charts_dir / "01_model_quality.svg",
        charts_dir / "02_correct_vs_wrong.svg",
        charts_dir / "03_worst_categories_f1.svg",
        charts_dir / "04_errors_by_category.svg",
        charts_dir / "05_confusion_pairs.svg",
    ]
    render_summary(outputs[0], metrics)
    render_correct_wrong(outputs[1], metrics)
    render_bar_chart(
        outputs[2],
        "Худшие категории по F1",
        "Категории, где модель чаще всего теряет качество",
        [
            (short_name(label, categories), float(values["f1"]), f'n={fmt_int(int(values["support"]))}')
            for label, values in worst_f1
        ],
        ORANGE,
        True,
    )
    render_bar_chart(
        outputs[3],
        "Где больше всего ошибок",
        "Абсолютное число неверных ответов по истинной категории",
        [
            (
                short_name(label, categories),
                float(values["errors"]),
                f'ошибка в {fmt_pct(float(values["error_rate"]))} случаев',
            )
            for label, values in most_errors
        ],
        BLUE,
        False,
    )
    render_confusion(outputs[4], metrics, categories, top_n)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate presentation-ready charts for predictions.")
    parser.add_argument("--truth", default="predictions.csv")
    parser.add_argument("--pred", default="test_predictions.csv")
    parser.add_argument("--category", default="category.csv")
    parser.add_argument("--charts_dir", default="presentation_charts")
    parser.add_argument("--encoding", default=None)
    parser.add_argument("--top_n", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    truth_cols, truth_rows = read_csv(args.truth, args.encoding)
    pred_cols, pred_rows = read_csv(args.pred, args.encoding)
    metrics = evaluate(
        truth_rows,
        pred_rows,
        pick(truth_cols, ["item_id"], args.truth),
        pick(pred_cols, ["item_id"], args.pred),
        pick(truth_cols, ["category_id", "true_category_id", "target", "label"], args.truth),
        pick(pred_cols, ["predicted_category_id", "category_id", "prediction", "label"], args.pred),
    )
    outputs = make_charts(
        Path(args.charts_dir),
        metrics,
        load_categories(args.category, args.encoding),
        args.top_n,
    )

    print(f"Matched rows: {fmt_int(int(metrics['matched']))}")
    print(f"Correct: {fmt_int(int(metrics['correct']))}")
    print(f"Wrong: {fmt_int(int(metrics['wrong']))}")
    print(f"Accuracy: {float(metrics['accuracy']):.6f}")
    print(f"F1 macro: {float(metrics['f1_macro']):.6f}")
    print("Charts saved:")
    for path in outputs:
        print(f"- {path}")


if __name__ == "__main__":
    main()
