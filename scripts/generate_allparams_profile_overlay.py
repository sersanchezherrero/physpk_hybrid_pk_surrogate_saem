from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = ["arialbd.ttf", "Arial Bold.ttf"] if bold else ["arial.ttf", "Arial.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def relabel_time_axis(image: Image.Image, label: str, font: ImageFont.ImageFont) -> Image.Image:
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    text_width, text_height = text_size(draw, label, font)
    y = height - 34
    draw.rectangle(
        [
            int(width * 0.42),
            y - 2,
            int(width * 0.58),
            y + text_height + 4,
        ],
        fill="white",
    )
    draw.text(((width - text_width) / 2, y), label, fill="black", font=font)
    return image


def relabel_concentration_axis(image: Image.Image, label: str, font: ImageFont.ImageFont) -> Image.Image:
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    _, height = image.size

    text_width, text_height = text_size(draw, label, font)
    label_image = Image.new("RGBA", (height, text_height + 8), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label_image)
    label_draw.text(
        ((height - text_width) / 2, 4),
        label,
        fill="black",
        font=font,
    )
    rotated_label = label_image.rotate(90, expand=True)

    x = 6
    y = int((height - rotated_label.height) / 2)
    draw.rectangle([0, y - 4, 34, y + rotated_label.height + 4], fill="white")
    image.paste(rotated_label.convert("RGB"), (x, y), rotated_label)
    return image


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a 2x2 supplementary profile-sensitivity overlay for theta1-theta4."
    )
    parser.add_argument(
        "--input_dir",
        default="outputs/sensitivity_real_design",
        help="Directory containing profile_theta1.png ... profile_theta4.png.",
    )
    parser.add_argument(
        "--output_png",
        default="paper_project_review_extract/figures/figure_allparams_profile_overlay.png",
    )
    parser.add_argument(
        "--copy_png",
        default="outputs/paper_q1_assets/figure_allparams_profile_overlay.png",
    )
    parser.add_argument("--x_label", default="Time (h)")
    parser.add_argument("--y_label", default="Predicted concentration (ug/L)")
    args = parser.parse_args()

    input_dir = ROOT / args.input_dir
    output_png = ROOT / args.output_png
    copy_png = ROOT / args.copy_png if args.copy_png else None

    panels = [
        ("A)", input_dir / "profile_theta1.png"),
        ("B)", input_dir / "profile_theta2.png"),
        ("C)", input_dir / "profile_theta3.png"),
        ("D)", input_dir / "profile_theta4.png"),
    ]

    missing = [str(path) for _, path in panels if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing sensitivity profile figure(s): " + ", ".join(missing))

    label_font = load_font(42, bold=True)
    axis_font = load_font(18, bold=False)
    images = []
    for _, path in panels:
        image = Image.open(path)
        image = relabel_time_axis(image, args.x_label, axis_font)
        image = relabel_concentration_axis(image, args.y_label, axis_font)
        images.append(image)

    image_width, image_height = images[0].size
    label_width = 58
    image_top_pad = 8
    gap_x = 34
    gap_y = 34
    panel_width = label_width + image_width
    panel_height = image_top_pad + image_height
    canvas = Image.new(
        "RGB",
        (2 * panel_width + gap_x, 2 * panel_height + gap_y),
        "white",
    )
    draw = ImageDraw.Draw(canvas)

    for index, ((label, _), image) in enumerate(zip(panels, images)):
        row, col = divmod(index, 2)
        x0 = col * (panel_width + gap_x)
        y0 = row * (panel_height + gap_y)
        draw.text((x0, y0 + 10), label, fill="black", font=label_font)
        canvas.paste(image, (x0 + label_width, y0 + image_top_pad))

    output_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_png, dpi=(300, 300))

    if copy_png is not None:
        copy_png.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_png, copy_png)

    print(f"Saved {output_png}")
    if copy_png is not None:
        print(f"Copied {copy_png}")


if __name__ == "__main__":
    main()
