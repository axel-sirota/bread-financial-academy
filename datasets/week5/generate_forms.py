#!/usr/bin/env python3
"""
Generate synthetic scanned form images for Week 5 AI Services lab.

This script creates 5 PNG images of filled business forms that will
produce interesting results when analyzed with Amazon Textract
(OCR, key-value extraction, form fields).

Usage:
    pip install Pillow
    python generate_forms.py

Output:
    scanned_forms/form_001.png through form_005.png
"""

import os
import random
from datetime import datetime, timedelta

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("ERROR: Pillow is required. Install with: pip install Pillow")
    exit(1)

# Set seed for reproducibility
random.seed(42)

# Form data for each of the 5 forms
FORM_DATA = [
    {
        "form_type": "Customer Complaint Form",
        "name": "John Smith",
        "date": "11/15/2024",
        "order_number": "ORD-789456",
        "email": "john.smith@email.com",
        "phone": "(555) 123-4567",
        "issue_type": "Damaged Product",
        "issue_types_available": ["Damaged Product", "Wrong Item", "Late Delivery", "Missing Item", "Quality Issue", "Other"],
        "description": "The laptop screen arrived with a visible crack in the lower left corner. The box showed no external damage so it must have been damaged during handling.",
        "resolution": "Replacement",
        "resolution_options": ["Replacement", "Refund", "Store Credit"],
    },
    {
        "form_type": "Product Return Request",
        "name": "Sarah Johnson",
        "date": "11/12/2024",
        "order_number": "ORD-456123",
        "email": "sarah.j@gmail.com",
        "phone": "(555) 987-6543",
        "issue_type": "Wrong Item",
        "issue_types_available": ["Damaged Product", "Wrong Item", "Late Delivery", "Missing Item", "Quality Issue", "Other"],
        "description": "I ordered size 8 Nike Air Max shoes but received size 10. The invoice shows size 8 so this was a warehouse error.",
        "resolution": "Replacement",
        "resolution_options": ["Replacement", "Refund", "Store Credit"],
    },
    {
        "form_type": "Customer Satisfaction Survey",
        "name": "Michael Chen",
        "date": "11/10/2024",
        "order_number": "ORD-321789",
        "email": "m.chen@company.org",
        "phone": "(555) 456-7890",
        "rating": "4",
        "rating_options": ["1", "2", "3", "4", "5"],
        "feedback": "Overall good experience. Delivery was on time and product quality is excellent. Customer service could be more responsive.",
        "recommend": "Yes",
        "recommend_options": ["Yes", "No", "Maybe"],
    },
    {
        "form_type": "Feedback Form",
        "name": "Emily Davis",
        "date": "11/08/2024",
        "order_number": "ORD-654987",
        "email": "emily.davis@outlook.com",
        "phone": "(555) 234-5678",
        "department": "Customer Service",
        "department_options": ["Customer Service", "Shipping", "Returns", "Billing", "Other"],
        "feedback": "Agent Maria was extremely helpful resolving my issue with a damaged Dyson vacuum. She arranged expedited replacement shipping at no extra cost.",
        "satisfaction": "Very Satisfied",
        "satisfaction_options": ["Very Satisfied", "Satisfied", "Neutral", "Dissatisfied", "Very Dissatisfied"],
    },
    {
        "form_type": "Warranty Claim Form",
        "name": "Robert Wilson",
        "date": "11/05/2024",
        "order_number": "ORD-147258",
        "email": "rwilson@email.net",
        "phone": "(555) 345-6789",
        "product": "Sony WH-1000XM5 Headphones",
        "purchase_date": "08/15/2024",
        "serial_number": "SN-78945612",
        "defect_description": "Left ear cup stopped producing sound after 3 months of normal use. No physical damage. Tried factory reset without success.",
        "claim_type": "Replacement",
        "claim_options": ["Replacement", "Repair", "Refund"],
    },
]


def get_font(size: int, bold: bool = False):
    """Get a font, falling back to default if system fonts unavailable."""
    font_paths = [
        # macOS
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        # Windows
        "C:/Windows/Fonts/arial.ttf",
    ]

    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue

    # Fall back to default
    return ImageFont.load_default()


def draw_checkbox(draw, x, y, checked: bool, size: int = 15):
    """Draw a checkbox at the given position."""
    # Draw box outline
    draw.rectangle([x, y, x + size, y + size], outline="black", width=1)
    # Draw X if checked
    if checked:
        draw.line([(x + 2, y + 2), (x + size - 2, y + size - 2)], fill="black", width=2)
        draw.line([(x + size - 2, y + 2), (x + 2, y + size - 2)], fill="black", width=2)


def create_complaint_form(data: dict, output_path: str):
    """Create a Customer Complaint Form or Product Return Request."""
    # Create image
    img = Image.new("RGB", (850, 1100), color="white")
    draw = ImageDraw.Draw(img)

    # Fonts
    font_title = get_font(22, bold=True)
    font_subtitle = get_font(16, bold=True)
    font_label = get_font(12)
    font_value = get_font(12)

    # Draw border
    draw.rectangle([20, 20, 830, 1080], outline="black", width=2)

    # Header
    y = 40
    draw.text((350, y), "ACME RETAIL", font=font_title, fill="black", anchor="mm")
    y += 35
    draw.text((350, y), data["form_type"].upper(), font=font_subtitle, fill="black", anchor="mm")
    y += 30
    draw.line([(40, y), (810, y)], fill="black", width=1)

    # Customer Information Section
    y += 20
    draw.text((40, y), "CUSTOMER INFORMATION", font=font_subtitle, fill="black")
    y += 25

    fields = [
        ("Customer Name:", data["name"]),
        ("Date:", data["date"]),
        ("Order Number:", data["order_number"]),
        ("Email:", data["email"]),
        ("Phone:", data["phone"]),
    ]

    for label, value in fields:
        draw.text((40, y), label, font=font_label, fill="black")
        draw.text((180, y), value, font=font_value, fill="darkblue")
        y += 25

    # Issue Type Section with checkboxes
    y += 15
    draw.line([(40, y), (810, y)], fill="gray", width=1)
    y += 15
    draw.text((40, y), "ISSUE TYPE (check one):", font=font_subtitle, fill="black")
    y += 25

    col1_x = 40
    col2_x = 280
    col3_x = 520
    positions = [(col1_x, y), (col2_x, y), (col3_x, y),
                 (col1_x, y + 25), (col2_x, y + 25), (col3_x, y + 25)]

    for i, issue in enumerate(data["issue_types_available"]):
        if i < len(positions):
            px, py = positions[i]
            is_checked = (issue == data["issue_type"])
            draw_checkbox(draw, px, py, is_checked)
            draw.text((px + 22, py), issue, font=font_label, fill="black")

    y += 60

    # Description Section
    draw.line([(40, y), (810, y)], fill="gray", width=1)
    y += 15
    draw.text((40, y), "DESCRIPTION OF ISSUE:", font=font_subtitle, fill="black")
    y += 25

    # Draw text box
    draw.rectangle([40, y, 810, y + 120], outline="gray", width=1)

    # Wrap and draw description text
    description = data["description"]
    words = description.split()
    lines = []
    current_line = []
    for word in words:
        test_line = " ".join(current_line + [word])
        if len(test_line) < 90:  # Approximate char limit per line
            current_line.append(word)
        else:
            lines.append(" ".join(current_line))
            current_line = [word]
    if current_line:
        lines.append(" ".join(current_line))

    text_y = y + 10
    for line in lines[:5]:  # Max 5 lines
        draw.text((50, text_y), line, font=font_value, fill="darkblue")
        text_y += 20

    y += 135

    # Resolution Section
    draw.line([(40, y), (810, y)], fill="gray", width=1)
    y += 15
    draw.text((40, y), "PREFERRED RESOLUTION:", font=font_subtitle, fill="black")
    y += 25

    for i, option in enumerate(data["resolution_options"]):
        px = 40 + (i * 200)
        is_checked = (option == data["resolution"])
        draw_checkbox(draw, px, y, is_checked)
        draw.text((px + 22, y), option, font=font_label, fill="black")

    y += 50

    # Signature Section
    draw.line([(40, y), (810, y)], fill="gray", width=1)
    y += 20
    draw.text((40, y), "Signature:", font=font_label, fill="black")
    draw.text((120, y), data["name"], font=font_value, fill="darkblue")
    draw.text((400, y), "Date:", font=font_label, fill="black")
    draw.text((450, y), data["date"], font=font_value, fill="darkblue")

    # Add some "noise" for realistic scan effect
    add_scan_noise(img)

    # Save
    img.save(output_path, "PNG")


def create_survey_form(data: dict, output_path: str):
    """Create a Customer Satisfaction Survey form."""
    img = Image.new("RGB", (850, 1100), color="white")
    draw = ImageDraw.Draw(img)

    font_title = get_font(22, bold=True)
    font_subtitle = get_font(16, bold=True)
    font_label = get_font(12)
    font_value = get_font(12)

    draw.rectangle([20, 20, 830, 1080], outline="black", width=2)

    y = 40
    draw.text((350, y), "ACME RETAIL", font=font_title, fill="black", anchor="mm")
    y += 35
    draw.text((350, y), "CUSTOMER SATISFACTION SURVEY", font=font_subtitle, fill="black", anchor="mm")
    y += 30
    draw.line([(40, y), (810, y)], fill="black", width=1)

    y += 20
    fields = [
        ("Customer Name:", data["name"]),
        ("Date:", data["date"]),
        ("Order Number:", data["order_number"]),
        ("Email:", data["email"]),
    ]

    for label, value in fields:
        draw.text((40, y), label, font=font_label, fill="black")
        draw.text((180, y), value, font=font_value, fill="darkblue")
        y += 25

    # Rating Section
    y += 20
    draw.line([(40, y), (810, y)], fill="gray", width=1)
    y += 15
    draw.text((40, y), "OVERALL RATING (1-5):", font=font_subtitle, fill="black")
    y += 25

    for i, rating in enumerate(data["rating_options"]):
        px = 40 + (i * 100)
        is_checked = (rating == data["rating"])
        draw_checkbox(draw, px, y, is_checked)
        draw.text((px + 22, y), rating, font=font_label, fill="black")

    y += 50

    # Would recommend Section
    draw.text((40, y), "WOULD YOU RECOMMEND US?", font=font_subtitle, fill="black")
    y += 25

    for i, option in enumerate(data["recommend_options"]):
        px = 40 + (i * 150)
        is_checked = (option == data["recommend"])
        draw_checkbox(draw, px, y, is_checked)
        draw.text((px + 22, y), option, font=font_label, fill="black")

    y += 50

    # Feedback Section
    draw.line([(40, y), (810, y)], fill="gray", width=1)
    y += 15
    draw.text((40, y), "ADDITIONAL FEEDBACK:", font=font_subtitle, fill="black")
    y += 25
    draw.rectangle([40, y, 810, y + 150], outline="gray", width=1)

    # Draw feedback text
    words = data["feedback"].split()
    lines = []
    current_line = []
    for word in words:
        test_line = " ".join(current_line + [word])
        if len(test_line) < 90:
            current_line.append(word)
        else:
            lines.append(" ".join(current_line))
            current_line = [word]
    if current_line:
        lines.append(" ".join(current_line))

    text_y = y + 10
    for line in lines[:6]:
        draw.text((50, text_y), line, font=font_value, fill="darkblue")
        text_y += 22

    add_scan_noise(img)
    img.save(output_path, "PNG")


def create_feedback_form(data: dict, output_path: str):
    """Create a general Feedback Form."""
    img = Image.new("RGB", (850, 1100), color="white")
    draw = ImageDraw.Draw(img)

    font_title = get_font(22, bold=True)
    font_subtitle = get_font(16, bold=True)
    font_label = get_font(12)
    font_value = get_font(12)

    draw.rectangle([20, 20, 830, 1080], outline="black", width=2)

    y = 40
    draw.text((350, y), "ACME RETAIL", font=font_title, fill="black", anchor="mm")
    y += 35
    draw.text((350, y), "FEEDBACK FORM", font=font_subtitle, fill="black", anchor="mm")
    y += 30
    draw.line([(40, y), (810, y)], fill="black", width=1)

    y += 20
    fields = [
        ("Name:", data["name"]),
        ("Date:", data["date"]),
        ("Order Number:", data["order_number"]),
        ("Email:", data["email"]),
        ("Phone:", data["phone"]),
    ]

    for label, value in fields:
        draw.text((40, y), label, font=font_label, fill="black")
        draw.text((180, y), value, font=font_value, fill="darkblue")
        y += 25

    # Department Section
    y += 20
    draw.line([(40, y), (810, y)], fill="gray", width=1)
    y += 15
    draw.text((40, y), "DEPARTMENT:", font=font_subtitle, fill="black")
    y += 25

    for i, dept in enumerate(data["department_options"]):
        if i < 3:
            px = 40 + (i * 200)
            py = y
        else:
            px = 40 + ((i - 3) * 200)
            py = y + 25
        is_checked = (dept == data["department"])
        draw_checkbox(draw, px, py, is_checked)
        draw.text((px + 22, py), dept, font=font_label, fill="black")

    y += 60

    # Satisfaction Section
    draw.line([(40, y), (810, y)], fill="gray", width=1)
    y += 15
    draw.text((40, y), "SATISFACTION LEVEL:", font=font_subtitle, fill="black")
    y += 25

    for i, level in enumerate(data["satisfaction_options"]):
        if i < 3:
            px = 40 + (i * 200)
            py = y
        else:
            px = 40 + ((i - 3) * 200)
            py = y + 25
        is_checked = (level == data["satisfaction"])
        draw_checkbox(draw, px, py, is_checked)
        draw.text((px + 22, py), level, font=font_label, fill="black")

    y += 60

    # Feedback Section
    draw.line([(40, y), (810, y)], fill="gray", width=1)
    y += 15
    draw.text((40, y), "YOUR FEEDBACK:", font=font_subtitle, fill="black")
    y += 25
    draw.rectangle([40, y, 810, y + 150], outline="gray", width=1)

    words = data["feedback"].split()
    lines = []
    current_line = []
    for word in words:
        test_line = " ".join(current_line + [word])
        if len(test_line) < 90:
            current_line.append(word)
        else:
            lines.append(" ".join(current_line))
            current_line = [word]
    if current_line:
        lines.append(" ".join(current_line))

    text_y = y + 10
    for line in lines[:6]:
        draw.text((50, text_y), line, font=font_value, fill="darkblue")
        text_y += 22

    add_scan_noise(img)
    img.save(output_path, "PNG")


def create_warranty_form(data: dict, output_path: str):
    """Create a Warranty Claim Form."""
    img = Image.new("RGB", (850, 1100), color="white")
    draw = ImageDraw.Draw(img)

    font_title = get_font(22, bold=True)
    font_subtitle = get_font(16, bold=True)
    font_label = get_font(12)
    font_value = get_font(12)

    draw.rectangle([20, 20, 830, 1080], outline="black", width=2)

    y = 40
    draw.text((350, y), "ACME RETAIL", font=font_title, fill="black", anchor="mm")
    y += 35
    draw.text((350, y), "WARRANTY CLAIM FORM", font=font_subtitle, fill="black", anchor="mm")
    y += 30
    draw.line([(40, y), (810, y)], fill="black", width=1)

    # Customer Section
    y += 20
    draw.text((40, y), "CUSTOMER INFORMATION", font=font_subtitle, fill="black")
    y += 25

    fields = [
        ("Name:", data["name"]),
        ("Date:", data["date"]),
        ("Email:", data["email"]),
        ("Phone:", data["phone"]),
    ]

    for label, value in fields:
        draw.text((40, y), label, font=font_label, fill="black")
        draw.text((180, y), value, font=font_value, fill="darkblue")
        y += 25

    # Product Section
    y += 15
    draw.line([(40, y), (810, y)], fill="gray", width=1)
    y += 15
    draw.text((40, y), "PRODUCT INFORMATION", font=font_subtitle, fill="black")
    y += 25

    product_fields = [
        ("Product:", data["product"]),
        ("Order Number:", data["order_number"]),
        ("Purchase Date:", data["purchase_date"]),
        ("Serial Number:", data["serial_number"]),
    ]

    for label, value in product_fields:
        draw.text((40, y), label, font=font_label, fill="black")
        draw.text((180, y), value, font=font_value, fill="darkblue")
        y += 25

    # Defect Description
    y += 15
    draw.line([(40, y), (810, y)], fill="gray", width=1)
    y += 15
    draw.text((40, y), "DEFECT DESCRIPTION:", font=font_subtitle, fill="black")
    y += 25
    draw.rectangle([40, y, 810, y + 120], outline="gray", width=1)

    words = data["defect_description"].split()
    lines = []
    current_line = []
    for word in words:
        test_line = " ".join(current_line + [word])
        if len(test_line) < 90:
            current_line.append(word)
        else:
            lines.append(" ".join(current_line))
            current_line = [word]
    if current_line:
        lines.append(" ".join(current_line))

    text_y = y + 10
    for line in lines[:5]:
        draw.text((50, text_y), line, font=font_value, fill="darkblue")
        text_y += 22

    y += 135

    # Claim Type
    draw.line([(40, y), (810, y)], fill="gray", width=1)
    y += 15
    draw.text((40, y), "CLAIM TYPE:", font=font_subtitle, fill="black")
    y += 25

    for i, option in enumerate(data["claim_options"]):
        px = 40 + (i * 200)
        is_checked = (option == data["claim_type"])
        draw_checkbox(draw, px, y, is_checked)
        draw.text((px + 22, y), option, font=font_label, fill="black")

    add_scan_noise(img)
    img.save(output_path, "PNG")


def add_scan_noise(img: Image.Image):
    """Add subtle noise to simulate scanned document effect."""
    # Very light noise - don't make it too noisy for Textract
    pixels = img.load()
    width, height = img.size

    for _ in range(500):  # Add some random gray pixels
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        gray = random.randint(220, 250)
        pixels[x, y] = (gray, gray, gray)


def main():
    """Generate all scanned form images."""
    output_dir = "scanned_forms"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating {len(FORM_DATA)} scanned form images...")

    for i, data in enumerate(FORM_DATA, start=1):
        output_path = os.path.join(output_dir, f"form_{i:03d}.png")
        form_type = data["form_type"]

        if form_type in ["Customer Complaint Form", "Product Return Request"]:
            create_complaint_form(data, output_path)
        elif form_type == "Customer Satisfaction Survey":
            create_survey_form(data, output_path)
        elif form_type == "Feedback Form":
            create_feedback_form(data, output_path)
        elif form_type == "Warranty Claim Form":
            create_warranty_form(data, output_path)

        print(f"  Created: {output_path} ({form_type})")

    print(f"\nGenerated {len(FORM_DATA)} form images in {output_dir}/")
    print("\nForm types:")
    for i, data in enumerate(FORM_DATA, start=1):
        print(f"  form_{i:03d}.png: {data['form_type']}")


if __name__ == "__main__":
    main()
