"""
Sample OMR Sheet Generator
Creates realistic sample OMR sheets for testing the ScoreVision Pro system
"""

import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import random

def create_sample_omr_sheet(student_name, student_id, exam_subject, total_questions=50, choices_per_question=4):
    """Create a sample OMR sheet with filled bubbles"""
    
    # Create a white background
    width, height = 2100, 2970  # A4 size at 300 DPI
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Try to load a font, fallback to default if not available
    try:
        title_font = ImageFont.truetype("arial.ttf", 60)
        header_font = ImageFont.truetype("arial.ttf", 40)
        text_font = ImageFont.truetype("arial.ttf", 30)
        small_font = ImageFont.truetype("arial.ttf", 24)
    except:
        title_font = ImageFont.load_default()
        header_font = ImageFont.load_default()
        text_font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # Header section
    draw.rectangle([(50, 50), (width-50, 200)], outline='black', width=3)
    draw.text((width//2, 80), "SCOREVISION PRO", font=title_font, fill='black', anchor='mt')
    draw.text((width//2, 140), f"{exam_subject} - ANSWER SHEET", font=header_font, fill='black', anchor='mt')
    
    # Student information section
    y_pos = 250
    draw.rectangle([(50, y_pos), (width-50, y_pos+150)], outline='black', width=2)
    draw.text((70, y_pos+20), f"Student Name: {student_name}", font=text_font, fill='black')
    draw.text((70, y_pos+60), f"Student ID: {student_id}", font=text_font, fill='black')
    draw.text((70, y_pos+100), f"Date: March 2024", font=text_font, fill='black')
    
    # Instructions
    y_pos = 450
    draw.text((70, y_pos), "INSTRUCTIONS:", font=header_font, fill='black')
    draw.text((70, y_pos+40), "• Fill bubbles completely with dark pencil or pen", font=small_font, fill='black')
    draw.text((70, y_pos+70), "• Make no stray marks on this sheet", font=small_font, fill='black')
    draw.text((70, y_pos+100), "• Erase completely to change answers", font=small_font, fill='black')
    
    # Question grid
    start_y = 600
    questions_per_column = 25
    column_width = (width - 200) // 2
    bubble_size = 25
    question_spacing = 45
    choice_spacing = 60
    
    # Generate random answers for demonstration
    answers = {}
    choice_labels = ['A', 'B', 'C', 'D'][:choices_per_question]
    
    for q in range(1, total_questions + 1):
        # 90% chance of having an answer, 10% blank
        if random.random() < 0.9:
            answers[q] = random.choice(choice_labels)
    
    # Draw questions in two columns
    for col in range(2):
        start_question = col * questions_per_column + 1
        end_question = min(start_question + questions_per_column, total_questions + 1)
        
        x_offset = 100 + col * column_width
        
        for q_num in range(start_question, end_question):
            y = start_y + (q_num - start_question) * question_spacing
            
            # Question number
            draw.text((x_offset, y), f"{q_num:2d}.", font=text_font, fill='black')
            
            # Choice bubbles
            for i, choice in enumerate(choice_labels):
                bubble_x = x_offset + 50 + i * choice_spacing
                bubble_y = y
                
                # Draw bubble circle
                draw.ellipse([
                    (bubble_x - bubble_size//2, bubble_y - bubble_size//2),
                    (bubble_x + bubble_size//2, bubble_y + bubble_size//2)
                ], outline='black', width=2)
                
                # Draw choice label
                draw.text((bubble_x, bubble_y - 40), choice, font=small_font, fill='black', anchor='mm')
                
                # Fill bubble if this is the student's answer
                if q_num in answers and answers[q_num] == choice:
                    draw.ellipse([
                        (bubble_x - bubble_size//2 + 3, bubble_y - bubble_size//2 + 3),
                        (bubble_x + bubble_size//2 - 3, bubble_y + bubble_size//2 - 3)
                    ], fill='black')
    
    # Add timing marks for alignment (used by OMR processing)
    timing_mark_size = 20
    # Top timing marks
    for i in range(10):
        x = 150 + i * 180
        draw.rectangle([(x, 30), (x + timing_mark_size, 30 + timing_mark_size)], fill='black')
    
    # Side timing marks
    for i in range(15):
        y = 300 + i * 180
        draw.rectangle([(30, y), (30 + timing_mark_size, y + timing_mark_size)], fill='black')
        draw.rectangle([(width - 50, y), (width - 50 + timing_mark_size, y + timing_mark_size)], fill='black')
    
    # Bottom timing marks
    for i in range(10):
        x = 150 + i * 180
        draw.rectangle([(x, height - 50), (x + timing_mark_size, height - 50 + timing_mark_size)], fill='black')
    
    # Convert PIL to OpenCV format
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # Add slight rotation and noise to simulate real scanning
    rows, cols = img_cv.shape[:2]
    
    # Small random rotation (-2 to 2 degrees)
    angle = random.uniform(-2, 2)
    rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    img_cv = cv2.warpAffine(img_cv, rotation_matrix, (cols, rows), borderValue=(255, 255, 255))
    
    # Add slight noise
    noise = np.random.normal(0, 5, img_cv.shape).astype(np.uint8)
    img_cv = cv2.add(img_cv, noise)
    
    # Slight blur to simulate scanning
    img_cv = cv2.GaussianBlur(img_cv, (3, 3), 0)
    
    return img_cv, answers

def generate_sample_sheets():
    """Generate multiple sample OMR sheets"""
    
    # Create output directory
    output_dir = "sample_data/sample_sheets"
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample student data
    students = [
        ("Alice Johnson", "STU001", "Advanced Physics"),
        ("Bob Smith", "STU002", "Advanced Physics"),
        ("Carol Davis", "STU003", "Advanced Mathematics"),
        ("David Wilson", "STU004", "Advanced Mathematics"),
        ("Eva Brown", "STU005", "Computer Science"),
        ("Frank Miller", "STU006", "Computer Science"),
    ]
    
    for i, (name, student_id, subject) in enumerate(students):
        print(f"Generating sample sheet {i+1}/{len(students)}: {name}")
        
        # Different question counts for different subjects
        if "Physics" in subject:
            total_q = 50
        elif "Mathematics" in subject:
            total_q = 40
        else:  # Computer Science
            total_q = 60
        
        img, answers = create_sample_omr_sheet(name, student_id, subject, total_q)
        
        # Save the image
        filename = f"sample_{student_id}_{subject.replace(' ', '_')}.jpg"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, img)
        
        # Save the answers (for reference)
        answers_filename = f"sample_{student_id}_answers.json"
        answers_filepath = os.path.join(output_dir, answers_filename)
        import json
        with open(answers_filepath, 'w') as f:
            json.dump({
                'student_name': name,
                'student_id': student_id,
                'subject': subject,
                'answers': answers,
                'total_questions': total_q
            }, f, indent=2)
    
    print(f"\n✅ Generated {len(students)} sample OMR sheets in {output_dir}")
    print("These can be used for testing the ScoreVision Pro system")

if __name__ == "__main__":
    generate_sample_sheets()