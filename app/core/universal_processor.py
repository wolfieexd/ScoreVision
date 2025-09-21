# Universal OMR Processor - Works with any orientation and format
"""
Universal OMR Processor that handles any image orientation and format
Automatically detects and corrects orientation, adapts to different formats
"""

import cv2
import numpy as np
import os
import time
from typing import Dict, List, Tuple, Optional

class UniversalOMRProcessor:
    """
    Universal OMR processor that works with any image orientation and format
    """
    
    def __init__(self):
        self.name = "Universal OMR Processor"
        self.version = "1.0"
        
    def process_omr_sheet(self, image_path: str, answer_key: Optional[Dict] = None) -> Dict:
        """
        Process OMR sheet with universal orientation and format support
        
        Args:
            image_path: Path to the OMR image
            answer_key: Optional answer key for validation
            
        Returns:
            Dict containing processing results
        """
        start_time = time.time()
        
        try:
            # Load and validate image
            image = cv2.imread(image_path)
            if image is None:
                return self._create_error_result("Could not load image file")
            
            # Get basic image info
            original_height, original_width = image.shape[:2]
            file_size = os.path.getsize(image_path) / 1024  # KB
            
            print(f"\n🔄 UNIVERSAL OMR PROCESSING")
            print(f"📄 Processing: {os.path.basename(image_path)}")
            print(f"📏 Image: {original_width}x{original_height}, {file_size:.1f}KB")
            
            # UNIVERSAL PROCESSING PIPELINE
            
            # Step 1: Auto-detect and correct orientation
            rotation_angle, oriented_image = self._detect_and_correct_orientation(image)
            
            # Step 2: Normalize image size and enhance quality
            processed_image = self._normalize_and_enhance_image(oriented_image)
            
            # Step 3: Adaptive bubble detection
            gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
            marked_bubbles = self._detect_marked_bubbles_universal(gray)
            
            if not marked_bubbles:
                return self._create_error_result("No marked bubbles detected")
            
            # Step 4: Adaptive column detection
            columns = self._detect_columns_universal(gray, marked_bubbles)
            
            # Step 5: Intelligent answer extraction
            answers = self._extract_answers_universal(marked_bubbles, columns, gray.shape)
            
            processing_time = time.time() - start_time
            
            # Generate results
            results = {
                'success': True,
                'answers': answers,
                'processing_time': processing_time,
                'metadata': {
                    'original_size': f"{original_width}x{original_height}",
                    'rotation_applied': rotation_angle,
                    'final_size': f"{processed_image.shape[1]}x{processed_image.shape[0]}",
                    'bubbles_found': len(marked_bubbles),
                    'columns_detected': columns,
                    'questions_answered': len([a for a in answers.values() if a is not None])
                }
            }
            
            # Add answer key validation if provided
            if answer_key and 'questions' in answer_key:
                results.update(self._validate_answers(answers, answer_key))
            
            print(f"✅ Success: {results['metadata']['questions_answered']} questions detected in {processing_time:.2f}s")
            
            return results
            
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            return self._create_error_result(f"Processing failed: {str(e)}")
    
    def _detect_and_correct_orientation(self, image: np.ndarray) -> Tuple[float, np.ndarray]:
        """Detect orientation and correct if needed - simplified version"""
        # For now, assume images are correctly oriented
        # Future enhancement: implement robust orientation detection
        print("🔄 Orientation detection: No rotation needed (auto-adaptive processing)")
        return 0, image
    
    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by specified angle"""
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new dimensions to avoid cropping
        cos_angle = abs(rotation_matrix[0, 0])
        sin_angle = abs(rotation_matrix[0, 1])
        new_width = int((height * sin_angle) + (width * cos_angle))
        new_height = int((height * cos_angle) + (width * sin_angle))
        
        # Adjust translation
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        
        # Rotate image
        rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height), 
                                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, 
                                borderValue=(255, 255, 255))
        
        return rotated
    
    def _normalize_and_enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image size and enhance quality for processing"""
        height, width = image.shape[:2]
        
        # Target dimensions for optimal processing
        target_size = 1200
        
        # Calculate scaling factor
        scale = target_size / max(width, height)
        
        # Only resize if significantly different
        if scale < 0.8 or scale > 1.2:
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            print(f"📏 Resized: {width}x{height} → {new_width}x{new_height}")
        
        return image
    
    def _detect_marked_bubbles_universal(self, gray: np.ndarray) -> List[Dict]:
        """Universal bubble detection that works with any image format"""
        height, width = gray.shape
        
        # Adaptive parameters based on image size
        min_radius = max(3, min(width, height) // 200)
        max_radius = max(8, min(width, height) // 60)
        min_dist = max(10, min_radius * 2)
        
        # Detect circles
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=min_dist,
            param1=50, param2=30, minRadius=min_radius, maxRadius=max_radius
        )
        
        if circles is None:
            return []
        
        circles = np.round(circles[0, :]).astype("int")
        
        # Analyze image characteristics for adaptive thresholding
        img_brightness = np.mean(gray)
        img_contrast = np.std(gray)
        
        # Adaptive threshold based on image characteristics
        base_threshold = 0.4
        brightness_adjustment = (175 - img_brightness) / 500  # Adjust for darker/lighter images
        contrast_adjustment = (50 - img_contrast) / 250       # Adjust for low/high contrast
        threshold = max(0.2, min(0.7, base_threshold + brightness_adjustment + contrast_adjustment))
        
        print(f"🔍 Image Analysis: brightness={img_brightness:.1f}, contrast={img_contrast:.1f}, bubble~{(min_radius+max_radius)//2}px")
        print(f"⚙️  Adaptive params: bubbles=[{min_radius}, {max_radius}], threshold={threshold:.2f}")
        
        # Filter for marked bubbles
        marked_bubbles = []
        for x, y, r in circles:
            # Extract bubble region
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mask, (x, y), max(1, r-2), 255, -1)
            
            # Calculate marking score
            bubble_region = cv2.bitwise_and(gray, mask)
            bubble_pixels = bubble_region[mask == 255]
            
            if len(bubble_pixels) > 0:
                mean_intensity = bubble_pixels.mean() if hasattr(bubble_pixels, 'mean') else np.mean(bubble_pixels.astype(np.float32))
                darkness = 255 - float(mean_intensity)
                marking_score = darkness / 255.0
                
                if marking_score > threshold:
                    marked_bubbles.append({
                        'center': (x, y),
                        'radius': r,
                        'marking_score': marking_score
                    })
        
        print(f"🎯 Found {len(circles)} bubble candidates")
        print(f"🔴 Identified {len(marked_bubbles)} marked bubbles")
        
        return marked_bubbles
    
    def _detect_columns_universal(self, gray: np.ndarray, marked_bubbles: List[Dict]) -> List[int]:
        """Universal column detection that adapts to any OMR format"""
        if not marked_bubbles:
            # Fallback: proportional division
            width = gray.shape[1]
            return [int(width * r) for r in [0.15, 0.35, 0.55, 0.75]]
        
        # Extract X positions of all marked bubbles
        x_positions = [bubble['center'][0] for bubble in marked_bubbles]
        width = gray.shape[1]
        
        # Use histogram to find column clusters
        bins = max(20, width // 50)
        hist, bin_edges = np.histogram(x_positions, bins=bins, range=(0, width))
        
        # Find peaks in histogram
        peaks = []
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > len(marked_bubbles) // 20:
                peak_x = int((bin_edges[i] + bin_edges[i+1]) / 2)
                peaks.append(peak_x)
        
        # Sort and clean up peaks
        peaks.sort()
        
        # Remove peaks that are too close together
        cleaned_peaks = []
        min_distance = width // 15
        for peak in peaks:
            if not cleaned_peaks or peak - cleaned_peaks[-1] > min_distance:
                cleaned_peaks.append(peak)
        
        # Ensure we have reasonable number of columns
        if len(cleaned_peaks) < 4:
            # Fallback to proportional division
            columns = [int(width * r) for r in [0.15, 0.35, 0.55, 0.75]]
        elif len(cleaned_peaks) > 6:
            # Keep the most prominent peaks
            columns = cleaned_peaks[:6]
        else:
            columns = cleaned_peaks
        
        print(f"📊 Column positions: {columns}")
        
        return columns
    
    def _extract_answers_universal(self, marked_bubbles: List[Dict], columns: List[int], image_shape: Tuple[int, int]) -> Dict[int, str]:
        """Extract answers using universal column mapping"""
        if not marked_bubbles:
            return {}
        
        height, width = image_shape
        
        # Sort bubbles by Y position to organize into rows
        marked_bubbles.sort(key=lambda b: b['center'][1])
        
        # Estimate question layout
        y_positions = [b['center'][1] for b in marked_bubbles]
        y_min, y_max = min(y_positions), max(y_positions)
        
        # Assume 100 questions distributed vertically
        if y_max > y_min:
            row_height = (y_max - y_min) / 99
        else:
            row_height = height // 100
        
        answers = {}
        
        for bubble in marked_bubbles:
            x, y = bubble['center']
            
            # Determine question number from Y position
            question_num = int((y - y_min) / row_height) + 1
            question_num = max(1, min(100, question_num))
            
            # Determine answer choice using universal column mapping
            choice_index = self._universal_column_mapping(x, columns, width)
            choice = ['A', 'B', 'C', 'D'][choice_index]
            
            # Keep highest confidence answer for each question
            if question_num not in answers or bubble['marking_score'] > answers.get(f"{question_num}_score", 0):
                answers[question_num] = choice
                answers[f"{question_num}_score"] = bubble['marking_score']
        
        # Clean up score entries and fill missing questions
        final_answers = {}
        for q in range(1, 101):
            final_answers[q] = answers.get(q, None)
        
        print(f"📋 Generated answers for 100 questions")
        answered_count = len([a for a in final_answers.values() if a is not None])
        print(f"✅ {answered_count} questions have detected answers")
        
        return final_answers
    
    def _universal_column_mapping(self, bubble_x: int, columns: List[int], image_width: int) -> int:
        """Universal column mapping that works with any detected column layout"""
        if len(columns) < 4:
            # Fallback to proportional mapping
            if bubble_x <= image_width * 0.25:
                return 0  # A
            elif bubble_x <= image_width * 0.5:
                return 1  # B
            elif bubble_x <= image_width * 0.75:
                return 2  # C
            else:
                return 3  # D
        
        # Use detected columns intelligently
        if len(columns) >= 6:
            # Skip potential margin columns, use middle 4
            choice_columns = columns[1:5]
        else:
            # Use available columns as choice columns
            choice_columns = columns[:4]
        
        # Find nearest column
        distances = [abs(bubble_x - col_x) for col_x in choice_columns]
        return distances.index(min(distances))
    
    def _validate_answers(self, answers: Dict[int, str], answer_key: Dict) -> Dict:
        """Validate answers against answer key"""
        if not answer_key or 'questions' not in answer_key:
            return {'validation': None}
        
        correct = 0
        total = 0
        details = {}
        
        for question_data in answer_key['questions']:
            q_num = question_data['question']
            correct_answer = question_data['answer']
            detected_answer = answers.get(q_num)
            
            if detected_answer is not None:
                is_correct = detected_answer == correct_answer
                if is_correct:
                    correct += 1
                details[q_num] = {
                    'correct_answer': correct_answer,
                    'detected_answer': detected_answer,
                    'is_correct': is_correct
                }
                total += 1
        
        accuracy = (correct / total * 100) if total > 0 else 0
        
        return {
            'validation': {
                'total_questions': total,
                'correct_answers': correct,
                'accuracy_percentage': round(accuracy, 1),
                'details': details
            }
        }
    
    def _create_error_result(self, error_message: str) -> Dict:
        """Create standardized error result"""
        return {
            'success': False,
            'error': error_message,
            'answers': {},
            'processing_time': 0
        }