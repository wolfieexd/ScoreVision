import json
import os
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
from pathlib import Path
import statistics

class QualityValidator:
    """
    Quality validation system for ensuring 100% accuracy in OMR processing
    
    Features:
    - Answer key validation and integrity checks
    - OMR result quality assessment
    - Image quality analysis
    - Confidence scoring and reliability metrics
    - Automated flagging for manual review
    - Comprehensive reporting
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Quality thresholds for different metrics
        self.thresholds = {
            'image_quality_min': 0.7,
            'detection_confidence_min': 0.8,
            'overall_quality_min': 0.75,
            'bubble_clarity_min': 0.6,
            'perspective_accuracy_min': 0.8,
            'contrast_min': 0.3,
            'sharpness_min': 0.5
        }
        
        # Priority levels for issues
        self.priority_levels = {
            'critical': 'high',
            'warning': 'medium',
            'info': 'low'
        }
    
    def validate_answer_key(self, answer_key: Dict) -> Dict:
        """
        Comprehensive validation of answer key structure and content
        
        Args:
            answer_key: Dictionary containing answer key data
            
        Returns:
            Validation report with detailed findings
        """
        validation_report = {
            'valid': True,
            'issues': [],
            'warnings': [],
            'suggestions': [],
            'metadata': {
                'validation_time': datetime.now().isoformat(),
                'validator_version': '2.0'
            }
        }
        
        try:
            # Check basic structure
            self._validate_basic_structure(answer_key, validation_report)
            
            # Validate metadata
            self._validate_metadata(answer_key, validation_report)
            
            # Validate answers
            self._validate_answers(answer_key, validation_report)
            
            # Check for consistency
            self._validate_consistency(answer_key, validation_report)
            
            # Performance suggestions
            self._generate_suggestions(answer_key, validation_report)
            
            # Determine overall validity
            validation_report['valid'] = len([issue for issue in validation_report['issues'] 
                                            if issue['severity'] == 'critical']) == 0
            
            self.logger.info(f"Answer key validation completed. Valid: {validation_report['valid']}")
            
        except Exception as e:
            self.logger.error(f"Error during answer key validation: {str(e)}")
            validation_report['valid'] = False
            validation_report['issues'].append({
                'type': 'validation_error',
                'severity': 'critical',
                'message': f"Validation process failed: {str(e)}"
            })
        
        return validation_report
    
    def validate_omr_result(self, result: Dict, image_path: str) -> Dict:
        """
        Validate OMR processing results for quality and accuracy
        
        Args:
            result: OMR processing result dictionary
            image_path: Path to the processed OMR image
            
        Returns:
            Quality assessment report
        """
        quality_report = {
            'overall_score': 0.0,
            'quality_grade': 'Unknown',
            'issues': [],
            'recommendations': [],
            'requires_review': False,
            'confidence_metrics': {},
            'image_analysis': {},
            'processing_metrics': {},
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Analyze image quality
            if os.path.exists(image_path):
                quality_report['image_analysis'] = self._analyze_image_quality(image_path)
            
            # Validate result structure
            self._validate_result_structure(result, quality_report)
            
            # Calculate confidence metrics
            quality_report['confidence_metrics'] = self._calculate_confidence_metrics(result)
            
            # Analyze processing quality
            quality_report['processing_metrics'] = self._analyze_processing_quality(result)
            
            # Determine overall quality score
            overall_score = self._calculate_overall_quality_score(quality_report)
            quality_report['overall_score'] = round(overall_score, 3)
            
            # Assign quality grade
            quality_report['quality_grade'] = self._assign_quality_grade(overall_score)
            
            # Determine if manual review is required
            quality_report['requires_review'] = self._requires_manual_review(quality_report)
            
            # Generate recommendations
            self._generate_quality_recommendations(quality_report)
            
            self.logger.info(f"OMR result validation completed. Quality score: {overall_score:.3f}")
            
        except Exception as e:
            self.logger.error(f"Error during OMR result validation: {str(e)}")
            quality_report['issues'].append({
                'type': 'validation_error',
                'severity': 'high',
                'message': f"Quality validation failed: {str(e)}"
            })
            quality_report['requires_review'] = True
        
        return quality_report
    
    def get_all_validation_reports(self, results_folder: str) -> List[Dict]:
        """
        Retrieve all validation reports from processed results
        
        Args:
            results_folder: Path to results directory
            
        Returns:
            List of validation report summaries
        """
        reports = []
        
        try:
            if not os.path.exists(results_folder):
                return reports
            
            for filename in os.listdir(results_folder):
                if filename.endswith('.json'):
                    filepath = os.path.join(results_folder, filename)
                    
                    try:
                        with open(filepath, 'r') as f:
                            result_data = json.load(f)
                        
                        if 'quality_report' in result_data:
                            quality_report = result_data['quality_report']
                            
                            # Create summary report
                            summary = {
                                'filename': filename,
                                'timestamp': result_data.get('timestamp', 'Unknown'),
                                'overall_score': quality_report.get('overall_score', 0.0),
                                'quality_grade': quality_report.get('quality_grade', 'Unknown'),
                                'requires_review': quality_report.get('requires_review', True),
                                'issues_count': len(quality_report.get('issues', [])),
                                'priority': self._determine_priority(quality_report),
                                'omr_file': result_data.get('omr_file', 'Unknown')
                            }
                            
                            reports.append(summary)
                            
                    except Exception as e:
                        self.logger.warning(f"Could not process result file {filename}: {str(e)}")
            
            # Sort by priority and timestamp
            reports.sort(key=lambda x: (
                {'high': 0, 'medium': 1, 'low': 2}.get(x['priority'], 3),
                x['timestamp']
            ), reverse=True)
            
        except Exception as e:
            self.logger.error(f"Error retrieving validation reports: {str(e)}")
        
        return reports
    
    def _validate_basic_structure(self, answer_key: Dict, report: Dict):
        """Validate basic answer key structure"""
        required_keys = ['metadata', 'answers']
        
        for key in required_keys:
            if key not in answer_key:
                report['issues'].append({
                    'type': 'missing_key',
                    'severity': 'critical',
                    'message': f"Required key '{key}' is missing from answer key"
                })
        
        if not isinstance(answer_key.get('answers'), dict):
            report['issues'].append({
                'type': 'invalid_structure',
                'severity': 'critical',
                'message': "Answers must be a dictionary"
            })
    
    def _validate_metadata(self, answer_key: Dict, report: Dict):
        """Validate metadata section"""
        metadata = answer_key.get('metadata', {})
        
        recommended_fields = ['title', 'questions', 'choices', 'created']
        missing_fields = [field for field in recommended_fields if field not in metadata]
        
        if missing_fields:
            report['warnings'].append({
                'type': 'missing_metadata',
                'severity': 'warning',
                'message': f"Recommended metadata fields missing: {', '.join(missing_fields)}"
            })
        
        # Validate numeric fields
        if 'questions' in metadata:
            try:
                questions_count = int(metadata['questions'])
                if questions_count <= 0:
                    report['issues'].append({
                        'type': 'invalid_value',
                        'severity': 'critical',
                        'message': "Number of questions must be positive"
                    })
            except (ValueError, TypeError):
                report['issues'].append({
                    'type': 'invalid_type',
                    'severity': 'critical',
                    'message': "Questions count must be a number"
                })
        
        if 'choices' in metadata:
            try:
                choices_count = int(metadata['choices'])
                if choices_count < 2 or choices_count > 10:
                    report['warnings'].append({
                        'type': 'unusual_value',
                        'severity': 'warning',
                        'message': f"Unusual number of choices: {choices_count} (typically 2-6)"
                    })
            except (ValueError, TypeError):
                report['issues'].append({
                    'type': 'invalid_type',
                    'severity': 'critical',
                    'message': "Choices count must be a number"
                })
    
    def _validate_answers(self, answer_key: Dict, report: Dict):
        """Validate answers section"""
        answers = answer_key.get('answers', {})
        metadata = answer_key.get('metadata', {})
        
        if not answers:
            report['issues'].append({
                'type': 'empty_answers',
                'severity': 'critical',
                'message': "No answers provided"
            })
            return
        
        expected_questions = metadata.get('questions', len(answers))
        expected_choices = metadata.get('choices', 5)
        
        # Check question numbering
        question_numbers = []
        for q_num in answers.keys():
            try:
                question_numbers.append(int(q_num))
            except ValueError:
                report['issues'].append({
                    'type': 'invalid_question_number',
                    'severity': 'critical',
                    'message': f"Invalid question number: {q_num}"
                })
        
        if question_numbers:
            # Check for gaps in sequence
            question_numbers.sort()
            for i in range(len(question_numbers) - 1):
                if question_numbers[i + 1] - question_numbers[i] > 1:
                    report['warnings'].append({
                        'type': 'question_gap',
                        'severity': 'warning',
                        'message': f"Gap in question sequence: {question_numbers[i]} to {question_numbers[i + 1]}"
                    })
        
        # Validate answer values
        for q_num, answer in answers.items():
            if answer is not None:
                try:
                    answer_int = int(answer)
                    if answer_int < 0 or answer_int >= expected_choices:
                        report['issues'].append({
                            'type': 'invalid_answer',
                            'severity': 'critical',
                            'message': f"Answer {answer} for question {q_num} is out of range (0-{expected_choices-1})"
                        })
                except (ValueError, TypeError):
                    report['issues'].append({
                        'type': 'invalid_answer_type',
                        'severity': 'critical',
                        'message': f"Answer for question {q_num} must be a number or null"
                    })
        
        # Check answer count vs metadata
        actual_questions = len(answers)
        if actual_questions != expected_questions:
            report['warnings'].append({
                'type': 'question_count_mismatch',
                'severity': 'warning',
                'message': f"Metadata indicates {expected_questions} questions, but {actual_questions} answers provided"
            })
    
    def _validate_consistency(self, answer_key: Dict, report: Dict):
        """Check for consistency issues"""
        answers = answer_key.get('answers', {})
        
        if not answers:
            return
        
        # Check answer distribution
        valid_answers = [ans for ans in answers.values() if ans is not None]
        
        if valid_answers:
            # Check for extremely uneven distribution
            answer_counts = {}
            for ans in valid_answers:
                answer_counts[ans] = answer_counts.get(ans, 0) + 1
            
            total_answers = len(valid_answers)
            expected_per_choice = total_answers / len(answer_counts)
            
            for choice, count in answer_counts.items():
                if count > expected_per_choice * 2:
                    report['warnings'].append({
                        'type': 'uneven_distribution',
                        'severity': 'warning',
                        'message': f"Choice {choice} appears unusually often ({count} times, {count/total_answers*100:.1f}%)"
                    })
        
        # Check for null answers
        null_count = sum(1 for ans in answers.values() if ans is None)
        if null_count > 0:
            report['warnings'].append({
                'type': 'null_answers',
                'severity': 'warning',
                'message': f"{null_count} questions have no answer specified"
            })
    
    def _generate_suggestions(self, answer_key: Dict, report: Dict):
        """Generate improvement suggestions"""
        metadata = answer_key.get('metadata', {})
        
        if 'title' not in metadata:
            report['suggestions'].append("Add a descriptive title to help identify this answer key")
        
        if 'created' not in metadata:
            report['suggestions'].append("Add creation date for better tracking")
        
        if 'description' not in metadata:
            report['suggestions'].append("Consider adding a description explaining the purpose or context")
    
    def _analyze_image_quality(self, image_path: str) -> Dict:
        """Analyze the quality of the OMR image"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {'error': 'Could not load image'}
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Convert to numpy array for calculations
            gray_array = np.array(gray, dtype=np.float64)
            
            # Calculate various quality metrics
            analysis = {
                'resolution': {'width': image.shape[1], 'height': image.shape[0]},
                'mean_brightness': float(np.mean(gray_array) / 255.0),
                'contrast': float(np.std(gray_array) / 255.0),
                'sharpness': self._calculate_sharpness(gray),
                'noise_level': self._estimate_noise_level(gray),
                'uniformity': self._check_illumination_uniformity(gray)
            }
            
            # Quality assessments
            analysis['quality_scores'] = {
                'brightness_score': self._score_brightness(analysis['mean_brightness']),
                'contrast_score': self._score_contrast(analysis['contrast']),
                'sharpness_score': self._score_sharpness(analysis['sharpness']),
                'noise_score': self._score_noise(analysis['noise_level']),
                'uniformity_score': self._score_uniformity(analysis['uniformity'])
            }
            
            # Overall image quality score
            scores = list(analysis['quality_scores'].values())
            analysis['overall_quality'] = float(np.mean(scores))
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing image quality: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_sharpness(self, gray_image: np.ndarray) -> float:
        """Calculate image sharpness using Laplacian variance"""
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        return float(laplacian.var())
    
    def _estimate_noise_level(self, gray_image: np.ndarray) -> float:
        """Estimate noise level in the image"""
        # Use high-pass filtering to estimate noise
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        filtered = cv2.filter2D(gray_image, -1, kernel)
        filtered_array = np.array(filtered, dtype=np.float64)
        return float(np.std(filtered_array))
    
    def _check_illumination_uniformity(self, gray_image: np.ndarray) -> float:
        """Check uniformity of illumination across the image"""
        # Divide image into blocks and check brightness variation
        h, w = gray_image.shape
        block_size = min(h, w) // 8
        
        block_means = []
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = gray_image[i:i+block_size, j:j+block_size]
                block_means.append(np.mean(block))
        
        if not block_means:
            return 1.0
        
        # Lower std deviation indicates better uniformity
        uniformity = 1.0 - (np.std(block_means) / 255.0)
        return float(max(0.0, uniformity))
    
    def _score_brightness(self, brightness: float) -> float:
        """Score brightness (optimal around 0.5)"""
        return 1.0 - abs(brightness - 0.5) * 2
    
    def _score_contrast(self, contrast: float) -> float:
        """Score contrast (higher is generally better, up to a point)"""
        return min(contrast / 0.3, 1.0)
    
    def _score_sharpness(self, sharpness: float) -> float:
        """Score sharpness (normalized)"""
        return min(sharpness / 1000.0, 1.0)
    
    def _score_noise(self, noise_level: float) -> float:
        """Score noise level (lower is better)"""
        return max(1.0 - noise_level / 100.0, 0.0)
    
    def _score_uniformity(self, uniformity: float) -> float:
        """Score illumination uniformity"""
        return uniformity
    
    def _validate_result_structure(self, result: Dict, report: Dict):
        """Validate the structure of OMR processing results"""
        required_fields = ['total_questions', 'correct_answers', 'percentage_score', 'detailed_results']
        
        for field in required_fields:
            if field not in result:
                report['issues'].append({
                    'type': 'missing_result_field',
                    'severity': 'high',
                    'message': f"Required result field '{field}' is missing"
                })
    
    def _calculate_confidence_metrics(self, result: Dict) -> Dict:
        """Calculate confidence metrics for the processing results"""
        detailed_results = result.get('detailed_results', {})
        
        if not detailed_results:
            return {'overall_confidence': 0.0}
        
        confidences = []
        answered_count = 0
        correct_count = 0
        
        for question_result in detailed_results.values():
            if isinstance(question_result, dict):
                confidence = question_result.get('confidence', 0.0)
                confidences.append(confidence)
                
                if question_result.get('detected_answer') is not None:
                    answered_count += 1
                
                if question_result.get('is_correct', False):
                    correct_count += 1
        
        metrics = {
            'overall_confidence': float(np.mean(confidences)) if confidences else 0.0,
            'min_confidence': float(min(confidences)) if confidences else 0.0,
            'max_confidence': float(max(confidences)) if confidences else 0.0,
            'confidence_std': float(np.std(confidences)) if confidences else 0.0,
            'answered_ratio': answered_count / len(detailed_results) if detailed_results else 0.0,
            'accuracy_ratio': correct_count / len(detailed_results) if detailed_results else 0.0
        }
        
        return metrics
    
    def _analyze_processing_quality(self, result: Dict) -> Dict:
        """Analyze the quality of the processing results"""
        quality_metrics = result.get('quality_metrics', {})
        
        return {
            'image_quality': quality_metrics.get('image_quality', 0.0),
            'detection_confidence': quality_metrics.get('detection_confidence', 0.0),
            'has_issues': len(quality_metrics.get('issues', [])) > 0,
            'issue_count': len(quality_metrics.get('issues', [])),
            'processing_version': result.get('processing_metadata', {}).get('processing_version', 'Unknown')
        }
    
    def _calculate_overall_quality_score(self, quality_report: Dict) -> float:
        """Calculate overall quality score from various metrics"""
        scores = []
        weights = []
        
        # Image analysis score
        image_analysis = quality_report.get('image_analysis', {})
        if 'overall_quality' in image_analysis:
            scores.append(image_analysis['overall_quality'])
            weights.append(0.3)
        
        # Confidence metrics score
        confidence_metrics = quality_report.get('confidence_metrics', {})
        if 'overall_confidence' in confidence_metrics:
            scores.append(confidence_metrics['overall_confidence'])
            weights.append(0.4)
        
        # Processing quality score
        processing_metrics = quality_report.get('processing_metrics', {})
        if 'detection_confidence' in processing_metrics:
            scores.append(processing_metrics['detection_confidence'])
            weights.append(0.3)
        
        if not scores:
            return 0.0
        
        # Calculate weighted average
        weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
        total_weight = sum(weights)
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _assign_quality_grade(self, score: float) -> str:
        """Assign quality grade based on score"""
        if score >= 0.9:
            return 'Excellent'
        elif score >= 0.8:
            return 'Good'
        elif score >= 0.7:
            return 'Acceptable'
        elif score >= 0.6:
            return 'Poor'
        else:
            return 'Critical'
    
    def _requires_manual_review(self, quality_report: Dict) -> bool:
        """Determine if manual review is required"""
        overall_score = quality_report.get('overall_score', 0.0)
        
        # Low overall score requires review
        if overall_score < self.thresholds['overall_quality_min']:
            return True
        
        # High priority issues require review
        issues = quality_report.get('issues', [])
        high_priority_issues = [issue for issue in issues if issue.get('severity') == 'high']
        
        if high_priority_issues:
            return True
        
        # Low confidence in key metrics
        confidence_metrics = quality_report.get('confidence_metrics', {})
        if confidence_metrics.get('overall_confidence', 1.0) < self.thresholds['detection_confidence_min']:
            return True
        
        return False
    
    def _generate_quality_recommendations(self, quality_report: Dict):
        """Generate recommendations for improving quality"""
        recommendations = []
        
        overall_score = quality_report.get('overall_score', 0.0)
        image_analysis = quality_report.get('image_analysis', {})
        
        # Image quality recommendations
        if image_analysis.get('quality_scores', {}).get('brightness_score', 1.0) < 0.7:
            recommendations.append("Improve lighting conditions - image appears too dark or too bright")
        
        if image_analysis.get('quality_scores', {}).get('contrast_score', 1.0) < 0.7:
            recommendations.append("Increase contrast - marks may not be clearly visible")
        
        if image_analysis.get('quality_scores', {}).get('sharpness_score', 1.0) < 0.7:
            recommendations.append("Ensure image is in focus - blurry images reduce accuracy")
        
        if image_analysis.get('quality_scores', {}).get('uniformity_score', 1.0) < 0.7:
            recommendations.append("Improve lighting uniformity - avoid shadows and uneven illumination")
        
        # Processing recommendations
        confidence_metrics = quality_report.get('confidence_metrics', {})
        if confidence_metrics.get('answered_ratio', 1.0) < 0.9:
            recommendations.append("Some questions appear unanswered - verify all required answers are marked")
        
        if overall_score < 0.8:
            recommendations.append("Consider manual verification of results due to quality concerns")
        
        quality_report['recommendations'] = recommendations
    
    def _determine_priority(self, quality_report: Dict) -> str:
        """Determine priority level for a quality report"""
        if quality_report.get('requires_review', False):
            return 'high'
        
        overall_score = quality_report.get('overall_score', 0.0)
        if overall_score < 0.7:
            return 'medium'
        
        return 'low'