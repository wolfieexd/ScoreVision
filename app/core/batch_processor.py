import os
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Callable
import logging
from datetime import datetime
from pathlib import Path
import zipfile
import shutil
from queue import Queue
import traceback

from omr_processor import OMRProcessor
from quality_validator import QualityValidator

class BatchProcessor:
    """
    Advanced batch processing system for handling multiple OMR sheets
    
    Features:
    - Parallel processing for optimal performance
    - Real-time progress tracking with detailed status
    - Comprehensive error handling and recovery
    - Quality validation integration
    - Batch result compilation and reporting
    - Resume capability for interrupted processing
    - Resource management and optimization
    """
    
    def __init__(self, max_workers: int = 4):
        self.logger = logging.getLogger(__name__)
        self.max_workers = max_workers
        
        # Initialize core components
        self.omr_processor = OMRProcessor()
        self.quality_validator = QualityValidator()
        
        # Processing state management
        self.processing_state = {
            'is_processing': False,
            'current_batch': None,
            'start_time': None,
            'progress': {
                'total_files': 0,
                'processed_files': 0,
                'successful_files': 0,
                'failed_files': 0,
                'skipped_files': 0
            },
            'current_file': None,
            'estimated_completion': None
        }
        
        # Results storage
        self.batch_results = {
            'summary': {},
            'individual_results': [],
            'failed_files': [],
            'quality_reports': []
        }
        
        # Progress callback function
        self.progress_callback: Optional[Callable] = None
        
        # Thread synchronization
        self.processing_lock = threading.Lock()
        self.stop_processing = threading.Event()
    
    def process_batch(self, file_paths: List[str], answer_key: Dict, 
                     output_folder: str, batch_name: Optional[str] = None,
                     progress_callback: Optional[Callable] = None) -> Dict:
        """
        Process multiple OMR files in batch with advanced features
        
        Args:
            file_paths: List of paths to OMR image files
            answer_key: Answer key dictionary for evaluation
            output_folder: Directory to save results
            batch_name: Optional name for this batch
            progress_callback: Function to call with progress updates
            
        Returns:
            Comprehensive batch processing results
        """
        with self.processing_lock:
            if self.processing_state['is_processing']:
                raise RuntimeError("Batch processing already in progress")
            
            self.processing_state['is_processing'] = True
            self.stop_processing.clear()
        
        try:
            # Initialize batch processing
            batch_id = self._initialize_batch(file_paths, batch_name, output_folder)
            self.progress_callback = progress_callback
            
            self.logger.info(f"Starting batch processing: {batch_id}")
            self.logger.info(f"Files to process: {len(file_paths)}")
            self.logger.info(f"Max workers: {self.max_workers}")
            
            # Validate answer key before processing
            key_validation = self.quality_validator.validate_answer_key(answer_key)
            if not key_validation['valid']:
                raise ValueError(f"Invalid answer key: {key_validation['issues'][0]['message']}")
            
            # Create output directories
            batch_output_dir = os.path.join(output_folder, f"batch_{batch_id}")
            os.makedirs(batch_output_dir, exist_ok=True)
            os.makedirs(os.path.join(batch_output_dir, "individual_results"), exist_ok=True)
            os.makedirs(os.path.join(batch_output_dir, "quality_reports"), exist_ok=True)
            
            # Process files with parallel execution
            results = self._process_files_parallel(file_paths, answer_key, batch_output_dir)
            
            # Compile comprehensive batch report
            batch_report = self._compile_batch_report(results, batch_id, answer_key)
            
            # Save batch summary
            summary_path = os.path.join(batch_output_dir, "batch_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(batch_report, f, indent=2)
            
            # Generate batch quality report
            quality_summary = self._generate_batch_quality_summary(results)
            quality_path = os.path.join(batch_output_dir, "quality_summary.json")
            with open(quality_path, 'w') as f:
                json.dump(quality_summary, f, indent=2)
            
            # Create downloadable ZIP if successful
            if batch_report['summary']['successful_files'] > 0:
                zip_path = self._create_results_zip(batch_output_dir, batch_id)
                batch_report['download_path'] = zip_path
            
            self.logger.info(f"Batch processing completed: {batch_id}")
            self.logger.info(f"Success rate: {batch_report['summary']['success_rate']:.1f}%")
            
            return batch_report
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            error_report = {
                'success': False,
                'error': str(e),
                'batch_id': getattr(self, '_current_batch_id', 'unknown'),
                'timestamp': datetime.now().isoformat()
            }
            
            # Call progress callback with error
            if self.progress_callback:
                try:
                    self.progress_callback({
                        'status': 'error',
                        'error': str(e),
                        'progress': self.processing_state['progress']
                    })
                except Exception as callback_error:
                    self.logger.error(f"Progress callback failed: {str(callback_error)}")
            
            return error_report
            
        finally:
            # Reset processing state
            with self.processing_lock:
                self.processing_state['is_processing'] = False
                self.processing_state['current_batch'] = None
                self.processing_state['current_file'] = None
    
    def get_processing_status(self) -> Dict:
        """
        Get current processing status and progress information
        
        Returns:
            Current processing status with detailed metrics
        """
        with self.processing_lock:
            status = {
                'is_processing': self.processing_state['is_processing'],
                'progress': self.processing_state['progress'].copy(),
                'current_file': self.processing_state['current_file'],
                'batch_info': {
                    'batch_id': self.processing_state['current_batch'],
                    'start_time': self.processing_state['start_time'],
                    'estimated_completion': self.processing_state['estimated_completion']
                },
                'performance_metrics': self._calculate_performance_metrics()
            }
            
            # Calculate additional derived metrics
            progress = status['progress']
            if progress['total_files'] > 0:
                status['completion_percentage'] = (progress['processed_files'] / progress['total_files']) * 100
                status['success_rate'] = (progress['successful_files'] / max(progress['processed_files'], 1)) * 100
            else:
                status['completion_percentage'] = 0.0
                status['success_rate'] = 0.0
            
            return status
    
    def stop_batch_processing(self) -> bool:
        """
        Stop the current batch processing gracefully
        
        Returns:
            True if stop signal was sent successfully
        """
        if self.processing_state['is_processing']:
            self.logger.info("Stop signal sent to batch processor")
            self.stop_processing.set()
            return True
        return False
    
    def get_batch_history(self, results_folder: str, limit: int = 50) -> List[Dict]:
        """
        Retrieve history of batch processing sessions
        
        Args:
            results_folder: Directory containing batch results
            limit: Maximum number of sessions to return
            
        Returns:
            List of batch session summaries
        """
        history = []
        
        try:
            if not os.path.exists(results_folder):
                return history
            
            # Find all batch directories
            batch_dirs = []
            for item in os.listdir(results_folder):
                if item.startswith('batch_') and os.path.isdir(os.path.join(results_folder, item)):
                    batch_dirs.append(item)
            
            # Sort by creation time (newest first)
            batch_dirs.sort(key=lambda x: os.path.getctime(os.path.join(results_folder, x)), reverse=True)
            
            # Process each batch directory
            for batch_dir in batch_dirs[:limit]:
                batch_path = os.path.join(results_folder, batch_dir)
                summary_path = os.path.join(batch_path, "batch_summary.json")
                
                try:
                    if os.path.exists(summary_path):
                        with open(summary_path, 'r') as f:
                            batch_data = json.load(f)
                        
                        # Create history entry
                        history_entry = {
                            'batch_id': batch_data.get('batch_id', batch_dir),
                            'batch_name': batch_data.get('batch_name', 'Unnamed Batch'),
                            'timestamp': batch_data.get('timestamp', 'Unknown'),
                            'total_files': batch_data.get('summary', {}).get('total_files', 0),
                            'successful_files': batch_data.get('summary', {}).get('successful_files', 0),
                            'success_rate': batch_data.get('summary', {}).get('success_rate', 0.0),
                            'processing_time': batch_data.get('summary', {}).get('total_processing_time', 0.0),
                            'average_quality_score': batch_data.get('quality_metrics', {}).get('average_quality_score', 0.0),
                            'has_download': os.path.exists(os.path.join(batch_path, f"{batch_data.get('batch_id', batch_dir)}_results.zip")),
                            'folder_path': batch_path
                        }
                        
                        history.append(history_entry)
                        
                except Exception as e:
                    self.logger.warning(f"Could not process batch directory {batch_dir}: {str(e)}")
                    
        except Exception as e:
            self.logger.error(f"Error retrieving batch history: {str(e)}")
        
        return history
    
    def _initialize_batch(self, file_paths: List[str], batch_name: Optional[str], output_folder: str) -> str:
        """Initialize batch processing session"""
        batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._current_batch_id = batch_id
        
        # Use provided batch name or generate one
        if batch_name is None:
            batch_name = f"batch_{batch_id}"
        
        # Filter valid image files
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        valid_files = [
            path for path in file_paths 
            if os.path.exists(path) and Path(path).suffix.lower() in valid_extensions
        ]
        
        # Update processing state
        self.processing_state.update({
            'current_batch': batch_id,
            'start_time': datetime.now().isoformat(),
            'progress': {
                'total_files': len(valid_files),
                'processed_files': 0,
                'successful_files': 0,
                'failed_files': 0,
                'skipped_files': len(file_paths) - len(valid_files)
            }
        })
        
        # Initialize batch results
        self.batch_results = {
            'batch_id': batch_id,
            'batch_name': batch_name or f"Batch_{batch_id}",
            'summary': {},
            'individual_results': [],
            'failed_files': [],
            'quality_reports': []
        }
        
        return batch_id
    
    def _process_files_parallel(self, file_paths: List[str], answer_key: Dict, output_dir: str) -> List[Dict]:
        """Process files using parallel execution"""
        results = []
        
        # Filter valid files
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        valid_files = [
            path for path in file_paths 
            if os.path.exists(path) and Path(path).suffix.lower() in valid_extensions
        ]
        
        # Create thread pool for parallel processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all processing tasks
            future_to_file = {
                executor.submit(self._process_single_file, file_path, answer_key, output_dir): file_path
                for file_path in valid_files
            }
            
            # Process completed tasks
            for future in as_completed(future_to_file):
                if self.stop_processing.is_set():
                    self.logger.info("Stopping batch processing as requested")
                    break
                
                file_path = future_to_file[future]
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Update progress
                    with self.processing_lock:
                        self.processing_state['progress']['processed_files'] += 1
                        if result['success']:
                            self.processing_state['progress']['successful_files'] += 1
                        else:
                            self.processing_state['progress']['failed_files'] += 1
                        
                        # Update estimated completion time
                        self._update_estimated_completion()
                    
                    # Call progress callback
                    if self.progress_callback:
                        try:
                            self.progress_callback({
                                'status': 'processing',
                                'current_file': os.path.basename(file_path),
                                'progress': self.processing_state['progress'].copy(),
                                'estimated_completion': self.processing_state['estimated_completion']
                            })
                        except Exception as e:
                            self.logger.warning(f"Progress callback failed: {str(e)}")
                    
                except Exception as e:
                    self.logger.error(f"Error processing {file_path}: {str(e)}")
                    
                    # Create error result
                    error_result = {
                        'file_path': file_path,
                        'filename': os.path.basename(file_path),
                        'success': False,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }
                    results.append(error_result)
                    
                    # Update progress
                    with self.processing_lock:
                        self.processing_state['progress']['processed_files'] += 1
                        self.processing_state['progress']['failed_files'] += 1
        
        return results
    
    def _process_single_file(self, file_path: str, answer_key: Dict, output_dir: str) -> Dict:
        """Process a single OMR file with comprehensive quality validation"""
        filename = os.path.basename(file_path)
        
        try:
            # Update current processing file
            with self.processing_lock:
                self.processing_state['current_file'] = filename
            
            self.logger.info(f"Processing file: {filename}")
            
            # Process OMR image
            processing_result = self.omr_processor.process_omr_sheet(file_path, answer_key)
            
            if not processing_result['success']:
                return {
                    'file_path': file_path,
                    'filename': filename,
                    'success': False,
                    'error': processing_result.get('error', 'Unknown processing error'),
                    'timestamp': datetime.now().isoformat()
                }
            
            # Validate processing quality
            quality_report = self.quality_validator.validate_omr_result(
                processing_result, file_path
            )
            
            # Prepare comprehensive result
            result = {
                'file_path': file_path,
                'filename': filename,
                'success': True,
                'omr_result': processing_result,
                'quality_report': quality_report,
                'requires_review': quality_report.get('requires_review', False),
                'quality_score': quality_report.get('overall_score', 0.0),
                'timestamp': datetime.now().isoformat()
            }
            
            # Save individual result file
            result_filename = f"{Path(filename).stem}_result.json"
            result_path = os.path.join(output_dir, "individual_results", result_filename)
            
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            # Save quality report
            quality_filename = f"{Path(filename).stem}_quality.json"
            quality_path = os.path.join(output_dir, "quality_reports", quality_filename)
            
            with open(quality_path, 'w') as f:
                json.dump(quality_report, f, indent=2)
            
            self.logger.info(f"Successfully processed: {filename} (Quality: {quality_report.get('overall_score', 0.0):.3f})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing file {filename}: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            return {
                'file_path': file_path,
                'filename': filename,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _update_estimated_completion(self):
        """Update estimated completion time based on current progress"""
        progress = self.processing_state['progress']
        
        if progress['processed_files'] > 0 and self.processing_state['start_time']:
            start_time = datetime.fromisoformat(self.processing_state['start_time'])
            elapsed_time = (datetime.now() - start_time).total_seconds()
            
            avg_time_per_file = elapsed_time / progress['processed_files']
            remaining_files = progress['total_files'] - progress['processed_files']
            estimated_remaining_time = remaining_files * avg_time_per_file
            
            estimated_completion = datetime.now().timestamp() + estimated_remaining_time
            self.processing_state['estimated_completion'] = datetime.fromtimestamp(estimated_completion).isoformat()
    
    def _calculate_performance_metrics(self) -> Dict:
        """Calculate performance metrics for current processing session"""
        if not self.processing_state['start_time']:
            return {}
        
        start_time = datetime.fromisoformat(self.processing_state['start_time'])
        elapsed_time = (datetime.now() - start_time).total_seconds()
        
        progress = self.processing_state['progress']
        
        metrics = {
            'elapsed_time_seconds': elapsed_time,
            'files_per_minute': (progress['processed_files'] / max(elapsed_time, 1)) * 60,
            'average_processing_time': elapsed_time / max(progress['processed_files'], 1),
            'estimated_total_time': None
        }
        
        if progress['processed_files'] > 0:
            avg_time_per_file = elapsed_time / progress['processed_files']
            estimated_total = avg_time_per_file * progress['total_files']
            metrics['estimated_total_time'] = estimated_total
        
        return metrics
    
    def _compile_batch_report(self, results: List[Dict], batch_id: str, answer_key: Dict) -> Dict:
        """Compile comprehensive batch processing report"""
        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]
        
        # Calculate summary statistics
        total_processing_time = 0
        if self.processing_state['start_time']:
            start_time = datetime.fromisoformat(self.processing_state['start_time'])
            total_processing_time = (datetime.now() - start_time).total_seconds()
        
        summary = {
            'total_files': len(results),
            'successful_files': len(successful_results),
            'failed_files': len(failed_results),
            'success_rate': (len(successful_results) / max(len(results), 1)) * 100,
            'total_processing_time': total_processing_time,
            'average_processing_time': total_processing_time / max(len(results), 1)
        }
        
        # Calculate quality metrics
        quality_scores = [r.get('quality_score', 0.0) for r in successful_results if 'quality_score' in r]
        review_required = [r for r in successful_results if r.get('requires_review', False)]
        
        quality_metrics = {
            'average_quality_score': sum(quality_scores) / max(len(quality_scores), 1),
            'min_quality_score': min(quality_scores) if quality_scores else 0.0,
            'max_quality_score': max(quality_scores) if quality_scores else 0.0,
            'files_requiring_review': len(review_required),
            'review_percentage': (len(review_required) / max(len(successful_results), 1)) * 100
        }
        
        # Compile full report
        report = {
            'batch_id': batch_id,
            'batch_name': self.batch_results['batch_name'],
            'timestamp': datetime.now().isoformat(),
            'success': len(failed_results) < len(results),  # Success if at least one file processed
            'summary': summary,
            'quality_metrics': quality_metrics,
            'answer_key_info': {
                'total_questions': answer_key.get('metadata', {}).get('questions', 'Unknown'),
                'choices_per_question': answer_key.get('metadata', {}).get('choices', 'Unknown'),
                'title': answer_key.get('metadata', {}).get('title', 'Unnamed Answer Key')
            },
            'individual_results': [
                {
                    'filename': r['filename'],
                    'success': r['success'],
                    'quality_score': r.get('quality_score', 0.0),
                    'requires_review': r.get('requires_review', False),
                    'error': r.get('error') if not r['success'] else None
                }
                for r in results
            ],
            'failed_files': [
                {
                    'filename': r['filename'],
                    'error': r.get('error', 'Unknown error')
                }
                for r in failed_results
            ]
        }
        
        return report
    
    def _generate_batch_quality_summary(self, results: List[Dict]) -> Dict:
        """Generate comprehensive quality summary for the batch"""
        successful_results = [r for r in results if r['success']]
        
        if not successful_results:
            return {'message': 'No successful results to analyze'}
        
        # Collect quality data
        quality_scores = []
        image_qualities = []
        confidence_scores = []
        review_required_files = []
        
        for result in successful_results:
            quality_report = result.get('quality_report', {})
            
            # Overall quality
            quality_scores.append(quality_report.get('overall_score', 0.0))
            
            # Image quality
            image_analysis = quality_report.get('image_analysis', {})
            if 'overall_quality' in image_analysis:
                image_qualities.append(image_analysis['overall_quality'])
            
            # Confidence metrics
            confidence_metrics = quality_report.get('confidence_metrics', {})
            if 'overall_confidence' in confidence_metrics:
                confidence_scores.append(confidence_metrics['overall_confidence'])
            
            # Files requiring review
            if quality_report.get('requires_review', False):
                review_required_files.append(result['filename'])
        
        # Calculate statistics
        quality_summary = {
            'total_analyzed_files': len(successful_results),
            'quality_statistics': {
                'average_score': sum(quality_scores) / len(quality_scores),
                'min_score': min(quality_scores),
                'max_score': max(quality_scores),
                'score_distribution': self._calculate_score_distribution(quality_scores)
            },
            'image_quality_stats': {
                'average': sum(image_qualities) / max(len(image_qualities), 1),
                'min': min(image_qualities) if image_qualities else 0.0,
                'max': max(image_qualities) if image_qualities else 0.0
            },
            'confidence_stats': {
                'average': sum(confidence_scores) / max(len(confidence_scores), 1),
                'min': min(confidence_scores) if confidence_scores else 0.0,
                'max': max(confidence_scores) if confidence_scores else 0.0
            },
            'review_summary': {
                'files_requiring_review': len(review_required_files),
                'review_percentage': (len(review_required_files) / len(successful_results)) * 100,
                'review_files': review_required_files
            },
            'recommendations': self._generate_batch_recommendations(quality_scores, image_qualities, confidence_scores)
        }
        
        return quality_summary
    
    def _calculate_score_distribution(self, scores: List[float]) -> Dict:
        """Calculate distribution of quality scores"""
        if not scores:
            return {}
        
        ranges = {
            'excellent': len([s for s in scores if s >= 0.9]),
            'good': len([s for s in scores if 0.8 <= s < 0.9]),
            'acceptable': len([s for s in scores if 0.7 <= s < 0.8]),
            'poor': len([s for s in scores if 0.6 <= s < 0.7]),
            'critical': len([s for s in scores if s < 0.6])
        }
        
        total = len(scores)
        return {
            grade: {
                'count': count,
                'percentage': (count / total) * 100
            }
            for grade, count in ranges.items()
        }
    
    def _generate_batch_recommendations(self, quality_scores: List[float], 
                                      image_qualities: List[float], 
                                      confidence_scores: List[float]) -> List[str]:
        """Generate recommendations for improving batch processing quality"""
        recommendations = []
        
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            if avg_quality < 0.8:
                recommendations.append("Consider improving scan quality - average quality score is below optimal")
        
        if image_qualities:
            avg_image_quality = sum(image_qualities) / len(image_qualities)
            if avg_image_quality < 0.7:
                recommendations.append("Image quality issues detected - ensure proper lighting and focus")
        
        if confidence_scores:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            if avg_confidence < 0.8:
                recommendations.append("Low confidence in bubble detection - verify answer sheet alignment")
        
        # Check for consistency
        if quality_scores and len(set(round(score, 1) for score in quality_scores)) == 1:
            recommendations.append("Consider varying scan conditions to validate processing robustness")
        
        return recommendations
    
    def _create_results_zip(self, batch_output_dir: str, batch_id: str) -> Optional[str]:
        """Create downloadable ZIP file with all batch results"""
        zip_filename = f"{batch_id}_results.zip"
        zip_path = os.path.join(batch_output_dir, zip_filename)
        
        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add batch summary
                summary_path = os.path.join(batch_output_dir, "batch_summary.json")
                if os.path.exists(summary_path):
                    zipf.write(summary_path, "batch_summary.json")
                
                # Add quality summary
                quality_path = os.path.join(batch_output_dir, "quality_summary.json")
                if os.path.exists(quality_path):
                    zipf.write(quality_path, "quality_summary.json")
                
                # Add individual results
                results_dir = os.path.join(batch_output_dir, "individual_results")
                if os.path.exists(results_dir):
                    for filename in os.listdir(results_dir):
                        file_path = os.path.join(results_dir, filename)
                        zipf.write(file_path, f"individual_results/{filename}")
                
                # Add quality reports
                quality_dir = os.path.join(batch_output_dir, "quality_reports")
                if os.path.exists(quality_dir):
                    for filename in os.listdir(quality_dir):
                        file_path = os.path.join(quality_dir, filename)
                        zipf.write(file_path, f"quality_reports/{filename}")
            
            self.logger.info(f"Created results ZIP: {zip_path}")
            return zip_path
            
        except Exception as e:
            self.logger.error(f"Error creating results ZIP: {str(e)}")
            return None