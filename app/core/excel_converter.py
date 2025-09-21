import pandas as pd
import json
import numpy as np
from typing import Dict, List, Union, Any, Optional
import logging
from datetime import datetime
import openpyxl
from io import BytesIO

class ExcelConverter:
    """
    Excel to JSON converter for OMR answer keys
    
    Supports multiple Excel formats and layouts:
    - Standard format: Question Number | Correct Answer
    - Extended format: Question | Choice A | Choice B | Choice C | Choice D | Choice E | Correct Answer
    - Multiple choice format with marking (X, 1, True, etc.)
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Supported answer representations
        self.positive_indicators = ['X', 'x', '1', 'TRUE', 'True', 'true', 'YES', 'Yes', 'yes', '✓', '*']
        self.choice_mappings = {
            'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
            'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5,
            '1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5,
            0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5
        }
    
    def parse_excel(self, file_obj) -> Dict[str, Dict]:
        """
        Parse Excel file and extract answer keys from all sheets
        
        Args:
            file_obj: File object or path to Excel file
            
        Returns:
            Dictionary with sheet names as keys and answer key data as values
        """
        try:
            # Read Excel file with all sheets
            if hasattr(file_obj, 'read'):
                # File object
                file_obj.seek(0)
                excel_data = pd.read_excel(file_obj, sheet_name=None, engine='openpyxl')
            else:
                # File path
                excel_data = pd.read_excel(file_obj, sheet_name=None, engine='openpyxl')
            
            parsed_sheets = {}
            
            for sheet_name, df in excel_data.items():
                self.logger.info(f"Processing sheet: {sheet_name}")
                
                # Clean the dataframe
                df = self._clean_dataframe(df)
                
                if df.empty:
                    self.logger.warning(f"Sheet '{sheet_name}' is empty, skipping")
                    continue
                
                # Detect format and parse accordingly
                answer_key = self._detect_and_parse_format(df, sheet_name)
                
                if answer_key:
                    parsed_sheets[sheet_name] = answer_key
                else:
                    self.logger.warning(f"Could not parse sheet '{sheet_name}'")
            
            return parsed_sheets
            
        except Exception as e:
            self.logger.error(f"Error parsing Excel file: {str(e)}")
            raise
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataframe by removing empty rows and columns
        """
        # Remove completely empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Reset index
        df = df.reset_index(drop=True)
        
        # Convert column names to strings and strip whitespace
        df.columns = [str(col).strip() if col is not None else f'Column_{i}' 
                      for i, col in enumerate(df.columns)]
        
        return df
    
    def _detect_and_parse_format(self, df: pd.DataFrame, sheet_name: str) -> Dict:
        """
        Detect the format of the Excel sheet and parse accordingly
        """
        if df.empty:
            return {
                'format': 'unknown',
                'questions': {},
                'metadata': {
                    'total_questions': 0,
                    'sheet_name': sheet_name,
                    'error': 'Empty dataframe'
                }
            }
        
        # Try different parsing strategies
        formats_to_try = [
            self._parse_standard_format,
            self._parse_choice_matrix_format,
            self._parse_extended_format,
            self._parse_custom_format
        ]
        
        for parse_method in formats_to_try:
            try:
                result = parse_method(df, sheet_name)
                if result and result.get('answers'):
                    self.logger.info(f"Successfully parsed '{sheet_name}' using {parse_method.__name__}")
                    return result
            except Exception as e:
                self.logger.debug(f"Failed to parse with {parse_method.__name__}: {str(e)}")
                continue
        
        return {
            'format': 'unknown',
            'questions': {},
            'metadata': {
                'total_questions': 0,
                'sheet_name': sheet_name,
                'error': 'No compatible format found'
            }
        }
    
    def _parse_standard_format(self, df: pd.DataFrame, sheet_name: str) -> Dict:
        """
        Parse standard format: Question Number | Correct Answer
        Expected columns: Question, Answer (or similar variations)
        """
        # Look for question and answer columns
        question_col = self._find_column(df, ['question', 'q', 'question_number', 'qno', 'number'])
        answer_col = self._find_column(df, ['answer', 'correct', 'correct_answer', 'key', 'solution'])
        
        if question_col is None or answer_col is None:
            raise ValueError("Could not find question and answer columns")
        
        answers = {}
        choices_count = 0
        
        for _, row in df.iterrows():
            question_num = self._normalize_question_number(row[question_col])
            answer_value = row[answer_col]
            
            if question_num is None or pd.isna(answer_value):
                continue
            
            # Convert answer to choice index
            choice_index = self._convert_answer_to_index(answer_value)
            
            if choice_index is not None:
                answers[str(question_num)] = choice_index
                choices_count = max(choices_count, choice_index + 1)
        
        if not answers:
            raise ValueError("No valid answers found")
        
        return {
            'metadata': {
                'title': f'Answer Key - {sheet_name}',
                'source': 'Excel Import',
                'format': 'Standard',
                'questions': len(answers),
                'choices': max(choices_count, 4),  # Minimum 4 choices
                'created': datetime.now().isoformat()
            },
            'answers': answers
        }
    
    def _parse_choice_matrix_format(self, df: pd.DataFrame, sheet_name: str) -> Dict:
        """
        Parse choice matrix format: Question | A | B | C | D | E
        Where choices are marked with X, 1, True, etc.
        """
        # Look for question column
        question_col = self._find_column(df, ['question', 'q', 'question_number', 'qno', 'number'])
        
        if question_col is None:
            raise ValueError("Could not find question column")
        
        # Find choice columns (A, B, C, D, etc.)
        choice_columns = []
        for col in df.columns:
            col_name = str(col).strip().upper()
            if col_name in ['A', 'B', 'C', 'D', 'E', 'F'] or col_name.startswith('CHOICE'):
                choice_columns.append(col)
        
        if len(choice_columns) < 2:
            raise ValueError("Not enough choice columns found")
        
        answers = {}
        
        for _, row in df.iterrows():
            question_num = self._normalize_question_number(row[question_col])
            
            if pd.isna(question_num):
                continue
            
            # Find marked choice
            marked_choice = None
            for i, choice_col in enumerate(choice_columns):
                cell_value = row[choice_col]
                
                if self._is_positive_indicator(cell_value):
                    if marked_choice is not None:
                        self.logger.warning(f"Multiple answers marked for question {question_num}")
                    marked_choice = i
            
            if marked_choice is not None:
                answers[str(question_num)] = marked_choice
        
        if not answers:
            raise ValueError("No valid answers found")
        
        return {
            'metadata': {
                'title': f'Answer Key - {sheet_name}',
                'source': 'Excel Import',
                'format': 'Choice Matrix',
                'questions': len(answers),
                'choices': len(choice_columns),
                'created': datetime.now().isoformat()
            },
            'answers': answers
        }
    
    def _parse_extended_format(self, df: pd.DataFrame, sheet_name: str) -> Dict:
        """
        Parse extended format with multiple columns including metadata
        """
        # Look for essential columns
        question_col = self._find_column(df, ['question', 'q', 'question_number', 'qno'])
        answer_col = self._find_column(df, ['answer', 'correct', 'correct_answer', 'key'])
        
        if question_col is None or answer_col is None:
            raise ValueError("Could not find required columns")
        
        # Optional columns
        explanation_col = self._find_column(df, ['explanation', 'rationale', 'reason'])
        difficulty_col = self._find_column(df, ['difficulty', 'level', 'complexity'])
        category_col = self._find_column(df, ['category', 'topic', 'subject'])
        
        answers = {}
        questions_metadata = {}
        choices_count = 0
        
        for _, row in df.iterrows():
            question_num = self._normalize_question_number(row[question_col])
            answer_value = row[answer_col]
            
            if question_num is None or pd.isna(answer_value):
                continue
            
            # Convert answer to choice index
            choice_index = self._convert_answer_to_index(answer_value)
            
            if choice_index is not None:
                answers[str(question_num)] = choice_index
                choices_count = max(choices_count, choice_index + 1)
                
                # Store metadata if available
                metadata = {}
                if explanation_col:
                    try:
                        exp_val = row[explanation_col]
                        if exp_val is not None and str(exp_val).strip():
                            metadata['explanation'] = str(exp_val)
                    except:
                        pass
                if difficulty_col:
                    try:
                        diff_val = row[difficulty_col]
                        if diff_val is not None and str(diff_val).strip():
                            metadata['difficulty'] = str(diff_val)
                    except:
                        pass
                if category_col:
                    try:
                        cat_val = row[category_col]
                        if cat_val is not None and str(cat_val).strip():
                            metadata['category'] = str(cat_val)
                    except:
                        pass
                
                if metadata:
                    questions_metadata[str(question_num)] = metadata
        
        if not answers:
            raise ValueError("No valid answers found")
        
        result = {
            'metadata': {
                'title': f'Answer Key - {sheet_name}',
                'source': 'Excel Import',
                'format': 'Extended',
                'questions': len(answers),
                'choices': max(choices_count, 4),
                'created': datetime.now().isoformat()
            },
            'answers': answers
        }
        
        if questions_metadata:
            result['questions_metadata'] = questions_metadata
        
        return result
    
    def _parse_custom_format(self, df: pd.DataFrame, sheet_name: str) -> Dict:
        """
        Attempt to parse custom or non-standard formats
        """
        # Try to infer structure from the first few rows
        if len(df) < 2:
            raise ValueError("Not enough data to infer format")
        
        # Look for any numeric sequence that could be question numbers
        for col in df.columns:
            values = df[col].dropna()
            if len(values) >= 2:
                # Check if this could be a question number sequence
                try:
                    numeric_values = pd.to_numeric(values, errors='coerce').dropna()
                    if len(numeric_values) >= 2 and self._is_sequential(numeric_values):
                        # Found potential question column
                        return self._parse_with_question_column(df, col, sheet_name)
                except:
                    continue
        
        raise ValueError("Could not determine format")
    
    def _parse_with_question_column(self, df: pd.DataFrame, question_col: str, sheet_name: str) -> Dict:
        """
        Parse using identified question column
        """
        # Find the most likely answer column (next non-empty column)
        answer_col = None
        for col in df.columns:
            if col != question_col and not df[col].dropna().empty:
                answer_col = col
                break
        
        if answer_col is None:
            raise ValueError("Could not find answer column")
        
        return self._parse_standard_format(df.rename(columns={
            question_col: 'question',
            answer_col: 'answer'
        }), sheet_name)
    
    def _find_column(self, df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
        """
        Find column by checking possible names (case-insensitive)
        """
        df_columns_lower = [str(col).lower().strip() for col in df.columns]
        
        for name in possible_names:
            name_lower = name.lower()
            for i, col_name in enumerate(df_columns_lower):
                if name_lower in col_name or col_name in name_lower:
                    return df.columns[i]
        return None
    
    def _normalize_question_number(self, value: Any) -> Optional[int]:
        """
        Normalize question number to integer
        """
        if pd.isna(value):
            return None
        
        # Try direct conversion
        try:
            return int(float(value))
        except:
            pass
        
        # Extract number from string
        import re
        text = str(value)
        numbers = re.findall(r'\d+', text)
        if numbers:
            return int(numbers[0])
        
        return None
    
    def _convert_answer_to_index(self, value: Any) -> Optional[int]:
        """
        Convert answer value to choice index (0-based)
        """
        if pd.isna(value):
            return None
        
        # Direct mapping
        if value in self.choice_mappings:
            return self.choice_mappings[value]
        
        # Try string conversion
        str_value = str(value).strip()
        if str_value in self.choice_mappings:
            return self.choice_mappings[str_value]
        
        # Try uppercase
        upper_value = str_value.upper()
        if upper_value in self.choice_mappings:
            return self.choice_mappings[upper_value]
        
        return None
    
    def _is_positive_indicator(self, value: Any) -> bool:
        """
        Check if value indicates a positive/marked choice
        """
        if pd.isna(value):
            return False
        
        return str(value).strip() in self.positive_indicators
    
    def _is_sequential(self, values: pd.Series) -> bool:
        """
        Check if values form a sequential pattern
        """
        sorted_values = sorted(values)
        for i in range(1, len(sorted_values)):
            if sorted_values[i] - sorted_values[i-1] > 2:  # Allow some gaps
                return False
        return True
    
    def create_sample_excel(self, output_path: str, format_type: str = 'standard') -> str:
        """
        Create a sample Excel file with the specified format
        
        Args:
            output_path: Path to save the sample file
            format_type: Type of format ('standard', 'matrix', 'extended')
            
        Returns:
            Path to created file
        """
        try:
            if format_type == 'standard':
                data = {
                    'Question': list(range(1, 21)),
                    'Correct Answer': ['A', 'B', 'C', 'D', 'A', 'B', 'C', 'D', 'A', 'B',
                                     'C', 'D', 'A', 'B', 'C', 'D', 'A', 'B', 'C', 'D']
                }
            elif format_type == 'matrix':
                data = {
                    'Question': list(range(1, 11)),
                    'A': ['X', '', '', '', 'X', '', '', '', 'X', ''],
                    'B': ['', 'X', '', '', '', 'X', '', '', '', 'X'],
                    'C': ['', '', 'X', '', '', '', 'X', '', '', ''],
                    'D': ['', '', '', 'X', '', '', '', 'X', '', '']
                }
            elif format_type == 'extended':
                data = {
                    'Question': list(range(1, 11)),
                    'Correct Answer': ['A', 'B', 'C', 'D', 'A', 'B', 'C', 'D', 'A', 'B'],
                    'Difficulty': ['Easy', 'Medium', 'Hard', 'Medium', 'Easy', 
                                 'Hard', 'Medium', 'Easy', 'Hard', 'Medium'],
                    'Category': ['Math', 'Science', 'English', 'History', 'Math',
                               'Science', 'English', 'History', 'Math', 'Science'],
                    'Explanation': [f'Explanation for question {i}' for i in range(1, 11)]
                }
            
            df = pd.DataFrame(data)
            df.to_excel(output_path, index=False, engine='openpyxl')
            
            self.logger.info(f"Sample Excel file created: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error creating sample Excel file: {str(e)}")
            raise