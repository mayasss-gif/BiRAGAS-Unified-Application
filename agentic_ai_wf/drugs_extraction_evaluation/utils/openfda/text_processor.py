import re
from typing import Optional


class TextProcessor:
    """
    A utility class for processing and extracting information from FDA drug label text.
    
    This class provides methods for cleaning and extracting specific sections
    from FDA drug labeling documents.
    """
    
    # Class-level constants for better maintainability
    SECTION_HEADERS_PATTERN = (
        r'^\s*(?:\d+\.?\d*\s*)?'
        r'(?:'
        r'INDICATIONS\s*(?:AND|&)\s*USAGE|'
        r'DOSAGE\s*(?:AND|&)\s*ADMINISTRATION|'
        r'CONTRAINDICATIONS|'
        r'WARNING|'
        r'PATIENT\s*COUNSELING\s*INFORMATION'
        r')\b[\.:]?\s*'
    )
    
    MECHANISM_OF_ACTION_PATTERN = r'12\.1\s*Mechanism of Action(.*?)(12\.|\Z)'
    
    def __init__(self):
        """Initialize the TextProcessor with compiled regex patterns."""
        self._section_headers_regex = re.compile(
            self.SECTION_HEADERS_PATTERN, 
            flags=re.IGNORECASE
        )
        self._mechanism_regex = re.compile(
            self.MECHANISM_OF_ACTION_PATTERN,
            flags=re.DOTALL | re.IGNORECASE
        )
    
    def strip_header_labels(self, text: str) -> str:
        """
        Remove leading numeric section headers from FDA drug label text.
        
        Args:
            text: The text to clean
            
        Returns:
            Cleaned text with section headers removed
            
        Example:
            >>> processor = TextProcessor()
            >>> processor.strip_header_labels("1 INDICATIONS AND USAGE: Take as directed")
            "Take as directed"
        """
        if not text:
            return text
        return self._section_headers_regex.sub('', text)
    
    def extract_mechanism_of_action(self, pharmacology_text: str) -> str:
        """
        Extract the mechanism of action section from clinical pharmacology text.
        
        Args:
            pharmacology_text: The clinical pharmacology section text
            
        Returns:
            Extracted mechanism of action text or "Not available" if not found
            
        Example:
            >>> processor = TextProcessor()
            >>> text = "12.1 Mechanism of Action: This drug works by..."
            >>> processor.extract_mechanism_of_action(text)
            "This drug works by..."
        """
        if not pharmacology_text:
            return "Not available"
            
        match = self._mechanism_regex.search(pharmacology_text)
        return match.group(1).strip() if match else "Not available"
    
    def clean_drug_label_text(self, text: str) -> str:
        """
        Comprehensive cleaning of drug label text.
        
        Args:
            text: Raw drug label text
            
        Returns:
            Cleaned text with headers removed and standardized formatting
        """
        if not text:
            return "Not available"
            
        # Strip section headers
        cleaned = self.strip_header_labels(text)
        
        # Remove bullet points and extra whitespace
        cleaned = cleaned.replace('•', '').strip()
        
        # Check for empty or invalid content
        lower_text = cleaned.lower()
        if (not cleaned or 
            lower_text.startswith('none') or 
            lower_text in ['na', 'not applicable', 'not available']):
            return "Not available"
            
        return cleaned
    
    @staticmethod
    def is_valid_text(text: str) -> bool:
        """
        Check if text contains meaningful content.
        
        Args:
            text: Text to validate
            
        Returns:
            True if text contains meaningful content, False otherwise
        """
        if not text:
            return False
            
        cleaned = text.strip().lower()
        invalid_values = ['', 'none', 'na', 'not applicable', 'not available']
        return cleaned not in invalid_values 