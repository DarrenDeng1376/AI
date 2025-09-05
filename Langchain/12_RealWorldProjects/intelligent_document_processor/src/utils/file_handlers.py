"""
File handling utilities for document upload and processing
"""
import os
import hashlib
import mimetypes
import logging
from typing import Dict, List, Optional, Tuple, BinaryIO
from pathlib import Path
from dataclasses import dataclass
import tempfile
import shutil

from config import app_config

logger = logging.getLogger(__name__)

@dataclass
class FileInfo:
    """Information about an uploaded file"""
    filename: str
    size: int
    mime_type: str
    file_hash: str
    extension: str
    is_valid: bool
    error_message: Optional[str] = None

@dataclass
class UploadResult:
    """Result of file upload operation"""
    success: bool
    file_info: Optional[FileInfo]
    file_path: Optional[str]
    error_message: Optional[str]
    temp_file: Optional[str] = None

class FileHandler:
    """Handles file upload, validation, and temporary storage"""
    
    def __init__(self, upload_dir: Optional[str] = None):
        """Initialize file handler"""
        self.upload_dir = upload_dir or tempfile.gettempdir()
        self.temp_files = set()  # Track temporary files for cleanup
        
        # Ensure upload directory exists
        Path(self.upload_dir).mkdir(parents=True, exist_ok=True)
    
    def validate_file(self, filename: str, file_size: int) -> FileInfo:
        """
        Validate uploaded file
        
        Args:
            filename: Original filename
            file_size: File size in bytes
            
        Returns:
            FileInfo with validation results
        """
        try:
            # Extract file extension
            file_path = Path(filename)
            extension = file_path.suffix.lower().replace('.', '')
            
            # Determine MIME type
            mime_type, _ = mimetypes.guess_type(filename)
            if not mime_type:
                mime_type = "application/octet-stream"
            
            # Check file extension
            if extension not in app_config.allowed_file_types:
                return FileInfo(
                    filename=filename,
                    size=file_size,
                    mime_type=mime_type,
                    file_hash="",
                    extension=extension,
                    is_valid=False,
                    error_message=f"File type '{extension}' not supported. Allowed types: {', '.join(app_config.allowed_file_types)}"
                )
            
            # Check file size
            max_size_bytes = app_config.max_file_size_mb * 1024 * 1024
            if file_size > max_size_bytes:
                return FileInfo(
                    filename=filename,
                    size=file_size,
                    mime_type=mime_type,
                    file_hash="",
                    extension=extension,
                    is_valid=False,
                    error_message=f"File too large: {file_size / (1024*1024):.1f}MB (max: {app_config.max_file_size_mb}MB)"
                )
            
            # Check filename for security issues
            if self._has_security_issues(filename):
                return FileInfo(
                    filename=filename,
                    size=file_size,
                    mime_type=mime_type,
                    file_hash="",
                    extension=extension,
                    is_valid=False,
                    error_message="Filename contains invalid characters or patterns"
                )
            
            return FileInfo(
                filename=filename,
                size=file_size,
                mime_type=mime_type,
                file_hash="",  # Will be calculated when file is processed
                extension=extension,
                is_valid=True
            )
            
        except Exception as e:
            logger.error(f"Error validating file {filename}: {e}")
            return FileInfo(
                filename=filename,
                size=file_size,
                mime_type="unknown",
                file_hash="",
                extension="",
                is_valid=False,
                error_message=f"Validation error: {str(e)}"
            )
    
    def save_uploaded_file(self, file_content: bytes, filename: str) -> UploadResult:
        """
        Save uploaded file content to temporary storage
        
        Args:
            file_content: Raw file bytes
            filename: Original filename
            
        Returns:
            UploadResult with file path and metadata
        """
        try:
            # Validate file first
            file_info = self.validate_file(filename, len(file_content))
            
            if not file_info.is_valid:
                return UploadResult(
                    success=False,
                    file_info=file_info,
                    file_path=None,
                    error_message=file_info.error_message
                )
            
            # Calculate file hash
            file_hash = self._calculate_hash(file_content)
            file_info.file_hash = file_hash
            
            # Generate safe filename
            safe_filename = self._generate_safe_filename(filename, file_hash)
            file_path = os.path.join(self.upload_dir, safe_filename)
            
            # Save file
            with open(file_path, 'wb') as f:
                f.write(file_content)
            
            # Track temp file for cleanup
            self.temp_files.add(file_path)
            
            logger.info(f"Saved file: {filename} -> {file_path}")
            
            return UploadResult(
                success=True,
                file_info=file_info,
                file_path=file_path,
                error_message=None,
                temp_file=file_path
            )
            
        except Exception as e:
            error_msg = f"Failed to save file {filename}: {str(e)}"
            logger.error(error_msg)
            return UploadResult(
                success=False,
                file_info=file_info if 'file_info' in locals() else None,
                file_path=None,
                error_message=error_msg
            )
    
    def read_file_content(self, file_path: str) -> Tuple[bool, Optional[bytes], Optional[str]]:
        """
        Read file content from disk
        
        Returns:
            (success, content, error_message)
        """
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            return True, content, None
            
        except FileNotFoundError:
            error_msg = f"File not found: {file_path}"
            logger.error(error_msg)
            return False, None, error_msg
        except PermissionError:
            error_msg = f"Permission denied reading file: {file_path}"
            logger.error(error_msg)
            return False, None, error_msg
        except Exception as e:
            error_msg = f"Error reading file {file_path}: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg
    
    def create_file_from_upload(self, uploaded_file) -> UploadResult:
        """
        Handle file upload from Streamlit file uploader
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            
        Returns:
            UploadResult with processing information
        """
        try:
            if uploaded_file is None:
                return UploadResult(
                    success=False,
                    file_info=None,
                    file_path=None,
                    error_message="No file uploaded"
                )
            
            # Read file content
            file_content = uploaded_file.read()
            
            # Reset file pointer if needed
            if hasattr(uploaded_file, 'seek'):
                uploaded_file.seek(0)
            
            # Save file
            return self.save_uploaded_file(file_content, uploaded_file.name)
            
        except Exception as e:
            error_msg = f"Error processing uploaded file: {str(e)}"
            logger.error(error_msg)
            return UploadResult(
                success=False,
                file_info=None,
                file_path=None,
                error_message=error_msg
            )
    
    def process_multiple_files(self, uploaded_files: List) -> List[UploadResult]:
        """
        Process multiple uploaded files
        
        Args:
            uploaded_files: List of uploaded file objects
            
        Returns:
            List of UploadResult objects
        """
        results = []
        
        # Check upload limit
        if len(uploaded_files) > app_config.max_documents_per_session:
            logger.warning(f"Too many files uploaded: {len(uploaded_files)} (max: {app_config.max_documents_per_session})")
            # Process only the allowed number
            uploaded_files = uploaded_files[:app_config.max_documents_per_session]
        
        for uploaded_file in uploaded_files:
            result = self.create_file_from_upload(uploaded_file)
            results.append(result)
        
        return results
    
    def cleanup_temp_file(self, file_path: str) -> bool:
        """Remove a temporary file"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                self.temp_files.discard(file_path)
                logger.debug(f"Cleaned up temp file: {file_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error cleaning up temp file {file_path}: {e}")
            return False
    
    def cleanup_all_temp_files(self) -> int:
        """Remove all temporary files created by this handler"""
        cleaned_count = 0
        
        for file_path in list(self.temp_files):
            if self.cleanup_temp_file(file_path):
                cleaned_count += 1
        
        self.temp_files.clear()
        return cleaned_count
    
    def get_file_stats(self, file_path: str) -> Optional[Dict[str, any]]:
        """Get file statistics"""
        try:
            if not os.path.exists(file_path):
                return None
            
            stat = os.stat(file_path)
            
            return {
                "size": stat.st_size,
                "created": stat.st_ctime,
                "modified": stat.st_mtime,
                "accessed": stat.st_atime,
                "is_file": os.path.isfile(file_path),
                "readable": os.access(file_path, os.R_OK),
                "writable": os.access(file_path, os.W_OK)
            }
            
        except Exception as e:
            logger.error(f"Error getting file stats for {file_path}: {e}")
            return None
    
    def _calculate_hash(self, content: bytes) -> str:
        """Calculate SHA-256 hash of file content"""
        return hashlib.sha256(content).hexdigest()
    
    def _generate_safe_filename(self, original_filename: str, file_hash: str) -> str:
        """Generate a safe filename for storage"""
        # Get file extension
        file_path = Path(original_filename)
        extension = file_path.suffix.lower()
        
        # Create safe filename using hash and timestamp
        import time
        timestamp = int(time.time())
        safe_name = f"{file_hash[:16]}_{timestamp}{extension}"
        
        return safe_name
    
    def _has_security_issues(self, filename: str) -> bool:
        """Check filename for security issues"""
        # Check for path traversal
        if '..' in filename or '/' in filename or '\\' in filename:
            return True
        
        # Check for null bytes
        if '\x00' in filename:
            return True
        
        # Check for extremely long filenames
        if len(filename) > 255:
            return True
        
        # Check for filenames that start with dot (hidden files)
        if filename.startswith('.'):
            return True
        
        # Check for reserved names (Windows)
        reserved_names = {
            'CON', 'PRN', 'AUX', 'NUL',
            'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
            'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
        }
        
        name_without_ext = Path(filename).stem.upper()
        if name_without_ext in reserved_names:
            return True
        
        return False

class BatchFileProcessor:
    """Process multiple files in batches with progress tracking"""
    
    def __init__(self, file_handler: FileHandler):
        self.file_handler = file_handler
        self.processing_results = []
    
    def process_file_batch(
        self, 
        uploaded_files: List, 
        progress_callback: Optional[callable] = None
    ) -> Dict[str, any]:
        """
        Process a batch of files with progress tracking
        
        Args:
            uploaded_files: List of uploaded file objects
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with batch processing results
        """
        start_time = time.time()
        results = {
            "total_files": len(uploaded_files),
            "successful": 0,
            "failed": 0,
            "results": [],
            "processing_time": 0,
            "errors": []
        }
        
        for i, uploaded_file in enumerate(uploaded_files):
            try:
                # Update progress
                if progress_callback:
                    progress = (i + 1) / len(uploaded_files)
                    progress_callback(progress, f"Processing {uploaded_file.name}...")
                
                # Process file
                result = self.file_handler.create_file_from_upload(uploaded_file)
                results["results"].append(result)
                
                if result.success:
                    results["successful"] += 1
                else:
                    results["failed"] += 1
                    if result.error_message:
                        results["errors"].append({
                            "filename": uploaded_file.name,
                            "error": result.error_message
                        })
                
            except Exception as e:
                results["failed"] += 1
                error_msg = f"Unexpected error processing {uploaded_file.name}: {str(e)}"
                results["errors"].append({
                    "filename": uploaded_file.name,
                    "error": error_msg
                })
                logger.error(error_msg)
        
        results["processing_time"] = time.time() - start_time
        
        if progress_callback:
            progress_callback(1.0, "Batch processing complete")
        
        return results

# Utility functions
def get_file_type_icon(extension: str) -> str:
    """Get emoji icon for file type"""
    icons = {
        'pdf': '📄',
        'doc': '📝',
        'docx': '📝',
        'txt': '📃',
        'xlsx': '📊',
        'xls': '📊',
        'pptx': '📋',
        'ppt': '📋',
        'png': '🖼️',
        'jpg': '🖼️',
        'jpeg': '🖼️',
        'tiff': '🖼️',
        'bmp': '🖼️'
    }
    return icons.get(extension.lower(), '📎')

def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format"""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    
    return f"{s} {size_names[i]}"

# Example usage and testing
if __name__ == "__main__":
    import time
    
    def test_file_handler():
        """Test file handler functionality"""
        handler = FileHandler()
        
        # Test file validation
        test_cases = [
            ("document.pdf", 1024 * 1024),  # Valid PDF
            ("image.jpg", 2 * 1024 * 1024),  # Valid image
            ("large.pdf", 100 * 1024 * 1024),  # Too large
            ("script.exe", 1024),  # Invalid type
            ("../../../etc/passwd", 1024),  # Security issue
        ]
        
        print("Testing file validation:")
        for filename, size in test_cases:
            file_info = handler.validate_file(filename, size)
            status = "✓" if file_info.is_valid else "✗"
            print(f"  {status} {filename}: {file_info.error_message or 'Valid'}")
        
        # Test file size formatting
        print("\nTesting file size formatting:")
        sizes = [0, 1024, 1048576, 1073741824]
        for size in sizes:
            formatted = format_file_size(size)
            print(f"  {size} bytes = {formatted}")
        
        # Cleanup
        cleaned = handler.cleanup_all_temp_files()
        print(f"\nCleaned up {cleaned} temporary files")
    
    # Run test
    # test_file_handler()