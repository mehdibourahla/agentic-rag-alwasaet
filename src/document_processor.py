from typing import List, Optional
from pathlib import Path
import pdfplumber
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from langdetect import detect
import pytesseract
from pdf2image import convert_from_path


class DocumentProcessor:
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def _is_text_readable(self, text: str) -> bool:

        if not text or len(text.strip()) < 10:
            return False
        
        # Check for Arabic text quality
        if self._contains_arabic(text):
            presentation_forms = sum(1 for char in text if '\uFB50' <= char <= '\uFEFF')
            total_arabic = sum(1 for char in text if self._is_arabic_char(char))
            
            if total_arabic > 0:
                # If more than 30% are presentation forms, text is likely garbled
                if presentation_forms / total_arabic > 0.3:
                    return False
            
            words = text.split()
            if len(words) < 3:
                return False
            
            # Check if average word length is reasonable
            avg_word_length = sum(len(word) for word in words) / len(words)
            if avg_word_length > 20:
                return False
        
        return True
    
    def _is_arabic_char(self, char: str) -> bool:
        return ('\u0600' <= char <= '\u06FF' or 
                '\u0750' <= char <= '\u077F' or
                '\uFB50' <= char <= '\uFEFF')
    
    def _contains_arabic(self, text: str) -> bool:
        return any(self._is_arabic_char(char) for char in text)
    
    def _detect_language(self, text: str) -> str:
        try:
            clean_text = text.strip()
            if not clean_text:
                return 'unknown'
            
            # Check for Arabic and Latin characters
            has_arabic = self._contains_arabic(clean_text)
            has_latin = any('a' <= char.lower() <= 'z' for char in clean_text)
            
            if has_arabic and has_latin:
                # Mixed content determine dominant language
                arabic_chars = sum(1 for char in clean_text if self._is_arabic_char(char))
                latin_chars = sum(1 for char in clean_text if 'a' <= char.lower() <= 'z')
                
                total_chars = arabic_chars + latin_chars
                if total_chars > 0:
                    arabic_ratio = arabic_chars / total_chars
                    if 0.2 < arabic_ratio < 0.8:
                        return 'mixed'
                    elif arabic_ratio >= 0.8:
                        return 'ar'
                    else:
                        return 'en'
            elif has_arabic:
                return 'ar'
            elif has_latin:
                return 'en'
            
            # Use langdetect as fallback
            detected = detect(clean_text)
            return detected
            
        except:
            return 'unknown'
    
    async def _extract_text_with_ocr(self, pdf_path: str, page_num: int) -> Optional[str]:
        try:
            # Convert specific page to image
            images = convert_from_path(
                pdf_path,
                first_page=page_num + 1,
                last_page=page_num + 1,
                dpi=300
            )
            
            if not images:
                return None
            
            page_text = pytesseract.image_to_string(
                images[0],
                lang='ara+eng'
            )
            
            return page_text
            
        except Exception as e:
            print(f"OCR failed for page {page_num}: {e}")
            return None
    
    async def process_pdf(self, file_path: str, filename: str) -> List[Document]:
        documents = []
        
        try:
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                uses_ocr = False
                
                # First, try to detect if we need OCR by sampling a few pages
                sample_pages = min(3, total_pages)
                needs_ocr = False
                
                for i in range(sample_pages):
                    page = pdf.pages[i]
                    sample_text = page.extract_text()
                    
                    if sample_text and self._contains_arabic(sample_text):
                        if not self._is_text_readable(sample_text):
                            needs_ocr = True
                            print(f"Detected garbled Arabic text in {filename}, using OCR...")
                            break
                
                # Process each page
                for page_num, page in enumerate(pdf.pages):
                    text = None
                    extraction_method = "standard"
                    
                    if not needs_ocr:
                        # Try standard extraction first
                        text = page.extract_text()
                        
                        # Check if extracted text is readable
                        if text and not self._is_text_readable(text):
                            print(f"Page {page_num + 1}: Standard extraction produced unreadable text, trying OCR...")
                            text = None
                    
                    # Use OCR if needed
                    if needs_ocr or text is None or len(text.strip()) < 10:
                        ocr_text = await self._extract_text_with_ocr(file_path, page_num)
                        if ocr_text and len(ocr_text.strip()) > 10:
                            text = ocr_text
                            extraction_method = "ocr"
                            uses_ocr = True
                    
                    # Skip empty pages
                    if not text or not text.strip():
                        continue
                    
                    # Clean up text
                    lines = text.split('\n')
                    cleaned_lines = [line.strip() for line in lines if line.strip()]
                    text = '\n'.join(cleaned_lines)
                    
                    # Detect page language
                    page_language = self._detect_language(text)
                    
                    # Create document with metadata
                    doc = Document(
                        text=text,
                        metadata={
                            "filename": filename,
                            "page": page_num + 1,
                            "source": file_path,
                            "page_language": page_language,
                            "extraction_method": extraction_method
                        }
                    )
                    documents.append(doc)
                
                if uses_ocr:
                    print(f"Successfully processed {filename} using OCR for Arabic text")
            
            # Split documents into chunks while preserving metadata
            chunked_documents = []
            for doc in documents:
                # Split the text
                chunks = self.text_splitter.split_text(doc.text)
                
                # Create new documents for each chunk
                for i, chunk in enumerate(chunks):
                    # Detect language for each chunk
                    chunk_language = self._detect_language(chunk)
                    
                    chunk_doc = Document(
                        text=chunk,
                        metadata={
                            **doc.metadata,
                            "chunk_index": i,
                            "language": chunk_language,
                            "is_mixed": chunk_language == 'mixed'
                        }
                    )
                    chunked_documents.append(chunk_doc)
            
            return chunked_documents
            
        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")
    
    def validate_file(self, file_path: str, max_size_mb: int = 10) -> bool:
        file_path = Path(file_path)
        
        # Check if file exists
        if not file_path.exists():
            raise ValueError("File does not exist")
        
        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > max_size_mb:
            raise ValueError(f"File size ({file_size_mb:.1f}MB) exceeds limit ({max_size_mb}MB)")
        
        # Check if it's a PDF
        with open(file_path, 'rb') as f:
            header = f.read(4)
            if header != b'%PDF':
                raise ValueError("File is not a valid PDF")
        
        return True