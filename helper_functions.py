import os
import asyncio
from typing import List
import shutil

import fitz  
from docx2python import docx2python
from fastapi import HTTPException



from logging_config import get_logger

logger = get_logger(__name__)

def extract_text_from_pdf(file_path: str) -> str:
    try:
        pdf_document = fitz.open(file_path)
        all_text = ""
        for page in pdf_document:
            all_text += page.get_text() + "\n"
        pdf_document.close()
        return all_text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting text from PDF: {str(e)}")


def extract_text_from_word(file_path: str) -> str:
    try:
        doc_content = docx2python(file_path)
        return doc_content.text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting text from Word document: {str(e)}")
    
async def cleanup_temp_dir(temp_dir):
    try:
        await asyncio.sleep(5)  # Optional: wait to ensure files are released
        shutil.rmtree(temp_dir)
        print(f"Deleted temporary directory: {temp_dir}")
    except Exception as e:
        print(f"Failed to delete temporary directory {temp_dir}: {e}")