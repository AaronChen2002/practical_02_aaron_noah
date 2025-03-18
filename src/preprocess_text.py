import pdfplumber
import os
from pathlib import Path

def clean_text(text):
    """Clean extracted text by removing extra whitespace and normalizing spacing."""
    # Remove extra whitespace and normalize line breaks
    cleaned = text.replace('●', '').replace('■', '').replace('○', '').replace('▲', '').replace('▴', '').replace('▾', '').replace('△', '').replace('▽', '').replace('▷', '').replace('◁', '').replace('□', '').replace('■', '').replace('▲', '').replace('▴', '').replace('▾', '').replace('△', '').replace('▽', '').replace('▷', '').replace('◁', '').replace('□', '')
    cleaned = ' '.join(text.split())
    return cleaned

def pdf_to_text(pdf_path):
    """Extract text from a PDF file."""
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return clean_text(text)
    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")
        return None

def process_pdfs(input_dir, output_dir):
    """Process all PDFs in input directory and save as text files in output directory."""
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Find all PDF files in input directory
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_dir, pdf_file)
        # Create output text file path (same name but .txt extension)
        txt_file = os.path.join(output_dir, pdf_file.rsplit('.', 1)[0] + '.txt')
        
        print(f"Processing: {pdf_file}")
        
        # Extract and clean text
        text = pdf_to_text(pdf_path)
        
        if text:
            # Save cleaned text to file
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"Created: {txt_file}")

if __name__ == "__main__":
    # Define input and output directories
    input_dir = "data/raw_data"  
    output_dir = "data/cleaned_data"  
    
    process_pdfs(input_dir, output_dir)