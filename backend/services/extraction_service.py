import fitz  # PyMuPDF
import re
import pytesseract
from pdf2image import convert_from_path

class PDFExtractionService:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.extractions = []

    def extract_acroform(self):
        """Extract data from AcroForm (interactive form) fields using PyMuPDF.

        Iterates through each page's widgets (form fields). If a widget has
        both a ``field_name`` and a ``field_value``, the pair is collected.
        This is the most reliable extraction source for PDFs that contain
        native interactive forms.

        Returns:
            dict: A mapping of ``{field_name: field_value}``.
                  Returns an empty dict if the PDF has no form fields
                  or if an error occurs.
        """
        acroform_data = {}
        try:
            document = fitz.open(self.pdf_path)
            try:
                for page in document:
                    # widgets() yields each interactive form field on the page
                    for widget in page.widgets():
                        field_name = widget.field_name
                        field_value = widget.field_value
                        # Only collect fields that have both a name and a value
                        if field_name and field_value:
                            # If the same field name appears on multiple pages,
                            # keep the first non-empty occurrence
                            if field_name not in acroform_data:
                                acroform_data[field_name] = field_value.strip()
            finally:
                document.close()
        except Exception as e:
            # Graceful degradation: return empty dict on any error
            # (e.g. encrypted PDF, corrupted file, no form fields)
            import logging
            logging.getLogger(__name__).warning(
                "AcroForm extraction failed for '%s': %s", self.pdf_path, e
            )
        return acroform_data

    def extract_layout_based(self):
        # Add logic to extract text based on layout
        document = fitz.open(self.pdf_path)
        for page in document:
            self.extractions.append(page.get_text())
        return self.extractions

    def extract_ocr(self):
        # Add logic to perform OCR on the PDF
        images = convert_from_path(self.pdf_path)
        for image in images:
            text = pytesseract.image_to_string(image)
            self.extractions.append(text)
        return self.extractions

    def extract_regex(self, pattern):
        # Add logic to extract text using regex patterns
        text = '\n'.join(self.extractions)
        return re.findall(pattern, text)

    def validate_extraction(self, criteria):
        # Add logic for validation of extracted data
        passed = all(criterion in self.extractions for criterion in criteria)
        return passed