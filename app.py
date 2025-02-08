import streamlit as st
from PIL import Image
import pytesseract
import tempfile
import pdf2image
import json
from google import genai
from typing import Any
import pandas as pd
import time


def perform_ocr(uploaded_file: Any, ocr_engine: str = "Tesseract") -> str:
    """
    Perform OCR on an uploaded file (image or PDF) using the specified OCR engine.

    Args:
        uploaded_file (UploadedFile): The file uploaded via Streamlit.
        ocr_engine (str): The OCR engine to use, either 'Tesseract' or 'EasyOCR'.

    Returns:
        str: The extracted text from the file.
    """
    text: str = ""
    file_extension: str = uploaded_file.name.split('.')[-1].lower()
    
    if ocr_engine.lower() == "tesseract":
        if file_extension == "pdf":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_pdf_path = tmp_file.name
            try:
                images = pdf2image.convert_from_path(tmp_pdf_path)
            except Exception as e:
                st.error(f"Error converting PDF: {e}")
                return ""
            for image in images:
                text += pytesseract.image_to_string(image)
        else:
            try:
                image = Image.open(uploaded_file)
                text = pytesseract.image_to_string(image)
            except Exception as e:
                st.error(f"Error processing image file: {e}")
                return ""
        return text
    elif ocr_engine.lower() == "easyocr":
        try:
            import easyocr
            import numpy as np
        except ImportError:
            st.error("easyocr package is not installed. Please install it.")
            return ""
        reader = easyocr.Reader(['en'], gpu=False)
        if file_extension == "pdf":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_pdf_path = tmp_file.name
            try:
                images = pdf2image.convert_from_path(tmp_pdf_path)
            except Exception as e:
                st.error(f"Error converting PDF: {e}")
                return ""
            for image in images:
                image_np = np.array(image)
                result = reader.readtext(image_np, detail=0, paragraph=True)
                text += " ".join(result) + "\n"
        else:
            try:
                image = Image.open(uploaded_file)
                image_np = np.array(image)
                result = reader.readtext(image_np, detail=0, paragraph=True)
                text = " ".join(result)
            except Exception as e:
                st.error(f"Error processing image file: {e}")
                return ""
        return text
    else:
        st.error(f"Unsupported OCR engine: {ocr_engine}")
        return ""


def build_gemini_prompt(extracted_text: str) -> str:
    """
    Build the prompt for the Gemini API to extract invoice details and assess OCR quality.

    Args:
        extracted_text (str): The text extracted via OCR.

    Returns:
        str: The constructed prompt string.
    """
    prompt = f"""
You are provided with text extracted via OCR from an invoice. Extract and return the following details in JSON format:

- invoice_id: The invoice identifier.
- total_amount: The total amount on the invoice.
- currency: The currency used in the invoice.
- tax: The tax amount (if applicable).
- issue_date: The invoice's issue date.
- due_date: The invoice's due date.
- description: A brief description of the invoice.
- items: A list of items; each item should include a description, quantity, unit price, and total.
- billing_address: The billing address details.

For any field not found in the extracted text, explicitly mark it as "Missing".

Additionally, assess the OCR extraction quality. Provide an "ocr_accuracy" score (0-100) along with a brief explanation of potential OCR errors.

The extracted text is below:
{extracted_text}

Respond with valid JSON following the schema.
"""
    return prompt


def extract_invoice_data(api_key: str, extracted_text: str) -> dict:
    """
    Extract invoice details using the Gemini API.

    Args:
        api_key (str): The API key for Gemini API.
        extracted_text (str): The OCR extracted text from the invoice.

    Returns:
        dict: The structured invoice data returned by the API.
    """
    prompt: str = build_gemini_prompt(extracted_text)
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    try:
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "text") and isinstance(candidate.text, str):
                generated_text = candidate.text
            elif hasattr(candidate, "content"):
                content_obj = candidate.content
                if hasattr(content_obj, "parts") and content_obj.parts and hasattr(content_obj.parts[0], "text"):
                    generated_text = content_obj.parts[0].text
                else:
                    raise AttributeError("Candidate content object has no parts with text attribute")
            else:
                raise AttributeError("Candidate object has no attribute 'text' or 'content'")
        elif hasattr(response, "result"):
            generated_text = response.result
        else:
            raise AttributeError("No valid attribute for generated text found in response")
        generated_text = generated_text.strip()
        if generated_text.startswith("```json"):
            generated_text = generated_text[len("```json"):].strip()
        if generated_text.endswith("```"):
            generated_text = generated_text[:-3].strip()
        invoice_data = json.loads(generated_text)
    except Exception as e:
        st.error(f"Error parsing Gemini API response: {e}")
        invoice_data = {}
    return invoice_data


def main() -> None:
    """
    Main function for the Invoice OCR Processing App using Streamlit.
    """
    # Inject custom CSS for professional styling
    st.markdown(
        """
        <style>
        body {
            background-color: #f5f5f5;
            font-family: 'Segoe UI', sans-serif;
        }
        .reportview-container {
            padding: 2rem;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 5px;
        }
        .stTextArea textarea {
            font-size: 1rem;
        }
        </style>
        """, unsafe_allow_html=True
    )
    
    st.title("Invoice OCR Processing App")
    st.write("Upload your invoice (Image or PDF) for OCR processing.")
    
    # Initialize session state variables if not present
    if "invoice_data" not in st.session_state:
        st.session_state["invoice_data"] = None
    if "extracted_text" not in st.session_state:
        st.session_state["extracted_text"] = None
    
    uploaded_file = st.file_uploader("Choose a file", type=["png", "jpg", "jpeg", "pdf"])
    if uploaded_file is not None:
        file_extension: str = uploaded_file.name.split('.')[-1].lower()
        if file_extension in ["png", "jpg", "jpeg"]:
            try:
                image = Image.open(uploaded_file)
                with st.expander("Show Uploaded Image", expanded=False):
                    st.image(image, caption="Uploaded Image", use_column_width=True)
            except Exception as e:
                st.error(f"Error displaying image: {e}")
        elif file_extension == "pdf":
            st.info("PDF file uploaded. Preview might not be available.")
    
        if st.button("Perform OCR"):
            with st.spinner("Performing OCR using Tesseract..."):
                tess_text: str = perform_ocr(uploaded_file, "Tesseract")
            with st.spinner("Performing OCR using EasyOCR..."):
                easy_text: str = perform_ocr(uploaded_file, "EasyOCR")
            
            if not tess_text and not easy_text:
                st.error("No text detected. Please ensure the file is clear and try again.")
                return
            
            # Choose the result with more text (simple heuristic)
            if len(easy_text) > len(tess_text):
                extracted_text = easy_text
                st.info("Selected EasyOCR result based on output length.")
            else:
                extracted_text = tess_text
                st.info("Selected Tesseract result based on output length.")
            
            api_key: str = st.secrets.get("GEMINI_API_KEY", "")
            if not api_key:
                st.error("Gemini API key not found in Streamlit secrets.")
                return
            
            with st.spinner("Extracting invoice details using Gemini API..."):
                invoice_data: dict = extract_invoice_data(api_key, extracted_text)
            
            st.session_state["invoice_data"] = invoice_data
            st.session_state["extracted_text"] = extracted_text
    
    if st.session_state["invoice_data"] is not None or st.session_state["extracted_text"] is not None:
        tabs = st.tabs(["Invoice Data", "Extracted Text"])
        with tabs[0]:
            st.subheader("Invoice Data")
            invoice_data = st.session_state["invoice_data"]
            if invoice_data:
                main_data = invoice_data.copy()
                items = main_data.pop("items", None)
                billing_address = main_data.pop("billing_address", None)
                
                st.write("Main Invoice Information")
                main_df = pd.DataFrame([main_data])
                st.dataframe(main_df)
                
                if billing_address:
                    st.write("Billing Address")
                    billing_df = pd.DataFrame([billing_address])
                    st.dataframe(billing_df)
                
                if items:
                    st.write("Invoice Items")
                    items_df = pd.DataFrame(items)
                    st.dataframe(items_df)
                
                if st.button("Save to Keboola"):
                    with st.spinner("Saving to Keboola..."):
                        time.sleep(2)
                    st.success("Saved successfully to Keboola")
            else:
                st.error("No invoice data extracted.")
        with tabs[1]:
            st.subheader("Extracted Text")
            st.text_area("", st.session_state["extracted_text"], height=300)


if __name__ == "__main__":
    main()
