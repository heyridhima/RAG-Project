import pypdf
import os

def list_pdfs(data_dir: str) -> list:
    pdfs = []
    for file in os.listdir(data_dir):
        if file.endswith(".pdf"):
            pdfs.append(os.path.join(data_dir, file))
    return pdfs

def extract_text(pdf_dir: str) -> list[dict]:
    pdf_info = []
    reader = pypdf.PdfReader(pdf_dir)
    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text(extraction_mode="plain")
        if page_text.strip():
            pdf_info.append({
                "text": page_text,
                "source": os.path.basename(pdf_dir),
                "page":  page_num + 1
            })
    return pdf_info

def load_documents(data_dir: str) -> list[dict]:
    doc_list = []
    pdfs = list_pdfs(data_dir)
    for pdf in pdfs:
        doc_list.extend(extract_text(pdf))
    return doc_list

if __name__ == "__main__":
    processed_files = load_documents("data")
    print(len(processed_files))
    print(processed_files[50])
    print(len(processed_files[0]["text"]))
    print(processed_files[0]["source"])
    print(processed_files[0]["page"])