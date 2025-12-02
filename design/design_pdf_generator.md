# Component Design: Output Generation (`pdf_generator.py`)

## 1. Overview
`pdf_generator.py` is the final stage of the pipeline. It takes the curated set of images and the processed transcript and assembles them into user-consumable documents (PDF and DOCX).

## 2. Responsibilities
- **PDF Creation**: Generates a slide-deck style PDF.
- **DOCX Creation**: Generates a Word document.
- **Synchronization**: Matches transcript segments to specific slides based on timestamps.
- **Image Optimization**: Resizes and compresses images before embedding to keep file sizes manageable.

## 3. Internal Logic Flow (Synchronization)

```mermaid
graph TD
    Start([Start Sync]) --> LoadTrans[Load & Parse Transcript]
    LoadTrans --> LoadImages[Load Images & Timestamps]
    
    LoadImages --> Loop{For Each Image}
    
    Loop --> TimeCalc[Get Image Timestamp]
    TimeCalc --> FindText[Find Transcript Lines within ±30s]
    
    FindText --> Clean[Clean & Merge Text]
    Clean --> Optimize[Optimize Image (Resize/JPEG)]
    
    Optimize --> AddPage[Add Page to Document]
    AddPage --> AddImg[Embed Image]
    AddImg --> AddText[Embed Text]
    
    AddText --> Next{More Images?}
    Next -->|Yes| Loop
    Next -->|No| Save[Save File]
```

## 4. Key Functions

### `convert_screenshots_to_pdf(...)`
Simple wrapper around `img2pdf` to create a PDF containing *only* the images, with no text. Fast and lossless.

### `sync_images_with_transcript(...)`
Creates a "Combined PDF" using `reportlab`.
- **Logic**:
    1.  Iterates through sorted images.
    2.  For each image at time `T`, finds all transcript lines where `timestamp` is close to `T` (simple window approach).
    3.  Uses `reportlab` to draw the image and the text on a new page.
    4.  **Optimization**: Converts PNGs to JPEGs in-memory to reduce PDF size significantly.

### `sync_images_with_transcript_docx(...)`
Creates a "Combined DOCX" using `python-docx`.
- **Logic**: Similar to the PDF sync function but uses the `docx` library API.
- **Encoding Handling**: Implements robust file reading (tries utf-8, utf-16, cp1252) to handle various transcript file encodings.

## 5. Dependencies
- **External**:
    - `reportlab`: Complex PDF generation.
    - `img2pdf`: Simple image-to-PDF conversion.
    - `python-docx`: Word document generation.
    - `Pillow` (PIL): Image manipulation.
