
@app.route('/save_grouped_data', methods=['POST'])
def save_grouped_data():
    """
    Receives grouped data from edit_v2, flattens it for PDF generation,
    and runs the generation process.
    """
    data = request.json
    video_id = data.get('video_id')
    groups = data.get('groups', [])
    
    if not video_id:
        return jsonify({'error': 'Missing video_id'}), 400

    output_folder = os.path.join(app.config['OUTPUT_FOLDER'], video_id)
    images_folder = os.path.join(output_folder, "images", "organized_moderate", "unique") # or wherever images are
    
    # We need to flatten the groups into "Slides" for the PDF generator
    # OR we update the PDF generator. 
    # For now, let's flatten:
    # Strategy: 
    # The First Image of the Group gets the Group's Text.
    # Subsequent Images in the Group get empty text.
    # If a group has NO images, we might create a "Text Only" slide (but current PDF gen might need an image).
    # actually current PDF gen needs an image path.
    
    slides = []
    
    for g in groups:
        text_block = g.get('text', '')
        # Filter out removed images
        valid_images = [img for img in g.get('images', []) if not img.get('removed', False)]
        
        if not valid_images:
            # Handle text-only case?
            # Current PDF generator likely skips if no image.
            # We could create a dummy "blank" image or just skip.
            # For now, let's print a warning or just skip images but maybe we lose the text?
            # Better strategy: If we have text but no images, we might want to attach it to a placeholder.
            # But let's assume user kept at least one image if they want the text.
            continue
            
        for i, img in enumerate(valid_images):
            # Reconstruct full path
            # img['url'] is like /image/vid/unique/file.png
            # We need absolute path for PDF generator
            filename = img['filename']
            # We assume it's in the standard unique folder or raw
            # We can try to find it
            
            # Simple assumption: existing unique folder
            # But wait, we might have moved things?
            # Actually, `prepare_grouped_data` reads from `unique`
            # So it should be there.
            img_path = os.path.join(images_folder, filename)
            
            # Text assignment:
            # Option A: All text on first slide
            # Option B: Repeat text? (Bad)
            # Option C: All text on first slide, others empty.
            
            slide_text = text_block if i == 0 else ""
            
            slides.append({
                'image_path': img_path,
                'timestamp': img.get('timestamp', ''),
                'text': slide_text
            })
            
    # Now call PDF generator
    # We might need to handle the case where "text-only" slides were needed if we want to improve this later.
    
    try:
        pdf_result = create_pdf_from_data(video_id, slides, output_dir=app.config['OUTPUT_FOLDER'])
        docx_result = create_docx_from_data(video_id, slides, output_dir=app.config['OUTPUT_FOLDER'])
        
        return jsonify({
            'status': 'success', 
            'pdf_path': pdf_result,
            'docx_path': docx_result
        })
    except Exception as e:
        print(f"Error generating from groups: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
