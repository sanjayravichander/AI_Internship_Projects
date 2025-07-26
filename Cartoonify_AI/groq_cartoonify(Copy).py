import streamlit as st
import base64
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter
import os
from dotenv import load_dotenv
import json
import re
import numpy as np
import cv2
import time
import tempfile
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
import threading
import requests
import zipfile
from pathlib import Path

# Load environment variables
load_dotenv()

# Try to import ONNX Runtime for AnimeGANv3
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
    print("✅ ONNX Runtime available for AnimeGANv3")
except ImportError:
    ONNX_AVAILABLE = False
    print("❌ ONNX Runtime not available - using OpenCV filters only")

# Check PIL version compatibility
try:
    LANCZOS_FILTER = Image.Resampling.LANCZOS
except AttributeError:
    LANCZOS_FILTER = Image.LANCZOS

# Try to import Groq and LangChain
try:
    from langchain_groq import ChatGroq
    from langchain.schema import HumanMessage
    GROQ_AVAILABLE = True
    
    # Initialize Groq client
    groq_client = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="meta-llama/llama-4-scout-17b-16e-instruct",  # Vision-capable model
        temperature=0.7
    )
    
except ImportError as e:
    GROQ_AVAILABLE = False
    st.error("❌ Groq/LangChain libraries not installed. Please install with:")
    st.code("pip install langchain-groq langchain")
    st.stop()
except Exception as e:
    GROQ_AVAILABLE = False
    st.error(f"❌ Groq setup failed: {e}")
    st.error("Please check your GROQ_API_KEY in the .env file")
    st.stop()

# === AnimeGANv3 ONNX Model Management ===
@st.cache_resource
def load_animegan_model(show_messages=True):
    """Load AnimeGANv3 ONNX model with caching"""
    if not ONNX_AVAILABLE:
        return None
    
    try:
        model_path = Path("AnimeGANv3_Hayao_STYLE_36.onnx")
        
        # Check if model exists
        if not model_path.exists():
            if show_messages:
                st.error(f"❌ AnimeGANv3 model not found at: {model_path}")
                st.info("💡 Please ensure 'AnimeGANv3_Hayao_STYLE_36.onnx' is in the project directory")
            return None
        
        # Load the ONNX model
        if show_messages:
            with st.spinner("Loading AnimeGANv3 ONNX model..."):
                session = ort.InferenceSession(str(model_path))
                input_name = session.get_inputs()[0].name
                output_name = session.get_outputs()[0].name
                st.success("✅ AnimeGANv3 ONNX model loaded successfully!")
                return session, input_name, output_name
        else:
            # Silent loading for video processing
            session = ort.InferenceSession(str(model_path))
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            return session, input_name, output_name
            
    except Exception as e:
        if show_messages:
            st.error(f"❌ Failed to load AnimeGANv3 ONNX model: {e}")
        return None

def apply_animegan_filter(image, model_data=None, silent_mode=False):
    """Apply AnimeGANv3 ONNX filter to image"""
    if not ONNX_AVAILABLE or model_data is None:
        if not silent_mode:
            st.warning("⚠️ AnimeGANv3 not available, using OpenCV filter")
        return apply_cartoon_filter(image, "comic", 1.0, 1.0, 1.3, 1.2, 1.1, silent_mode)
    
    try:
        session, input_name, output_name = model_data
        
        # Prepare image
        img_array = np.array(image)
        if img_array.shape[-1] == 4:
            img_array = img_array[:, :, :3]
        
        original_height, original_width = img_array.shape[:2]
        
        # Resize to 512x512 for AnimeGANv3
        img_resized = cv2.resize(img_array, (512, 512))
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        # Run inference
        output = session.run([output_name], {input_name: img_batch})[0]
        output_img = np.clip(output[0] * 255.0, 0, 255).astype(np.uint8)
        
        # Resize back to original size
        output_img = cv2.resize(output_img, (original_width, original_height))
        
        return Image.fromarray(output_img)
        
    except Exception as e:
        if not silent_mode:
            st.error(f"❌ AnimeGANv3 processing failed: {e}")
        return apply_cartoon_filter(image, "comic", 1.0, 1.0, 1.3, 1.2, 1.1, silent_mode)

# === Utility Functions ===
def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    if max(image.size) > 1024:
        image.thumbnail((1024, 1024), LANCZOS_FILTER)
    
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    return img_base64

def test_groq_api():
    """Test if Groq API is working"""
    try:
        response = groq_client.invoke([HumanMessage(content="Hello")])
        return True, "✅ Groq API is working"
    except Exception as e:
        return False, f"❌ Groq API Error: {str(e)}"

def analyze_image_with_groq(image, style_prompt="cartoon style"):
    """Use Groq's vision model to analyze image and create cartoon description"""
    try:
        img_b64 = image_to_base64(image)
        
        # Create message with image
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": f"""Analyze this image and describe it in detail for creating a {style_prompt} version.

Focus on:
- Main subject(s) and their characteristics
- Pose, expression, and mood
- Background elements and setting
- Colors and lighting
- Composition

Provide a detailed description that could be used to recreate this image in {style_prompt}. Be specific about visual elements, colors, and artistic style."""
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_b64}"
                    }
                }
            ]
        )
        
        response = groq_client.invoke([message])
        return response.content
        
    except Exception as e:
        st.error(f"❌ Groq vision analysis failed: {e}")
        return None


def apply_cartoon_filter(image, style="comic", intensity=1.0, edge_strength=1.0, saturation=1.3, contrast=1.2, brightness=1.1, silent_mode=False):
    """Apply cartoon-style filters to the image using OpenCV or AnimeGANv3"""
    try:
        cartoon_image = None

        # Handle AnimeGANv3 style
        if style == "animegan_hayao":
            if ONNX_AVAILABLE:
                model_data = load_animegan_model(show_messages=not silent_mode)
                if model_data:
                    cartoon_image = apply_animegan_filter(image, model_data, silent_mode)
                else:
                    if not silent_mode:
                        st.warning("⚠️ AnimeGANv3 model failed to load, using OpenCV comic filter")
                    style = "comic"
            else:
                if not silent_mode:
                    st.warning("⚠️ ONNX Runtime not available, using OpenCV comic filter")
                style = "comic"

        # If not AnimeGAN or fallback to OpenCV
        if cartoon_image is None:
            # Convert PIL to OpenCV format for OpenCV-based filters
            img_array = np.array(image)
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            # Apply bilateral filter with intensity scaling
            bilateral_d = max(5, int(15 * intensity))
            bilateral_sigma = max(20, int(80 * intensity))
            img_cv = cv2.bilateralFilter(img_cv, bilateral_d, bilateral_sigma, bilateral_sigma)

            # Adjust parameters based on intensity and edge strength
            def ensure_odd_block_size(value):
                block_size = int(round(value))
                if block_size % 2 == 0:
                    block_size += 1
                if block_size < 3:
                    block_size = 3
                return block_size

            # Comic book style processing with intensity scaling
            data = img_cv.reshape((-1, 3))
            data = np.float32(data)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            color_levels = max(4, int(8 * intensity))
            _, labels, centers = cv2.kmeans(data, color_levels, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            centers = np.uint8(centers)
            cartoon = centers[labels.flatten()]
            cartoon = cartoon.reshape(img_cv.shape)

            # Edge detection with intensity scaling
            gray = cv2.cvtColor(cartoon, cv2.COLOR_BGR2GRAY)
            edge_size = ensure_odd_block_size(7 + (2 * intensity))  # Scale edge detection
            edge_c_value = max(1, int(edge_size / 3))
            edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, edge_size, edge_c_value)
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            
            # Apply edge intensity
            if intensity > 1.0:
                # Stronger edges for higher intensity
                cartoon = cv2.bitwise_and(cartoon, edges)
            else:
                # Softer edges for lower intensity
                edge_weight = intensity
                cartoon = cv2.addWeighted(cartoon, 1.0, edges, edge_weight * 0.3, 0)

            # Convert back to PIL
            cartoon_rgb = cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB)
            cartoon_image = Image.fromarray(cartoon_rgb)

        # Always apply enhancement at the end with intensity scaling
        enhanced_saturation = saturation * intensity
        enhanced_contrast = contrast * intensity
        cartoon_image = enhance_cartoon_colors(cartoon_image, enhanced_saturation, enhanced_contrast, brightness)
        return cartoon_image

    except Exception as e:
        st.error(f"❌ Cartoon filter failed: {e}")
        return None


def enhance_cartoon_colors(image, saturation=1.3, contrast=1.2, brightness=1.1):
    """Enhance cartoon image colors"""
    try:
        # Enhance saturation
        enhancer = ImageEnhance.Color(image)
        enhanced = enhancer.enhance(saturation)
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = enhancer.enhance(contrast)
        
        # Enhance brightness
        enhancer = ImageEnhance.Brightness(enhanced)
        enhanced = enhancer.enhance(brightness)
        
        return enhanced
        
    except Exception as e:
        st.error(f"❌ Color enhancement failed: {e}")
        return image

def create_blended_image(original_image, cartoon_image, blend_ratio=0.5):
    """Create a blended image between original and cartoon"""
    try:
        # Resize cartoon to match original
        cartoon_resized = cartoon_image.resize(original_image.size, LANCZOS_FILTER)
        
        # Convert both to RGBA for blending
        original_rgba = original_image.convert('RGBA')
        cartoon_rgba = cartoon_resized.convert('RGBA')
        
        # Create blended image
        blended = Image.blend(original_rgba, cartoon_rgba, blend_ratio)
        
        # Convert back to RGB
        blended_rgb = Image.new('RGB', blended.size, (255, 255, 255))
        blended_rgb.paste(blended, mask=blended.split()[-1])
        
        return blended_rgb
        
    except Exception as e:
        st.error(f"❌ Image blending failed: {e}")
        return None

def generate_all_styles(original_image, intensity=1.0, edge_strength=1.0, saturation=1.3, contrast=1.2, brightness=1.1, silent_mode=False):
    """Generate all cartoon styles at once"""
    styles = ["comic"]
    if ONNX_AVAILABLE:
        styles.append("animegan_hayao")
    results = {}
    
    if not silent_mode:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    for i, style in enumerate(styles):
        try:
            if not silent_mode:
                status_text.text(f"🎨 Creating {style} style...")
            
            # Apply cartoon filter
            cartoon_image = apply_cartoon_filter(original_image, style, intensity, edge_strength, saturation, contrast, brightness, silent_mode)
            
            if cartoon_image:
                results[style] = cartoon_image
            
            # Update progress
            if not silent_mode:
                progress_bar.progress((i + 1) / len(styles))
            
        except Exception as e:
            if not silent_mode:
                st.error(f"❌ Failed to create {style} style: {e}")
    
    if not silent_mode:
        progress_bar.empty()
        status_text.empty()
    
    return results

def process_all_images_batch(uploaded_files, current_settings):
    """Process all uploaded images at once"""
    if not uploaded_files:
        return
    
    st.markdown("---")
    st.subheader("🚀 Batch Processing All Images")
    
    # Initialize progress tracking
    total_images = len(uploaded_files)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Process each image
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            status_text.text(f"🎨 Processing {uploaded_file.name} ({i+1}/{total_images})...")
            
            # Load image
            original_image = Image.open(uploaded_file)
            unique_id = f"multi_{i}"
            
            # Initialize session state for this image
            if f"results_{unique_id}" not in st.session_state:
                st.session_state[f"results_{unique_id}"] = {}
            
            # Analyze with Groq
            analysis_style = "all cartoon styles" if current_settings['batch_mode'] else f"{current_settings['cartoon_style']} cartoon style"
            analysis = analyze_image_with_groq(original_image, analysis_style)
            
            if analysis:
                st.session_state[f"results_{unique_id}"]["analysis"] = analysis
                
                # Apply cartoon filters
                if current_settings['batch_mode']:
                    # Generate both styles
                    all_styles = generate_all_styles(
                        original_image, 
                        current_settings['filter_intensity'], 
                        current_settings['edge_strength'], 
                        current_settings['saturation'], 
                        current_settings['contrast'], 
                        current_settings['brightness']
                    )
                    
                    if all_styles:
                        st.session_state[f"results_{unique_id}"]["all_styles"] = all_styles
                else:
                    # Generate single style
                    cartoon_image = apply_cartoon_filter(
                        original_image, 
                        current_settings['cartoon_style'], 
                        current_settings['filter_intensity'], 
                        current_settings['edge_strength']
                    )
                    
                    if cartoon_image:
                        enhanced_cartoon = enhance_cartoon_colors(
                            cartoon_image, 
                            current_settings['saturation'], 
                            current_settings['contrast'], 
                            current_settings['brightness']
                        )
                        st.session_state[f"results_{unique_id}"]["cartoon_image"] = enhanced_cartoon
                        st.session_state[f"results_{unique_id}"]["cartoon_style"] = current_settings['cartoon_style']
            
            # Update progress
            progress_bar.progress((i + 1) / total_images)
            
        except Exception as e:
            st.error(f"❌ Failed to process {uploaded_file.name}: {str(e)}")
    
    progress_bar.empty()
    status_text.empty()
    st.success(f"✅ Batch processing completed! {total_images} images processed.")
    
    # Force rerun to show results
    st.rerun()

def create_download_all_zip(uploaded_files, current_settings):
    """Create a ZIP file with all processed images"""
    try:
        # Create a temporary directory for the ZIP
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, "cartoonified_images.zip")
            
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for i, uploaded_file in enumerate(uploaded_files):
                    unique_id = f"multi_{i}"
                    image_results = st.session_state.get(f"results_{unique_id}", {})
                    
                    if 'all_styles' in image_results:
                        # Add both styles
                        for style_name, style_image in image_results['all_styles'].items():
                            img_buffer = BytesIO()
                            style_image.save(img_buffer, format='JPEG', quality=95)
                            zipf.writestr(f"{style_name}_{uploaded_file.name}", img_buffer.getvalue())
                    elif 'cartoon_image' in image_results:
                        # Add single style
                        img_buffer = BytesIO()
                        image_results['cartoon_image'].save(img_buffer, format='JPEG', quality=95)
                        zipf.writestr(f"cartoon_{image_results['cartoon_style']}_{uploaded_file.name}", img_buffer.getvalue())
            
            # Read the ZIP file
            with open(zip_path, 'rb') as f:
                return f.read()
                
    except Exception as e:
        st.error(f"❌ Failed to create ZIP file: {e}")
        return None

def process_image_input(original_image, filename, current_settings, unique_id="", is_multi_mode=False):
    """Process uploaded image with individual results storage"""
    try:
        # Initialize session state for this specific image if not exists
        if f"results_{unique_id}" not in st.session_state:
            st.session_state[f"results_{unique_id}"] = {}
        
        st.markdown("---")
        if is_multi_mode:
            st.subheader(f"📷 {filename}")
        else:
            st.subheader("📷 Your Original Image")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.image(original_image, caption=f"Original: {filename}", use_container_width=True)
        
        with col2:
            st.markdown("**📊 Image Info:**")
            st.write(f"**Size:** {original_image.size[0]} × {original_image.size[1]} px")
            st.write(f"**Format:** {original_image.format}")
            st.write(f"**Mode:** {original_image.mode}")
        
        # Generate cartoon button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            button_text = "🚀 Generate Both Styles with Groq" if current_settings['batch_mode'] else "🚀 Analyze & Cartoonify with Groq"
            if st.button(button_text, type="primary", use_container_width=True, key=f"generate_btn_{unique_id}"):
                
                # Start timing
                start_time = time.time()
                
                # Step 1: Groq Vision Analysis
                with st.spinner("🧠 Groq Vision is analyzing your image..."):
                    analysis_style = "all cartoon styles" if current_settings['batch_mode'] else f"{current_settings['cartoon_style']} cartoon style"
                    analysis = analyze_image_with_groq(original_image, analysis_style)
                    
                    if analysis:
                        st.session_state[f"results_{unique_id}"]["analysis"] = analysis
                        analysis_time = time.time() - start_time
                        st.success(f"✅ Image analyzed in {analysis_time:.1f}s!")
                    else:
                        st.error("❌ Analysis failed. Please try again.")
                        st.stop()
                
                # Step 2: Apply Cartoon Filters
                if current_settings['batch_mode']:
                    # Generate both styles
                    with st.spinner("🎨 Generating both cartoon styles..."):
                        all_styles = generate_all_styles(
                            original_image, 
                            current_settings['filter_intensity'], 
                            current_settings['edge_strength'], 
                            current_settings['saturation'], 
                            current_settings['contrast'], 
                            current_settings['brightness']
                        )
                        
                        if all_styles:
                            total_time = time.time() - start_time
                            st.session_state[f"results_{unique_id}"]["all_styles"] = all_styles
                            st.session_state[f"results_{unique_id}"]["processing_time"] = total_time
                            # Set flag for captured images
                            if unique_id == "captured":
                                st.session_state["captured_processed"] = True
                            st.success(f"✅ Created {len(all_styles)} cartoon styles in {total_time:.1f}s!")
                        else:
                            st.error("❌ Batch generation failed. Please try again.")
                else:
                    # Generate single style
                    with st.spinner(f"🎨 Applying {current_settings['cartoon_style']} cartoon filters..."):
                        cartoon_image = apply_cartoon_filter(
                            original_image, 
                            current_settings['cartoon_style'], 
                            current_settings['filter_intensity'], 
                            current_settings['edge_strength']
                        )
                        
                        if cartoon_image:
                            # Enhance colors
                            enhanced_cartoon = enhance_cartoon_colors(
                                cartoon_image, 
                                current_settings['saturation'], 
                                current_settings['contrast'], 
                                current_settings['brightness']
                            )
                            
                            total_time = time.time() - start_time
                            st.session_state[f"results_{unique_id}"]["cartoon_image"] = enhanced_cartoon
                            st.session_state[f"results_{unique_id}"]["cartoon_style"] = current_settings['cartoon_style']
                            st.session_state[f"results_{unique_id}"]["processing_time"] = total_time
                            # Set flag for captured images
                            if unique_id == "captured":
                                st.session_state["captured_processed"] = True
                            st.success(f"✅ Cartoon created in {total_time:.1f}s!")
                        else:
                            st.error("❌ Cartoon generation failed. Please try again.")
        
        # Get results for this specific image
        image_results = st.session_state.get(f"results_{unique_id}", {})
        
        # Show batch results if available
        if 'all_styles' in image_results and 'analysis' in image_results:
            st.markdown("---")
            st.subheader("🎨 Both Cartoon Styles Generated")
            
            # Display all styles in a grid
            styles = image_results['all_styles']
            col1, col2 = st.columns(2)
            
            for i, (style_name, style_image) in enumerate(styles.items()):
                with col1 if i % 2 == 0 else col2:
                    st.image(style_image, caption=f"{style_name.title()} Style", use_container_width=True)
            
            # Show AI Analysis
            with st.expander("🧠 Groq Vision Analysis", expanded=False):
                st.write(image_results['analysis'])
            
            # Batch download section
            st.markdown("---")
            st.subheader("📥 Download Both Styles")
            
            download_cols = st.columns(len(styles))
            
            # Style downloads
            for i, (style_name, style_image) in enumerate(styles.items()):
                with download_cols[i]:
                    st.markdown(f"**🎨 {style_name.title()}**")
                    img_buffer = BytesIO()
                    style_image.save(img_buffer, format='JPEG', quality=95)
                    st.download_button(
                        f"📥 {style_name.title()}",
                        data=img_buffer.getvalue(),
                        file_name=f"{style_name}_{filename}",
                        mime="image/jpeg",
                        use_container_width=True,
                        key=f"download_{style_name}_{unique_id}"
                    )
        
        # Show single results if available
        elif 'cartoon_image' in image_results and 'analysis' in image_results:
            st.markdown("---")
            st.subheader("🎨 Your Cartoon Result")
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(original_image, caption="Original Image", use_container_width=True)
            with col2:
                st.image(image_results['cartoon_image'], caption=f"{image_results['cartoon_style'].title()} Cartoon", use_container_width=True)
            
            # Show AI Analysis
            with st.expander("🧠 Groq Vision Analysis", expanded=False):
                st.write(image_results['analysis'])
            
            # Blending section
            st.markdown("---")
            st.subheader("🔀 Blend with Original")
            
            blend_ratio = st.slider(
                "Cartoon Effect Strength",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="0 = Original image, 1 = Full cartoon effect",
                key=f"blend_slider_{unique_id}"
            )
            
            # Create blended image
            blended_image = create_blended_image(original_image, image_results['cartoon_image'], blend_ratio)
            
            if blended_image:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.image(original_image, caption="Original", use_container_width=True)
                
                with col2:
                    st.image(blended_image, caption=f"Blended ({int(blend_ratio*100)}%)", use_container_width=True)
                
                with col3:
                    st.image(image_results['cartoon_image'], caption="Full Cartoon", use_container_width=True)
                
                # Store blended image for this specific image
                st.session_state[f"results_{unique_id}"]["blended_image"] = blended_image
                st.session_state[f"results_{unique_id}"]["blend_ratio"] = blend_ratio
            
            # Download section
            st.markdown("---")
            st.subheader("📥 Download Your Creations")
            
            download_cols = st.columns(2)
            
            with download_cols[0]:
                st.markdown("**🎨 Cartoon**")
                img_buffer = BytesIO()
                image_results['cartoon_image'].save(img_buffer, format='JPEG', quality=95)
                st.download_button(
                    "📥 Download Cartoon",
                    data=img_buffer.getvalue(),
                    file_name=f"cartoon_{image_results['cartoon_style']}_{filename}",
                    mime="image/jpeg",
                    use_container_width=True,
                    key=f"download_cartoon_{unique_id}"
                )
            
            with download_cols[1]:
                if 'blended_image' in image_results:
                    st.markdown("**🔀 Blended**")
                    img_buffer = BytesIO()
                    image_results['blended_image'].save(img_buffer, format='JPEG', quality=95)
                    st.download_button(
                        "📥 Download Blend",
                        data=img_buffer.getvalue(),
                        file_name=f"blended_{int(image_results['blend_ratio']*100)}_{filename}",
                        mime="image/jpeg",
                        use_container_width=True,
                        key=f"download_blend_{unique_id}"
                    )
    
    except Exception as e:
        st.error(f"❌ Image processing failed: {str(e)}")

def process_video(video_file, current_settings):
    """Process uploaded video file"""
    temp_video_path = None
    output_path = None
    
    try:
        st.markdown("---")
        st.subheader("🎬 Video Processing")
        
        # Save uploaded video to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(video_file.read())
            temp_video_path = tmp_file.name
        
        # Display video info
        st.video(video_file)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.info("📹 **Video uploaded successfully!**")
            st.write(f"**File:** {video_file.name}")
        
        with col2:
            if st.button("🎨 Process Video", type="primary", use_container_width=True, key="process_video"):
                with st.spinner("🎬 Processing video frames..."):
                    # Open video with OpenCV
                    cap = cv2.VideoCapture(temp_video_path)
                    
                    # Get video properties
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    st.info(f"📊 **Video Info:** {width}x{height}, {fps} FPS, {total_frames} frames")
                    
                    # Create output video writer
                    output_path = temp_video_path.replace('.mp4', '_cartoon.mp4')
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    
                    # Process frames
                    progress_bar = st.progress(0)
                    frame_count = 0
                    
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Convert frame to PIL Image (preserve original orientation)
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_frame = Image.fromarray(frame_rgb)
                        
                        # Apply cartoon filter (silent mode to avoid repeated messages)
                        cartoon_frame = apply_cartoon_filter(
                            pil_frame, 
                            current_settings['cartoon_style'], 
                            current_settings['filter_intensity'],
                            current_settings['edge_strength'],
                            current_settings['saturation'],
                            current_settings['contrast'],
                            current_settings['brightness'],
                            silent_mode=True
                        )
                        
                        if cartoon_frame:
                            # Convert back to OpenCV format
                            cartoon_array = np.array(cartoon_frame)
                            cartoon_bgr = cv2.cvtColor(cartoon_array, cv2.COLOR_RGB2BGR)
                            out.write(cartoon_bgr)
                        
                        frame_count += 1
                        progress_bar.progress(frame_count / total_frames)
                    
                    cap.release()
                    out.release()
                    progress_bar.empty()
                    
                    st.success("✅ Video processing completed!")
                    
                    # Provide download link
                    with open(output_path, 'rb') as f:
                        st.download_button(
                            "📥 Download Cartoon Video",
                            data=f.read(),
                            file_name=f"cartoon_{video_file.name}",
                            mime="video/mp4",
                            use_container_width=True,
                            key="download_video"
                        )
        
    except Exception as e:
        st.error(f"❌ Video processing failed: {str(e)}")
    
    finally:
        # Clean up temporary files
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.unlink(temp_video_path)
            except:
                pass
        if output_path and os.path.exists(output_path):
            try:
                os.unlink(output_path)
            except:
                pass

def display_captured_image_results(original_image, filename, unique_id):
    """Display the results of captured image processing"""
    try:
        # Get results for this specific image
        image_results = st.session_state.get(f"results_{unique_id}", {})
        
        if not image_results:
            st.error("❌ No results found for captured image")
            return
        
        st.markdown("---")
        st.subheader("📷 Your Captured Image Results")
        
        # Show batch results if available
        if 'all_styles' in image_results and 'analysis' in image_results:
            st.subheader("🎨 Both Cartoon Styles Generated")
            
            # Display all styles in a grid
            styles = image_results['all_styles']
            col1, col2 = st.columns(2)
            
            for i, (style_name, style_image) in enumerate(styles.items()):
                with col1 if i % 2 == 0 else col2:
                    st.image(style_image, caption=f"{style_name.title()} Style", use_container_width=True)
            
            # Show AI Analysis
            with st.expander("🧠 Groq Vision Analysis", expanded=False):
                st.write(image_results['analysis'])
            
            # Batch download section
            st.markdown("---")
            st.subheader("📥 Download Both Styles")
            
            download_cols = st.columns(len(styles))
            
            # Style downloads
            for i, (style_name, style_image) in enumerate(styles.items()):
                with download_cols[i]:
                    st.markdown(f"**🎨 {style_name.title()}**")
                    img_buffer = BytesIO()
                    style_image.save(img_buffer, format='JPEG', quality=95)
                    st.download_button(
                        f"📥 {style_name.title()}",
                        data=img_buffer.getvalue(),
                        file_name=f"{style_name}_{filename}",
                        mime="image/jpeg",
                        use_container_width=True,
                        key=f"download_{style_name}_{unique_id}"
                    )
        
        # Show single results if available
        elif 'cartoon_image' in image_results and 'analysis' in image_results:
            st.subheader("🎨 Your Cartoon Result")
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(original_image, caption="Original Image", use_container_width=True)
            with col2:
                st.image(image_results['cartoon_image'], caption=f"{image_results['cartoon_style'].title()} Cartoon", use_container_width=True)
            
            # Show AI Analysis
            with st.expander("🧠 Groq Vision Analysis", expanded=False):
                st.write(image_results['analysis'])
            
            # Blending section
            st.markdown("---")
            st.subheader("🔀 Blend with Original")
            
            blend_ratio = st.slider(
                "Cartoon Effect Strength",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="0 = Original image, 1 = Full cartoon effect",
                key=f"blend_slider_{unique_id}"
            )
            
            # Create blended image
            blended_image = create_blended_image(original_image, image_results['cartoon_image'], blend_ratio)
            
            if blended_image:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.image(original_image, caption="Original", use_container_width=True)
                
                with col2:
                    st.image(blended_image, caption=f"Blended ({int(blend_ratio*100)}%)", use_container_width=True)
                
                with col3:
                    st.image(image_results['cartoon_image'], caption="Full Cartoon", use_container_width=True)
                
                # Store blended image for this specific image
                st.session_state[f"results_{unique_id}"]["blended_image"] = blended_image
                st.session_state[f"results_{unique_id}"]["blend_ratio"] = blend_ratio
            
            # Download section
            st.markdown("---")
            st.subheader("📥 Download Your Creations")
            
            download_cols = st.columns(2)
            
            with download_cols[0]:
                st.markdown("**🎨 Cartoon**")
                img_buffer = BytesIO()
                image_results['cartoon_image'].save(img_buffer, format='JPEG', quality=95)
                st.download_button(
                    "📥 Download Cartoon",
                    data=img_buffer.getvalue(),
                    file_name=f"cartoon_{image_results['cartoon_style']}_{filename}",
                    mime="image/jpeg",
                    use_container_width=True,
                    key=f"download_cartoon_{unique_id}"
                )
            
            with download_cols[1]:
                if 'blended_image' in image_results:
                    st.markdown("**🔀 Blended**")
                    img_buffer = BytesIO()
                    image_results['blended_image'].save(img_buffer, format='JPEG', quality=95)
                    st.download_button(
                        "📥 Download Blend",
                        data=img_buffer.getvalue(),
                        file_name=f"blended_{int(image_results['blend_ratio']*100)}_{filename}",
                        mime="image/jpeg",
                        use_container_width=True,
                        key=f"download_blend_{unique_id}"
                    )
    
    except Exception as e:
        st.error(f"❌ Results display failed: {str(e)}")

def process_captured_image_automatically(original_image, filename, current_settings):
    """Automatically process captured image without requiring button click"""
    try:
        unique_id = "captured"
        
        # Initialize session state for this specific image if not exists
        if f"results_{unique_id}" not in st.session_state:
            st.session_state[f"results_{unique_id}"] = {}
        
        st.markdown("---")
        st.subheader("📷 Processing Captured Image")
        
        # Show the original image
        col1, col2 = st.columns([3, 1])
        with col1:
            st.image(original_image, caption=f"Captured: {filename}", use_container_width=True)
        
        with col2:
            st.markdown("**📊 Image Info:**")
            st.write(f"**Size:** {original_image.size[0]} × {original_image.size[1]} px")
            st.write(f"**Format:** {original_image.format or 'JPEG'}")
            st.write(f"**Mode:** {original_image.mode}")
        
        # Start timing
        start_time = time.time()
        
        # Step 1: Groq Vision Analysis
        with st.spinner("🧠 Groq Vision is analyzing your captured image..."):
            analysis_style = "all cartoon styles" if current_settings['batch_mode'] else f"{current_settings['cartoon_style']} cartoon style"
            analysis = analyze_image_with_groq(original_image, analysis_style)
            
            if analysis:
                st.session_state[f"results_{unique_id}"]["analysis"] = analysis
                analysis_time = time.time() - start_time
                st.success(f"✅ Image analyzed in {analysis_time:.1f}s!")
            else:
                st.error("❌ Analysis failed. Please try again.")
                return
        
        # Step 2: Apply Cartoon Filters
        if current_settings['batch_mode']:
            # Generate both styles
            with st.spinner("🎨 Generating both cartoon styles..."):
                all_styles = generate_all_styles(
                    original_image, 
                    current_settings['filter_intensity'], 
                    current_settings['edge_strength'], 
                    current_settings['saturation'], 
                    current_settings['contrast'], 
                    current_settings['brightness']
                )
                
                if all_styles:
                    total_time = time.time() - start_time
                    st.session_state[f"results_{unique_id}"]["all_styles"] = all_styles
                    st.session_state[f"results_{unique_id}"]["processing_time"] = total_time
                    # Set flag for captured images
                    st.session_state["captured_processed"] = True
                    st.success(f"✅ Created {len(all_styles)} cartoon styles in {total_time:.1f}s!")
                    
                    # Force rerun to show results
                    st.rerun()
                else:
                    st.error("❌ Batch generation failed. Please try again.")
        else:
            # Generate single style
            with st.spinner(f"🎨 Applying {current_settings['cartoon_style']} cartoon filters..."):
                cartoon_image = apply_cartoon_filter(
                    original_image, 
                    current_settings['cartoon_style'], 
                    current_settings['filter_intensity'], 
                    current_settings['edge_strength']
                )
                
                if cartoon_image:
                    # Enhance colors
                    enhanced_cartoon = enhance_cartoon_colors(
                        cartoon_image, 
                        current_settings['saturation'], 
                        current_settings['contrast'], 
                        current_settings['brightness']
                    )
                    
                    total_time = time.time() - start_time
                    st.session_state[f"results_{unique_id}"]["cartoon_image"] = enhanced_cartoon
                    st.session_state[f"results_{unique_id}"]["cartoon_style"] = current_settings['cartoon_style']
                    st.session_state[f"results_{unique_id}"]["processing_time"] = total_time
                    # Set flag for captured images
                    st.session_state["captured_processed"] = True
                    st.success(f"✅ Cartoon created in {total_time:.1f}s!")
                    
                    # Force rerun to show results
                    st.rerun()
                else:
                    st.error("❌ Cartoon generation failed. Please try again.")
    
    except Exception as e:
        st.error(f"❌ Captured image processing failed: {str(e)}")

def capture_image_from_camera():
    """Capture image from camera using streamlit-webrtc"""
    st.markdown("---")
    st.subheader("📸 Camera Capture")
    
    # Check if we should start processing
    if st.session_state.get('processing_captured', False):
        # Clear the processing flag and start processing
        del st.session_state.processing_captured
        
        # Use the captured image with timestamp to ensure fresh processing
        captured_img = st.session_state.captured_image
        timestamp = st.session_state.get('capture_timestamp', time.time())
        filename = f"captured_image_{int(timestamp)}.jpg"
        
        # Process the captured image automatically
        process_captured_image_automatically(captured_img, filename, st.session_state.current_settings)
        return
    
    # Check if we have processing results to show (check if results exist for captured image)
    if "captured_processed" in st.session_state and st.session_state["captured_processed"]:
        # If we have results, display them directly
        if 'captured_image' in st.session_state:
            captured_img = st.session_state.captured_image
            timestamp = st.session_state.get('capture_timestamp', time.time())
            filename = f"captured_image_{int(timestamp)}.jpg"
            
            # Display results directly without processing button
            display_captured_image_results(captured_img, filename, "captured")
            
            # Add buttons to manage the captured image state
            st.markdown("---")
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("📸 Take Another Photo", use_container_width=True, key="take_another_after_results"):
                    # Clear all captured image data and show camera interface again
                    if 'captured_image' in st.session_state:
                        del st.session_state.captured_image
                    if 'capture_timestamp' in st.session_state:
                        del st.session_state.capture_timestamp
                    if "captured_processed" in st.session_state:
                        del st.session_state["captured_processed"]
                    if "show_camera" in st.session_state:
                        del st.session_state.show_camera
                    if "processing_captured" in st.session_state:
                        del st.session_state.processing_captured
                    st.rerun()
            
            with col2:
                if st.button("🗑️ Clear All", use_container_width=True, key="clear_all_after_results"):
                    # Clear all captured image data
                    if 'captured_image' in st.session_state:
                        del st.session_state.captured_image
                    if 'capture_timestamp' in st.session_state:
                        del st.session_state.capture_timestamp
                    if "captured_processed" in st.session_state:
                        del st.session_state["captured_processed"]
                    if "show_camera" in st.session_state:
                        del st.session_state.show_camera
                    if "processing_captured" in st.session_state:
                        del st.session_state.processing_captured
                    st.rerun()
        return
    
    # Check if we have a captured image and are not showing camera
    if 'captured_image' in st.session_state and not st.session_state.get('show_camera', True):
        # Show captured image and processing options
        st.markdown("### 📷 Captured Image")
        st.image(st.session_state.captured_image, caption="Captured from camera", use_container_width=True)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("🎨 Cartoonify Captured Image", type="primary", use_container_width=True, key="cartoonify_captured"):
                # Set a flag to indicate we're processing
                st.session_state.processing_captured = True
                st.rerun()
        
        with col2:
            if st.button("📸 Take Another Photo", use_container_width=True, key="take_another"):
                # Show camera interface again
                st.session_state.show_camera = True
                # Clear any processing flags
                if "processing_captured" in st.session_state:
                    del st.session_state.processing_captured
                if "captured_processed" in st.session_state:
                    del st.session_state["captured_processed"]
                st.rerun()
        
        with col3:
            if st.button("🗑️ Clear Captured Image", use_container_width=True, key="clear_captured"):
                # Clear both the captured image and its results
                if 'captured_image' in st.session_state:
                    del st.session_state.captured_image
                if 'capture_timestamp' in st.session_state:
                    del st.session_state.capture_timestamp
                if "captured_processed" in st.session_state:
                    del st.session_state["captured_processed"]
                if "show_camera" in st.session_state:
                    del st.session_state.show_camera
                if "processing_captured" in st.session_state:
                    del st.session_state.processing_captured
                st.rerun()
        return
    
    # Show camera interface
    class VideoTransformer(VideoTransformerBase):
        def __init__(self):
            self.captured_frame = None
            
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            self.captured_frame = img
            return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    # WebRTC configuration
    rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        webrtc_ctx = webrtc_streamer(
            key="camera",
            video_transformer_factory=VideoTransformer,
            rtc_configuration=rtc_config,
            media_stream_constraints={"video": True, "audio": False},
        )
    
    with col2:
        st.markdown("**📸 Instructions:**")
        st.write("1. Allow camera access")
        st.write("2. Position yourself")
        st.write("3. Click capture button")
        
        if st.button("� Capture Image", type="primary", use_container_width=True, key="capture_image"):
            if webrtc_ctx.video_transformer:
                captured_frame = webrtc_ctx.video_transformer.captured_frame
                if captured_frame is not None:
                    # Convert to PIL Image
                    frame_rgb = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB)
                    captured_image = Image.fromarray(frame_rgb)
                    
                    # Clear any previous captured image results to avoid cache conflicts
                    if "captured_processed" in st.session_state:
                        del st.session_state["captured_processed"]
                    
                    # Store in session state with a unique timestamp to force refresh
                    st.session_state.captured_image = captured_image
                    st.session_state.capture_timestamp = time.time()
                    st.session_state.show_camera = False  # Hide camera interface
                    st.success("✅ Image captured successfully!")
                    st.rerun()
                else:
                    st.error("❌ No frame captured. Make sure camera is active.")
            else:
                st.error("❌ Camera not initialized. Please start the camera first.")

# === Streamlit App ===
def main():
    st.set_page_config(
        page_title="Groq Vision Cartoonify",
        page_icon="🚀",
        layout="wide"
    )
    
    st.title("Groq Vision Cartoonify - AI-Powered Cartoon Generation")
    if ONNX_AVAILABLE:
        st.markdown("Transform your **photos** into cartoons using **Groq's Llama Vision + AnimeGANv3 + Comic Filters**!")
        st.success("✨ **AnimeGANv3 Available** - Professional Studio Ghibli style conversion enabled!")
    else:
        st.markdown("Transform your **photos** into cartoons using **Groq's Llama Vision + Comic Filters**!")
        st.warning("⚠️ **AnimeGANv3 Not Available** - Install ONNX Runtime for professional anime conversion:")
        st.code("pip install onnxruntime", language="bash")
    
    # Check Groq API
    api_working, api_message = test_groq_api()
    if not api_working:
        st.error(api_message)
        st.markdown("""
        **To fix this:**
        1. Get your free Groq API key from: https://console.groq.com/keys
        2. Create a `.env` file in your project directory
        3. Add to your .env file: `GROQ_API_KEY=your_key_here`
        """)
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("🚀 Groq Advantages")
        st.success("✅ FREE tier with generous limits")
        st.success("✅ Lightning fast inference")
        st.success("✅ Best Llama Models")
        st.success("✅ No OpenAI costs!")
        
        st.header("🎨 Cartoon Styles")
        # Updated style options
        if ONNX_AVAILABLE:
            style_options = ["animegan_hayao", "comic"]
            default_index = 0  # animegan_hayao is first, so index 0
            help_text = "AnimeGANv3 Hayao provides professional Studio Ghibli style, Comic style uses OpenCV"
        else:
            style_options = ["comic"]
            default_index = 0
            help_text = "Comic style using OpenCV filters"
        
        cartoon_style = st.selectbox(
            "Choose cartoon style:",
            style_options,
            index=default_index,
            help=help_text
        )
        
        # Show style descriptions
        if cartoon_style == "comic":
            st.info("💥 **Comic Style:** Bold colors and strong edges using OpenCV")
        elif cartoon_style == "animegan_hayao":
            st.success("✨ **AnimeGANv3 Hayao:** Professional Studio Ghibli style using deep learning")
        
        # Updated quick style presets
        st.markdown("**⚡ Quick Presets:**")
        if ONNX_AVAILABLE:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✨ Hayao", use_container_width=True, key="preset_hayao"):
                    st.session_state.preset_applied = "animegan_hayao"
                    st.rerun()
            with col2:
                if st.button("💥 Comic", use_container_width=True, key="preset_comic_1"):
                    st.session_state.preset_applied = "comic"
                    st.rerun()
        else:
            if st.button("💥 Comic", use_container_width=True, key="preset_comic_2"):
                st.session_state.preset_applied = "comic"
                st.rerun()
        
        st.header("🔧 Filter Settings")
        
        # Set default values for all settings (filter intensity slider removed)
        filter_intensity = 1.0
        saturation = 1.3
        contrast = 1.2
        brightness = 1.1
        edge_strength = 1.0
        
        st.header("⚡ Batch Generation")
        batch_mode = st.checkbox("Generate Both Styles", help="Create both comic and AnimeGAN styles at once")
        
        if batch_mode:
            st.info("💡 This will generate both comic and AnimeGAN styles simultaneously!")
    
    # Store slider values in session state
    st.session_state.current_settings = {
        'cartoon_style': cartoon_style,
        'saturation': saturation,
        'contrast': contrast,
        'brightness': brightness,
        'filter_intensity': filter_intensity,
        'edge_strength': edge_strength,
        'batch_mode': batch_mode
    }
    
    # Input Mode Selection
    st.markdown("### 📁 Choose Input Method")
    
    # Create tabs for different input methods
    tab1, tab2, tab3, tab4 = st.tabs(["📷 Multiple Images", "🎬 Video", "📸 Camera Capture", "🖼️ Single Image"])
    
    with tab1:
        st.markdown("#### 📷 Upload Multiple Images")
        uploaded_files = st.file_uploader(
            "Choose multiple image files", 
            type=["jpg", "jpeg", "png", "bmp", "tiff"],
            accept_multiple_files=True,
            key="multiple_images"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} images uploaded!")
            
            # Show processing summary
            processed_count = 0
            for i in range(len(uploaded_files)):
                unique_id = f"multi_{i}"
                if f"results_{unique_id}" in st.session_state:
                    image_results = st.session_state[f"results_{unique_id}"]
                    if 'all_styles' in image_results or 'cartoon_image' in image_results:
                        processed_count += 1
            
            if processed_count > 0:
                st.info(f"📊 **Processing Status:** {processed_count}/{len(uploaded_files)} images processed")
            
            # Batch processing options
            st.markdown("---")
            st.subheader("🚀 Processing Options")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                if st.button("🚀 Process All Images", type="primary", use_container_width=True, key="process_all_btn"):
                    process_all_images_batch(uploaded_files, st.session_state.current_settings)
            
            with col2:
                # Use the already calculated processed_count
                if processed_count > 0:
                    zip_data = create_download_all_zip(uploaded_files, st.session_state.current_settings)
                    if zip_data:
                        st.download_button(
                            f"📦 Download All ({processed_count})",
                            data=zip_data,
                            file_name="cartoonified_images.zip",
                            mime="application/zip",
                            use_container_width=True,
                            key="download_all_zip"
                        )
                else:
                    st.button("📦 Download All", disabled=True, use_container_width=True, help="Process images first")
            
            with col3:
                if st.button("🗑️ Clear All Results", use_container_width=True, key="clear_all_btn"):
                    # Clear all multi-image results
                    for i in range(len(uploaded_files)):
                        unique_id = f"multi_{i}"
                        if f"results_{unique_id}" in st.session_state:
                            del st.session_state[f"results_{unique_id}"]
                    # Also clear any cached results and captured images
                    keys_to_clear = [key for key in st.session_state.keys() if key.startswith('results_') or key == 'captured_image']
                    for key in keys_to_clear:
                        del st.session_state[key]
                    st.success("✅ All results and cache cleared!")
                    st.rerun()
            
            # Individual image processing
            st.markdown("---")
            st.subheader("🖼️ Individual Image Processing")
            st.info("💡 **Tip:** You can process images individually using the buttons below, or use 'Process All Images' for batch processing.")
            
            # Process each image individually
            for i, uploaded_file in enumerate(uploaded_files):
                try:
                    st.markdown(f"---")
                    
                    # Header with status and clear button for individual image
                    col_header1, col_header2, col_header3 = st.columns([3, 1, 1])
                    with col_header1:
                        st.markdown(f"### 🖼️ Image {i+1}: {uploaded_file.name}")
                    with col_header2:
                        unique_id = f"multi_{i}"
                        # Show processing status
                        if f"results_{unique_id}" in st.session_state:
                            image_results = st.session_state[f"results_{unique_id}"]
                            if 'all_styles' in image_results or 'cartoon_image' in image_results:
                                st.success("✅ Processed")
                            else:
                                st.info("⏳ Analyzing...")
                        else:
                            st.warning("⚪ Not processed")
                    with col_header3:
                        if f"results_{unique_id}" in st.session_state:
                            if st.button("🗑️ Clear", key=f"clear_individual_{i}", help=f"Clear results for {uploaded_file.name}"):
                                del st.session_state[f"results_{unique_id}"]
                                st.success(f"✅ Cleared results for {uploaded_file.name}")
                                st.rerun()
                    
                    # Load and display original image
                    original_image = Image.open(uploaded_file)
                    process_image_input(original_image, uploaded_file.name, st.session_state.current_settings, f"multi_{i}", is_multi_mode=True)
                    
                except Exception as e:
                    st.error(f"❌ Failed to process {uploaded_file.name}: {str(e)}")
    
    with tab2:
        st.markdown("#### 🎬 Upload Video")
        uploaded_video = st.file_uploader(
            "Choose a video file", 
            type=["mp4", "avi", "mov", "mkv"],
            key="video_upload"
        )
        
        if uploaded_video is not None:
            process_video(uploaded_video, st.session_state.current_settings)
    
    with tab3:
        st.markdown("#### 📸 Camera Capture")
        capture_image_from_camera()
    
    with tab4:
        st.markdown("#### 🖼️ Upload Single Image")
        uploaded_file = st.file_uploader(
            "Choose an image file", 
            type=["jpg", "jpeg", "png", "bmp", "tiff"],
            key="single_image"
        )
        
        if uploaded_file is not None:
            try:
                # Load and display original image
                original_image = Image.open(uploaded_file)
                process_image_input(original_image, uploaded_file.name, st.session_state.current_settings, "single")
            except Exception as e:
                st.error(f"Image processing failed: {str(e)}")

if __name__ == "__main__":
    main()
