import cv2
import numpy as np
import gradio as gr
from PIL import Image
import os
import uuid

# Placing the output directory in the same folder as the script
outpath_path=os.path.join("Caartoonify_AI", "output")
os.makedirs(outpath_path,exist_ok=True)

# Cartoonify function with all filters
def cartoonify_image(img,blur_val,color_strength,edge_thresh,cartooness,sharpness,show_comparison):
    # converting to opencv format
    img=np.array(img)
    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    img=cv2.resize(img,(600,600))

    # Grayscale and Median Blur
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray_blur=cv2.medianBlur(gray,blur_val)

    # Edge Detection
    edges = cv2.adaptiveThreshold(
    gray_blur, 255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY,
    9, edge_thresh
)

    
    # Color Smoothening
    color=cv2.bilateralFilter(img,sigmaColor=color_strength,
                              sigmaSpace=color_strength,d=0)
    
    # Cartoon Effect
    cartoon=cv2.bitwise_and(color,color,mask=edges)

    # Cartooness Adjustment
    cartoon_mixed=cv2.addWeighted(cartoon,cartooness,img,1-cartooness,0)

    # Sharpness Kernel
    if sharpness > 0:
        kernel=np.array([[0,-1,0],
                         [-1,5+sharpness,-1],
                         [0,-1,0]])
        cartoon_mixed=cv2.filter2D(cartoon_mixed,-1,kernel)

    # Enabling Side by Side Comparison
    if show_comparison:
        comparison=np.hstack((img,cartoon_mixed))
    else:
        comparison=cartoon_mixed

    # Converting back to RGB PIL
    final_rgb=cv2.cvtColor(comparison,cv2.COLOR_BGR2RGB)
    pil_result=Image.fromarray(final_rgb)

    # Saving File
    filename = os.path.join(outpath_path, f"cartoon_{uuid.uuid4().hex[:8]}.png")
    pil_result.save(filename)

    return pil_result,filename

# Batch processor for multiple images
def process_images(imgs, blur_val, color_strength, edge_thresh, cartoonness, sharpness, show_comparison):
    outputs = []
    for img in imgs:
        cartoon_img, path = cartoonify_image(img, blur_val, color_strength, edge_thresh, cartoonness, sharpness, show_comparison)
        outputs.append((cartoon_img, path))
    return outputs


# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🎨 Advanced AI Cartoonify App")
    gr.Markdown("Upload images and adjust filters to cartoonify. Preview side-by-side and download results.")

    with gr.Row():
        img_input = gr.File(label="Upload Images", file_types=[".jpg", ".png"], file_count="multiple")

    with gr.Accordion("Image Filters", open=True):
        with gr.Row():
            blur = gr.Slider(1, 15, value=5, step=2, label="Blur")
            color_strength = gr.Slider(50, 300, value=150, label="Color Strength")
            edge_thresh = gr.Slider(1, 20, value=9, label="Edge Threshold")
        with gr.Row():
            cartoonness = gr.Slider(0, 1, value=1.0, step=0.05, label="Cartoonness")
            sharpness = gr.Slider(0, 5, value=1.0, step=0.1, label="Sharpness")
            show_comparison = gr.Checkbox(label="Show Original vs Cartoon Side-by-Side", value=True)

    run_btn = gr.Button("Cartoonify Now")

    gallery = gr.Gallery(label="Cartoonified Results")
    download_list = gr.File(label="Download Files", file_types=[".png"], file_count="multiple")

    def run_all(img_input, blur, color_strength, edge_thresh, cartoonness, sharpness, show_comparison):
        images = [Image.open(img.name) for img in img_input]
        results = process_images(images, blur, color_strength, edge_thresh, cartoonness, sharpness, show_comparison)
        images_only, paths_only = zip(*results)
        return list(images_only), list(paths_only)

    run_btn.click(fn=run_all,
                  inputs=[img_input, blur, color_strength, edge_thresh, cartoonness, sharpness, show_comparison],
                  outputs=[gallery, download_list])

if __name__ == "__main__":
    demo.launch()

