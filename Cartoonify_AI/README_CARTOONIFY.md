# 🎨 Cartoonify AI - Advanced Image & Video Cartoonification

Transform your photos and videos into stunning cartoon-style artwork using AI-powered filters and computer vision techniques!

## ✨ Features

### 🖼️ Image Processing
- **Multiple Cartoon Styles**: Comic book style and AnimeGAN Hayao style
- **Batch Processing**: Process multiple images simultaneously
- **Real-time Camera Capture**: Take photos and cartoonify instantly
- **AI-Powered Analysis**: Groq LLM analyzes images and provides detailed descriptions
- **Customizable Filters**: Adjust intensity, edge strength, saturation, contrast, and brightness
- **Blend Mode**: Create artistic blends between original and cartoon versions

### 🎬 Video Processing
- **Full Video Cartoonification**: Transform entire videos frame by frame
- **Automatic Orientation**: Maintains correct video orientation
- **Progress Tracking**: Real-time processing progress with frame count
- **Multiple Formats**: Supports various video formats
- **Download Ready**: Processed videos ready for immediate download

### 🎛️ Advanced Controls
- **Color Enhancement**: Fine-tune saturation (0.5 - 2.0), contrast (0.5 - 2.0), and brightness (0.5 - 2.0)
- **Preset Configurations**: Quick-apply settings for different styles
- **Batch Mode**: Generate multiple styles simultaneously

### 🤖 AI Integration
- **Groq LLM Vision**: Advanced image analysis and description generation
- **AnimeGAN v3**: Professional anime-style transformation (ONNX model)
- **Intelligent Processing**: Automatic fallback to OpenCV filters when needed

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Groq API Key
- Webcam (optional, for camera capture)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd Cartoonify_AI
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
```

4. **Download AnimeGAN model (optional)**
Place `AnimeGANv3_Hayao_STYLE_36.onnx` in the project directory for enhanced anime-style processing.

5. **Run the application**
```bash
streamlit run groq_cartoonify.py
```

## 📋 Requirements

### Core Dependencies
```
streamlit>=1.28.0
opencv-python>=4.8.0
Pillow>=10.0.0
numpy>=1.24.0
python-dotenv>=1.0.0
langchain-groq>=0.1.0
langchain>=0.1.0
streamlit-webrtc>=0.47.0
requests>=2.31.0
```

### Optional Dependencies
```
onnxruntime>=1.16.0  # For AnimeGAN v3 support
```

## 🎯 Usage Guide

### 1. Image Processing

#### Single Image
1. Navigate to "📸 Single Image" tab
2. Upload an image (JPG, PNG, JPEG)
3. Adjust filter settings if needed
4. Click "🎨 Cartoonify Image"
5. Download your cartoon image

#### Multiple Images
1. Go to "🖼️ Multiple Images" tab
2. Upload multiple images
3. Enable "Batch Mode" for multiple styles
4. Process all images at once
5. Download individual results or all as ZIP

#### Camera Capture
1. Select "📷 Camera Capture" tab
2. Allow camera access when prompted
3. Position yourself in the camera view
4. Click "Capture Image" button
5. Automatic processing begins
6. View and download results

### 2. Video Processing

1. Navigate to "🎬 Video Upload" tab
2. Upload your video file
3. Adjust cartoon settings
4. Click "🎨 Process Video"
5. Wait for frame-by-frame processing
6. Download the cartoonified video

### 3. Settings & Customization

#### Filter Settings
- **Cartoon Style**: Choose between Comic and AnimeGAN Hayao
- **Filter Intensity**: Control overall effect strength
- **Edge Strength**: Adjust edge detection
- **Saturation**: Enhance or reduce color intensity
- **Contrast**: Adjust image contrast
- **Brightness**: Control image brightness

#### Preset Configurations
- **Comic Style 1**: Balanced comic book effect
- **Hayao Style**: Anime-inspired transformation
- **Custom**: Manual adjustment of all parameters

## 🛠️ Technical Details

### Architecture
- **Frontend**: Streamlit web interface
- **Image Processing**: OpenCV + PIL
- **AI Models**: Groq LLM + AnimeGAN v3 (ONNX)
- **Video Processing**: OpenCV VideoCapture/VideoWriter
- **Camera Integration**: streamlit-webrtc

### Supported Formats
- **Images**: JPG, JPEG, PNG
- **Videos**: MP4, AVI, MOV (most common formats)

### Processing Pipeline
1. **Input Validation**: Format and size checks
2. **Preprocessing**: Color space conversion and resizing
3. **Filter Application**: Cartoon effects using OpenCV or ONNX
4. **Enhancement**: Color and contrast adjustments
5. **Output Generation**: Optimized file creation

## 🔧 Configuration

### Environment Variables
```env
GROQ_API_KEY=your_groq_api_key_here
```

### Optional Model Files
- `AnimeGANv3_Hayao_STYLE_36.onnx` - Place in project root for anime-style processing

## 📊 Performance

### Processing Times (Approximate)
- **Single Image (1080p)**: 2-5 seconds
- **Batch Images (5 images)**: 10-25 seconds
- **Video (30 seconds, 1080p)**: 2-5 minutes
- **Camera Capture**: Real-time processing

### System Requirements
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space for temporary files
- **GPU**: Optional, improves ONNX model performance

## 🎨 Examples

### Before & After
Transform regular photos into:
- **Comic Book Style**: Bold edges, vibrant colors, simplified details
- **Anime Style**: Smooth gradients, soft edges, anime-like appearance
- **Custom Blends**: Artistic combinations of original and cartoon

### Use Cases
- **Social Media Content**: Eye-catching profile pictures and posts
- **Digital Art**: Base for further artistic work
- **Entertainment**: Fun photo transformations
- **Content Creation**: Unique video content for platforms

## 🐛 Troubleshooting

### Common Issues

#### "ONNX Runtime not available"
- Install ONNX Runtime: `pip install onnxruntime`
- Download AnimeGAN model file

#### "Groq API Error"
- Check your API key in `.env` file
- Verify internet connection
- Ensure API key has proper permissions

#### Video Processing Slow
- Reduce video resolution before upload
- Use shorter video clips for testing
- Close other applications to free up resources

#### Camera Not Working
- Grant camera permissions to your browser
- Check if camera is being used by other applications
- Try refreshing the page
- Ensure you're using HTTPS (required for camera access)

#### Video Orientation Issues
- The app automatically maintains correct video orientation
- No manual rotation needed - processed videos preserve original orientation

### Performance Tips
- **Resize large images** before processing for faster results
- **Use batch mode** for multiple images to save time
- **Close unnecessary browser tabs** during video processing
- **Ensure stable internet** for AI analysis features
- **Use modern browsers** for best camera capture experience

## 🔒 Security & Privacy

### Data Handling
- **No Data Storage**: Images and videos are processed locally and not stored on servers
- **Temporary Files**: All temporary files are automatically cleaned up after processing
- **API Security**: Groq API key is securely handled through environment variables

### Camera Privacy
- **Local Processing**: Camera capture is processed locally in your browser
- **No Recording**: Only captures single frames when you click the capture button
- **Permission Based**: Requires explicit camera permission from user

## 🌟 Advanced Features

### Batch Processing
- Process multiple images simultaneously
- Generate different cartoon styles for the same image
- Bulk download as ZIP file

### AI Analysis
- Detailed image descriptions using Groq LLM
- Intelligent scene understanding
- Contextual cartoon style recommendations

### Real-time Camera
- Live camera feed with instant capture
- Automatic image processing after capture
- No need to save files manually

## 🚀 Future Enhancements

### Planned Features
- **Style Transfer**: Additional artistic styles beyond cartoon
- **Video Filters**: Real-time video filtering
- **Batch Video Processing**: Process multiple videos at once
- **Custom Style Training**: Train your own cartoon styles

### Performance Improvements
- **GPU Acceleration**: Enhanced processing speed with GPU support
- **Progressive Processing**: Show processing progress for large files
- **Memory Optimization**: Better handling of large files

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Test thoroughly
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Submit a pull request

### Development Setup
```bash
# Clone the repo
git clone <repository-url>
cd Cartoonify_AI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your API keys

# Run in development mode
streamlit run groq_cartoonify.py
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **AnimeGAN v3**: For anime-style transformation capabilities
- **Groq**: For AI-powered image analysis
- **OpenCV**: For computer vision processing
- **Streamlit**: For the intuitive web interface
- **streamlit-webrtc**: For camera integration

## 📞 Support

For issues, questions, or feature requests:

### Getting Help
1. **Check Documentation**: Review this README and troubleshooting section
2. **Search Issues**: Look through existing GitHub issues
3. **Create New Issue**: Provide detailed description with:
   - Steps to reproduce
   - Expected vs actual behavior
   - System information
   - Error messages (if any)

### Contact Information
- **Email**: sanjay.1991999@gmail.com


## 📈 Changelog

### Version 1.0.0
- ✅ Initial release
- ✅ Single image cartoonification
- ✅ Multiple image batch processing
- ✅ Video processing with correct orientation
- ✅ Camera capture functionality
- ✅ AI-powered image analysis
- ✅ Multiple cartoon styles
- ✅ Customizable filter settings

---

**Happy Cartoonifying!** 🎉
