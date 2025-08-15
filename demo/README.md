# 🎭 Leonardo DiCaprio AI Portrait Generator - Web Demo

A beautiful Streamlit web interface for generating Leonardo DiCaprio portraits using the fine-tuned LoRA model.

## 🌟 Features

- **Beautiful Web Interface**: Modern, responsive design with custom styling
- **Real-time Generation**: Generate Leonardo DiCaprio portraits from text prompts
- **Adjustable Parameters**: Control inference steps, guidance scale, and random seed
- **Example Prompts**: Quick access to popular prompts
- **Download Images**: Save generated portraits directly from the web interface
- **Model Information**: Display model details and generation statistics

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Web App
```bash
streamlit run app.py
```

### 3. Open Your Browser
The app will automatically open at `http://localhost:8501`

## 🎨 How to Use

1. **Enter a Prompt**: Describe the Leonardo DiCaprio portrait you want to generate
2. **Adjust Settings**: Use the sidebar to modify generation parameters
3. **Click Generate**: Watch as your portrait is created in real-time
4. **Download**: Save your generated image with one click

## 💡 Example Prompts

- `leonardo dicaprio eating`
- `leonardo dicaprio smiling`
- `leonardo dicaprio in a suit`
- `leonardo dicaprio portrait`
- `leonardo dicaprio on a red carpet`
- `leonardo dicaprio with sunglasses`
- `leonardo dicaprio in a movie scene`
- `leonardo dicaprio at an award ceremony`

## ⚙️ Generation Parameters

- **Inference Steps**: 20-100 (higher = better quality, slower)
- **Guidance Scale**: 1.0-20.0 (higher = more prompt adherence)
- **Random Seed**: Set for reproducible results, or leave empty for random

## 🖥️ System Requirements

- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **RAM**: 8GB+ (16GB recommended)
- **Storage**: 5GB+ free space for model downloads

## 🔧 Troubleshooting

### Model Loading Issues
- Ensure you have a stable internet connection (model downloads from Hugging Face)
- Check that all dependencies are installed correctly
- Verify CUDA is available if using GPU

### Generation Issues
- Try reducing inference steps for faster generation
- Adjust guidance scale if results are too strict or too loose
- Use different seeds for varied results

## 📱 Interface Features

- **Responsive Design**: Works on desktop and mobile
- **Real-time Feedback**: Progress indicators and generation time
- **Session Management**: Remembers your settings and generated images
- **Professional Styling**: Custom CSS for a polished look

## 🔗 Model Information

- **Base Model**: Stable Diffusion 1.5
- **Fine-tuning**: LoRA (Low-Rank Adaptation)
- **Training Data**: 100 Leonardo DiCaprio images
- **Model Size**: 12.8 MB (LoRA weights)
- **Repository**: [KarunyaRonith29/leonardo_dicaprio_lora](https://huggingface.co/KarunyaRonith29/leonardo_dicaprio_lora)

## 🎯 Use Cases

- **Creative Projects**: Generate unique Leonardo DiCaprio portraits
- **Educational**: Learn about AI image generation
- **Entertainment**: Create fun and interesting images
- **Research**: Experiment with different prompts and parameters

## 📄 License

This demo is for educational and research purposes. Please respect copyright and use responsibly.

---

**Enjoy generating Leonardo DiCaprio portraits! 🎭✨** 