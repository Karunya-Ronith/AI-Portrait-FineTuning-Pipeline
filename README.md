# AI Portrait FineTuning Pipeline

A complete end-to-end project for fine-tuning Stable Diffusion 1.5 with LoRA to generate realistic Leonardo DiCaprio portraits from text prompts. This project includes data preparation, model training, Hugging Face deployment, and a beautiful Streamlit web interface.

## Model on Hugging Face

**Try the model online**: [Leonardo DiCaprio LoRA Model](https://huggingface.co/KarunyaRonith29/leonardo_dicaprio_lora)

The trained LoRA model is available on Hugging Face for easy integration and testing.

## Project Overview

This project demonstrates how to:
- Fine-tune Stable Diffusion 1.5 using LoRA (Low-Rank Adaptation)
- Train on a custom dataset of 100 Leonardo DiCaprio images
- Deploy the model to Hugging Face
- Create a professional web interface for image generation
- Optimize for limited hardware (RTX 3050 with 6GB VRAM)

## Model Performance

- **Training Time**: ~45 minutes on RTX 3050
- **Model Size**: 12.8 MB (LoRA weights only)
- **Generation Time**: ~50 seconds per image
- **Image Quality**: High-quality 512x512 portraits
- **Hardware Used**: NVIDIA RTX 3050 (6GB VRAM, 16GB RAM)

## Project Structure

```
AI-Portrait-FineTuning-Pipeline/
├── Leonardo DiCaprio/           # Original training images (100 images)
├── dataset/                     # Processed training dataset
│   ├── images/                  # Resized and normalized images
│   └── metadata.jsonl          # Training captions
├── models/                      # Trained model outputs
│   ├── lora/                   # LoRA weights
│   └── pipeline/               # Full pipeline
├── outputs/                     # Generated sample images
├── samples/                     # Sample images for documentation
├── demo/                        # Streamlit web interface
│   ├── app.py                  # Main web application
│   ├── requirements.txt        # Web app dependencies
│   ├── README.md              # Demo documentation
│   └── run_demo.py            # Launcher script
├── leonardo_dicaprio_lora/      # Hugging Face model package
├── data_preparation.py         # Dataset preprocessing script
├── train_lora_final.py         # Main training script
├── generate_images_fixed.py    # Image generation script
├── test_training.py            # System testing script
├── install_dependencies.py     # Dependency installer
└── requirements.txt            # Main project dependencies
```

## Quick Start

### Prerequisites

Before running this project, you need to set up the required data and models. Since large files are gitignored to keep the repository size manageable, follow these steps:

**Note**: The following files/directories are intentionally excluded from the repository:
- `Leonardo DiCaprio/` - Original training images (100+ MB)
- `dataset/images/` - Processed training images
- `models/pipeline/` - Full trained model (1.9 GB)
- `models/lora/` - LoRA weights
- `outputs/` - Generated images
- `leonardo_dicaprio_lora/` - Hugging Face model package

These files will be created when you run the respective scripts.

#### 1. Download the Dataset

The training dataset is not included in the repository due to size constraints. You need to:

1. **Download the Celebrity Face Image Dataset** from Kaggle:
   - Visit: https://www.kaggle.com/datasets/vishesh1412/celebrity-face-image-dataset
   - Download the dataset
   - Extract and place Leonardo DiCaprio images in a folder named `Leonardo DiCaprio/` in the project root

2. **Alternative: Use your own images**
   - Create a folder named `Leonardo DiCaprio/`
   - Add 50-100 high-quality Leonardo DiCaprio face images
   - Images should be in JPG/PNG format

#### 2. Download Pre-trained Models

The base Stable Diffusion model will be downloaded automatically when you run the scripts, but you can also download it manually:

```bash
# This will be done automatically, but you can pre-download
python -c "from diffusers import StableDiffusionPipeline; StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5')"
```

#### 3. Clone and Setup

```bash
git clone https://github.com/Karunya-Ronith/AI-Portrait-FineTuning-Pipeline
cd AI-Portrait-FineTuning-Pipeline
conda activate main  # or your preferred environment
pip install -r requirements.txt
```

### 4. Data Preparation

Run the data preparation script to process your images:

```bash
python data_preparation.py
```

This will:
- Resize images to 512x512 pixels
- Create the `dataset/` folder with processed images
- Generate `metadata.jsonl` with training captions

### 5. Training (Optional)

If you want to train your own model:

```bash
python train_lora_final.py
```

**Note**: Training requires:
- NVIDIA GPU with 6GB+ VRAM
- ~45 minutes on RTX 3050
- The processed dataset from step 4

### 6. Run the Web Demo

```bash
cd demo
pip install -r requirements.txt
streamlit run app.py
```

Open your browser at `http://localhost:8501`

### 7. Generate Images from Command Line

```bash
python generate_images_fixed.py --prompt "leonardo dicaprio eating"
```

## Detailed Training Process

### Step 1: Data Preparation

**Script**: `data_preparation.py`

**Dataset Source**: 
- **Original Dataset**: [Celebrity Face Image Dataset](https://www.kaggle.com/datasets/vishesh1412/celebrity-face-image-dataset) on Kaggle
- **Citation**: This project uses images from the Celebrity Face Image Dataset by Vishesh1412
- **License**: Please refer to the original dataset license on Kaggle

**Process**:
1. **Image Loading**: Load 100 Leonardo DiCaprio face images from `Leonardo DiCaprio/` directory
2. **Resizing**: Resize all images to 512x512 pixels (Stable Diffusion standard)
3. **Aspect Ratio Preservation**: Use padding to maintain aspect ratio without distortion
4. **Normalization**: Convert to RGB format and apply proper normalization
5. **Metadata Generation**: Create `metadata.jsonl` with simple captions ("leonardo dicaprio")

**Key Features**:
- Automatic aspect ratio preservation with black padding
- High-quality LANCZOS resampling
- JPEG compression with 95% quality
- Consistent captioning for all images

### Step 2: Model Training

**Script**: `train_lora_final.py`

**Training Configuration**:
```python
# LoRA Parameters
lora_rank = 16
lora_alpha = 16
target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
lora_dropout = 0.1

# Training Parameters
learning_rate = 1e-4
batch_size = 1
gradient_accumulation_steps = 4
max_train_steps = 100
mixed_precision = "fp16"
```

**Training Process**:
1. **Base Model Loading**: Load Stable Diffusion 1.5 components (UNet, VAE, Text Encoder, Tokenizer)
2. **Component Freezing**: Freeze VAE and Text Encoder (only train UNet)
3. **LoRA Application**: Apply LoRA layers to UNet attention modules using PEFT
4. **Data Loading**: Load processed images and captions
5. **Training Loop**: 
   - Convert images to latent space using VAE
   - Add noise at random timesteps
   - Predict noise using UNet
   - Calculate MSE loss
   - Backpropagate and update LoRA weights
6. **Model Saving**: Save LoRA weights and full pipeline

**Memory Optimizations**:
- Mixed precision training (FP16)
- Gradient accumulation (effective batch size = 4)
- Manual batching to avoid DataLoader overhead
- Gradient checkpointing support

### Step 3: Model Deployment

**Hugging Face Upload**:
1. **Model Packaging**: Create `leonardo_dicaprio_lora/` directory with:
   - `adapter_config.json` (LoRA configuration)
   - `adapter_model.safetensors` (LoRA weights)
   - `README.md` (model documentation)
   - `model_card.md` (metadata)
   - `requirements.txt` (dependencies)
   - `example_usage.py` (usage examples)
   - `examples/` (sample generated images)

2. **Repository Creation**: Create Hugging Face model repository
3. **File Upload**: Upload all model files
4. **Documentation**: Update README with usage instructions

**Model URL**: [KarunyaRonith29/leonardo_dicaprio_lora](https://huggingface.co/KarunyaRonith29/leonardo_dicaprio_lora)

### Step 4: Web Interface Development

**Demo Directory**: `demo/`

**Features**:
- **Beautiful UI**: Modern Streamlit interface with custom CSS
- **Real-time Generation**: Live image generation with progress indicators
- **Parameter Control**: Adjustable inference steps, guidance scale, and random seed
- **Example Prompts**: Quick-access buttons for popular prompts
- **Download Functionality**: Save generated images directly from web
- **Session Management**: Remember settings and generated images
- **Responsive Design**: Works on desktop and mobile

**Technical Implementation**:
- Model caching with `@st.cache_resource`
- Automatic GPU detection and utilization
- Error handling and user feedback
- Professional styling with custom CSS
- File download with proper naming

## How to Use the Project

### Option 1: Web Interface (Recommended)

1. **Navigate to Demo Directory**:
   ```bash
   cd demo
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Web App**:
   ```bash
   streamlit run app.py
   ```

4. **Open Browser**: Go to `http://localhost:8501`

5. **Generate Images**:
   - Enter a prompt (e.g., "leonardo dicaprio eating")
   - Adjust parameters in the sidebar
   - Click "Generate Portrait"
   - Download the generated image

### Option 2: Command Line

1. **Generate Single Image**:
   ```bash
   python generate_images_fixed.py --prompt "leonardo dicaprio smiling"
   ```

2. **Generate Multiple Images**:
   ```bash
   python generate_images_fixed.py --prompt "leonardo dicaprio in a suit" --num_images 3
   ```

3. **Custom Parameters**:
   ```bash
   python generate_images_fixed.py \
     --prompt "leonardo dicaprio portrait" \
     --num_inference_steps 75 \
     --guidance_scale 8.5 \
     --seed 42
   ```

### Option 3: Programmatic Usage

```python
from peft import PeftModel
from diffusers import UNet2DConditionModel, StableDiffusionPipeline

# Load base model
unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")

# Load LoRA weights
unet = PeftModel.from_pretrained(unet, "KarunyaRonith29/leonardo_dicaprio_lora")

# Create pipeline and generate
pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", unet=unet)
image = pipeline("leonardo dicaprio eating").images[0]
image.save("dicaprio_eating.png")
```

## Example Prompts

- `leonardo dicaprio eating`
- `leonardo dicaprio smiling`
- `leonardo dicaprio in a suit`
- `leonardo dicaprio portrait`
- `leonardo dicaprio on a red carpet`
- `leonardo dicaprio with sunglasses`
- `leonardo dicaprio in a movie scene`
- `leonardo dicaprio at an award ceremony`

## Technical Details

### LoRA Implementation
- **Rank**: 16 (controls expressiveness)
- **Alpha**: 16 (scaling factor)
- **Target Modules**: Attention layers (to_q, to_k, to_v, to_out.0)
- **Dropout**: 0.1 (regularization)

### Training Optimizations
- **Mixed Precision**: FP16 for memory efficiency
- **Gradient Accumulation**: Simulate larger batch sizes
- **Manual Batching**: Avoid DataLoader overhead
- **Memory Management**: Optimized for 6GB VRAM

### Model Architecture
- **Base Model**: Stable Diffusion 1.5
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Training Data**: 100 high-quality Leonardo DiCaprio images
- **Image Resolution**: 512x512 pixels
- **Caption Strategy**: Simple, consistent captions

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size or gradient accumulation steps
   - Use gradient checkpointing
   - Close other GPU applications

2. **Model Loading Errors**:
   - Ensure stable internet connection
   - Check Hugging Face model availability
   - Verify all dependencies are installed

3. **Generation Quality Issues**:
   - Increase inference steps (50-100)
   - Adjust guidance scale (7.5-10.0)
   - Try different random seeds

4. **Web Interface Issues**:
   - Check if port 8501 is available
   - Ensure Streamlit is installed
   - Verify model loading in sidebar

### System Requirements

**Minimum**:
- GPU: NVIDIA GPU with 4GB+ VRAM
- RAM: 8GB
- Storage: 5GB free space
- Python: 3.8+

**Recommended**:
- GPU: NVIDIA RTX 3050 or better
- RAM: 16GB
- Storage: 10GB free space
- Python: 3.9+

## Performance Metrics

- **Training Loss**: Converges within 100 steps
- **Generation Quality**: High-fidelity Leonardo DiCaprio portraits
- **Consistency**: Good facial feature preservation
- **Speed**: ~50 seconds per image on RTX 3050
- **Memory Usage**: ~4GB VRAM during training, ~2GB during inference

## Use Cases

- **Creative Projects**: Generate unique Leonardo DiCaprio portraits
- **Educational**: Learn about AI fine-tuning and LoRA
- **Entertainment**: Create fun and interesting images
- **Research**: Experiment with different prompts and parameters
- **Portfolio**: Demonstrate AI/ML skills

## License and Ethics

- **License**: MIT License
- **Educational Use**: This project is for educational and research purposes
- **Copyright**: Please respect copyright and use responsibly
- **Ethics**: Do not use for creating misleading or harmful content

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

- **Issues**: Create an issue on GitHub
- **Questions**: Check the documentation or create a discussion
- **Model**: Visit the [Hugging Face model page](https://huggingface.co/Karunya-Ronith/leonardo_dicaprio_lora)

## Acknowledgments

- **Base Model**: [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- **LoRA Implementation**: PEFT library
- **Web Framework**: Streamlit
- **Training Framework**: PyTorch and Diffusers
- **Dataset**: [Celebrity Face Image Dataset](https://www.kaggle.com/datasets/vishesh1412/celebrity-face-image-dataset) by Vishesh1412 on Kaggle

## Citation

If you use this project in your research or work, please cite:

```bibtex
@misc{leonardo_dicaprio_lora_2024,
  title={AI Portrait FineTuning Pipeline: Leonardo DiCaprio LoRA Model},
  author={Karunya-Ronith},
  year={2024},
  url={https://github.com/Karunya-Ronith/AI-Portrait-FineTuning-Pipeline},
  note={Fine-tuned Stable Diffusion 1.5 with LoRA for Leonardo DiCaprio portrait generation}
}
```

**Dataset Citation**:
```bibtex
@dataset{celebrity_face_dataset,
  title={Celebrity Face Image Dataset},
  author={Vishesh1412},
  year={2024},
  url={https://www.kaggle.com/datasets/vishesh1412/celebrity-face-image-dataset}
}
```

---

**Enjoy generating Leonardo DiCaprio portraits! This project demonstrates the power of fine-tuning large language models for specific tasks while maintaining efficiency and accessibility.** 