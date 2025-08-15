import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
import time
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="Leonardo DiCaprio AI Portrait Generator",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .generated-image {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .prompt-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the Leonardo DiCaprio LoRA model"""
    with st.spinner("Loading Leonardo DiCaprio AI model..."):
        try:
            # Load base model components
            tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
            text_encoder = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="text_encoder")
            vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
            unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")
            scheduler = DDPMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
            
            # Load LoRA weights from Hugging Face
            unet = PeftModel.from_pretrained(unet, "KarunyaRonith29/leonardo_dicaprio_lora")
            
            # Create pipeline
            pipeline = StableDiffusionPipeline(
                unet=unet,
                text_encoder=text_encoder,
                vae=vae,
                tokenizer=tokenizer,
                scheduler=scheduler,
                feature_extractor=None,
                safety_checker=None,
                requires_safety_checker=False,
            )
            
            # Move to GPU if available
            if torch.cuda.is_available():
                pipeline = pipeline.to("cuda")
                st.success("✅ Model loaded on GPU")
            else:
                st.warning("⚠️ GPU not available, using CPU (will be slower)")
            
            return pipeline
            
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            return None

def generate_image(pipeline, prompt, num_inference_steps=50, guidance_scale=7.5, seed=None):
    """Generate image using the model"""
    try:
        if seed is not None:
            generator = torch.Generator(device=pipeline.device).manual_seed(seed)
        else:
            generator = None
        
        # Generate image
        image = pipeline(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]
        
        return image
        
    except Exception as e:
        st.error(f"❌ Error generating image: {e}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🎭 Leonardo DiCaprio AI Portrait Generator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Generate realistic Leonardo DiCaprio portraits from text descriptions</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Generation Settings")
        
        # Model loading
        if 'pipeline' not in st.session_state:
            st.session_state.pipeline = load_model()
        
        if st.session_state.pipeline is None:
            st.error("Model failed to load. Please refresh the page.")
            return
        
        # Generation parameters
        st.subheader("Parameters")
        num_inference_steps = st.slider("Inference Steps", 20, 100, 50, help="More steps = higher quality but slower")
        guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 7.5, 0.5, help="Higher values = more prompt adherence")
        seed = st.number_input("Random Seed", value=None, help="Leave empty for random, or set for reproducible results")
        
        # Example prompts
        st.subheader("💡 Example Prompts")
        example_prompts = [
            "leonardo dicaprio eating",
            "leonardo dicaprio smiling",
            "leonardo dicaprio in a suit",
            "leonardo dicaprio portrait",
            "leonardo dicaprio on a red carpet",
            "leonardo dicaprio with sunglasses",
            "leonardo dicaprio in a movie scene",
            "leonardo dicaprio at an award ceremony"
        ]
        
        for prompt in example_prompts:
            if st.button(prompt, key=prompt):
                st.session_state.prompt = prompt
                st.rerun()
        
        # Model info
        st.subheader("ℹ️ Model Information")
        st.markdown("""
        - **Base Model**: Stable Diffusion 1.5
        - **Fine-tuning**: LoRA (Low-Rank Adaptation)
        - **Training Data**: 100 Leonardo DiCaprio images
        - **Model Size**: 12.8 MB (LoRA weights)
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Enter Your Prompt")
        
        # Text input
        prompt = st.text_area(
            "Describe the Leonardo DiCaprio portrait you want to generate:",
            value=st.session_state.get('prompt', 'leonardo dicaprio eating'),
            height=100,
            placeholder="e.g., leonardo dicaprio eating, leonardo dicaprio smiling, leonardo dicaprio in a suit..."
        )
        
        # Generate button
        if st.button("🎨 Generate Portrait", type="primary", use_container_width=True):
            if prompt.strip():
                with st.spinner("🎭 Generating your Leonardo DiCaprio portrait..."):
                    start_time = time.time()
                    
                    # Generate image
                    image = generate_image(
                        st.session_state.pipeline,
                        prompt,
                        num_inference_steps,
                        guidance_scale,
                        seed
                    )
                    
                    if image:
                        generation_time = time.time() - start_time
                        st.session_state.generated_image = image
                        st.session_state.generation_time = generation_time
                        st.session_state.current_prompt = prompt
                        st.rerun()
            else:
                st.warning("Please enter a prompt!")
    
    with col2:
        st.header("🖼️ Generated Portrait")
        
        if 'generated_image' in st.session_state:
            # Display the generated image
            st.image(
                st.session_state.generated_image,
                caption=f"Generated in {st.session_state.generation_time:.1f} seconds",
                use_column_width=True,
                output_format="PNG"
            )
            
            # Download button
            img_buffer = io.BytesIO()
            st.session_state.generated_image.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            st.download_button(
                label="📥 Download Image",
                data=img_buffer.getvalue(),
                file_name=f"dicaprio_{st.session_state.current_prompt.replace(' ', '_')}.png",
                mime="image/png",
                use_container_width=True
            )
            
            # Generation info
            st.markdown(f"""
            <div class="info-box">
                <strong>Generation Details:</strong><br>
                • Prompt: "{st.session_state.current_prompt}"<br>
                • Time: {st.session_state.generation_time:.1f} seconds<br>
                • Steps: {num_inference_steps}<br>
                • Guidance: {guidance_scale}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("👆 Enter a prompt and click 'Generate Portrait' to create your Leonardo DiCaprio image!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🎭 Leonardo DiCaprio AI Portrait Generator | Powered by Stable Diffusion + LoRA</p>
        <p>Model: <a href="https://huggingface.co/KarunyaRonith29/leonardo_dicaprio_lora" target="_blank">KarunyaRonith29/leonardo_dicaprio_lora</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 