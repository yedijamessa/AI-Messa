import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pretty_midi
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional
import random
import json
import time
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="AI Music Generator",
    page_icon="🎵",
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
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .generation-card {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

class MusicTransformer(nn.Module):
    """
    Simplified Music Generation Transformer
    """
    def __init__(self, vocab_size=128, d_model=256, nhead=8, num_layers=6, 
                 dim_feedforward=1024, max_length=512):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # Embedding layers
        self.note_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_length, d_model)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, mask=None):
        batch_size, seq_len = x.shape
        
        # Create position indices
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        note_emb = self.note_embedding(x)
        pos_emb = self.position_embedding(positions)
        
        # Combine embeddings
        x = note_emb + pos_emb
        x = self.dropout(x)
        
        # Apply transformer
        if mask is not None:
            x = self.transformer(x, mask)
        else:
            x = self.transformer(x)
        
        # Output projection
        x = self.output_projection(x)
        
        return x

class MusicGenerator:
    """
    Music Generation Engine
    """
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.vocab_size = 128  # MIDI note range
        self.sequence_length = 256
        
        # Musical scales and patterns
        self.scales = {
            'major': [0, 2, 4, 5, 7, 9, 11],
            'minor': [0, 2, 3, 5, 7, 8, 10],
            'pentatonic': [0, 2, 4, 7, 9],
            'blues': [0, 3, 5, 6, 7, 10],
            'dorian': [0, 2, 3, 5, 7, 9, 10],
            'mixolydian': [0, 2, 4, 5, 7, 9, 10]
        }
        
        self.chord_progressions = {
            'pop': [[0, 4, 7], [5, 9, 12], [2, 5, 9], [0, 4, 7]],
            'jazz': [[0, 4, 7, 11], [5, 9, 12, 16], [2, 5, 9, 12], [0, 4, 7, 11]],
            'classical': [[0, 4, 7], [7, 11, 14], [5, 9, 12], [0, 4, 7]],
            'rock': [[0, 4, 7], [5, 9, 12], [7, 11, 14], [0, 4, 7]]
        }
        
    def initialize_model(self):
        """Initialize the transformer model"""
        if self.model is None:
            self.model = MusicTransformer(
                vocab_size=self.vocab_size,
                d_model=256,
                nhead=8,
                num_layers=4,
                dim_feedforward=512,
                max_length=512
            ).to(self.device)
            
            # Initialize with random weights (in a real scenario, you'd load pre-trained weights)
            self.model.eval()
            
    def generate_seed_sequence(self, scale='major', key=60, length=32):
        """Generate a musically coherent seed sequence"""
        scale_notes = [key + note for note in self.scales[scale]]
        seed = []
        
        # Create a simple melodic pattern
        for i in range(length):
            if i % 8 == 0:  # Strong beats
                note = random.choice(scale_notes[:3])  # Tonic chord tones
            elif i % 4 == 0:  # Medium beats
                note = random.choice(scale_notes)
            else:  # Weak beats
                if random.random() < 0.3:  # 30% chance of rest
                    note = 0
                else:
                    note = random.choice(scale_notes)
            seed.append(note)
            
        return seed
    
    def generate_music(self, genre='pop', scale='major', key=60, length=128, 
                      temperature=1.0, top_k=40):
        """Generate music sequence"""
        self.initialize_model()
        
        # Generate seed sequence
        seed = self.generate_seed_sequence(scale, key, 32)
        sequence = seed.copy()
        
        # Generate continuation using the model
        with torch.no_grad():
            for _ in range(length - len(seed)):
                # Prepare input
                input_seq = torch.tensor(sequence[-32:], dtype=torch.long).unsqueeze(0).to(self.device)
                
                # Get model predictions
                outputs = self.model(input_seq)
                logits = outputs[0, -1, :] / temperature
                
                # Apply top-k sampling
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k)
                    logits = torch.full_like(logits, -float('inf'))
                    logits[top_k_indices] = top_k_logits
                
                # Sample next note
                probs = F.softmax(logits, dim=-1)
                next_note = torch.multinomial(probs, 1).item()
                
                # Apply musical constraints
                next_note = self.apply_musical_constraints(next_note, sequence, scale, key)
                sequence.append(next_note)
        
        return sequence
    
    def apply_musical_constraints(self, note, sequence, scale, key):
        """Apply musical theory constraints to generated notes"""
        scale_notes = [key + n for n in self.scales[scale]]
        
        # Ensure note is in scale (with some flexibility)
        if note > 0 and note not in scale_notes:
            # Find closest scale note
            closest = min(scale_notes, key=lambda x: abs(x - note))
            if abs(closest - note) <= 2:  # Within 2 semitones
                note = closest
        
        # Avoid large jumps
        if len(sequence) > 0 and sequence[-1] > 0 and note > 0:
            if abs(note - sequence[-1]) > 12:  # More than an octave
                note = sequence[-1] + random.choice([-1, 0, 1, 2, -2])
        
        # Ensure note is in valid MIDI range
        note = max(0, min(127, note))
        
        return note
    
    def sequence_to_midi(self, sequence, tempo=120):
        """Convert note sequence to MIDI"""
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        instrument = pretty_midi.Instrument(program=0)  # Piano
        
        time = 0
        note_duration = 60 / tempo / 4  # 16th notes
        
        for note_pitch in sequence:
            if note_pitch > 0:  # Skip rests
                note = pretty_midi.Note(
                    velocity=80,
                    pitch=int(note_pitch),
                    start=time,
                    end=time + note_duration
                )
                instrument.notes.append(note)
            time += note_duration
        
        midi.instruments.append(instrument)
        return midi

@st.cache_resource
def get_music_generator():
    """Cache the music generator instance"""
    return MusicGenerator()

def create_piano_roll_visualization(sequence, title="Generated Music"):
    """Create a piano roll visualization"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Filter out rests (0 values)
    notes = [(i, note) for i, note in enumerate(sequence) if note > 0]
    
    if notes:
        times, pitches = zip(*notes)
        
        # Create scatter plot
        scatter = ax.scatter(times, pitches, c=pitches, cmap='viridis', 
                           alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='MIDI Pitch')
        
        # Customize plot
        ax.set_xlabel('Time Step')
        ax.set_ylabel('MIDI Pitch')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Add note names for reference
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        ax.set_yticks(range(60, 84, 12))
        ax.set_yticklabels([f"{note_names[i%12]}{i//12}" for i in range(60, 84, 12)])
    
    plt.tight_layout()
    return fig

def download_midi(midi_data, filename):
    """Create download link for MIDI file"""
    buffer = io.BytesIO()
    midi_data.write(buffer)
    buffer.seek(0)
    
    b64 = base64.b64encode(buffer.read()).decode()
    href = f'<a href="data:audio/midi;base64,{b64}" download="{filename}">Download MIDI File</a>'
    return href

def main():
    # Header
    st.markdown('<h1 class="main-header">🎵 AI Music Generator</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Generate beautiful music using transformer-based AI technology
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.header("🎛️ Music Parameters")
    
    # Genre selection
    genre = st.sidebar.selectbox(
        "Genre",
        ["pop", "jazz", "classical", "rock"],
        help="Choose the musical genre for generation"
    )
    
    # Scale selection
    scale = st.sidebar.selectbox(
        "Musical Scale",
        ["major", "minor", "pentatonic", "blues", "dorian", "mixolydian"],
        help="Select the musical scale for the generation"
    )
    
    # Key selection
    key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    key_name = st.sidebar.selectbox("Key", key_names, index=0)
    key = 60 + key_names.index(key_name)  # Convert to MIDI note number
    
    # Length and tempo
    length = st.sidebar.slider("Sequence Length", 32, 256, 128, 
                              help="Number of notes to generate")
    tempo = st.sidebar.slider("Tempo (BPM)", 60, 180, 120)
    
    # Advanced parameters
    st.sidebar.subheader("🔧 Advanced Settings")
    temperature = st.sidebar.slider("Creativity (Temperature)", 0.1, 2.0, 1.0, 0.1,
                                   help="Higher values = more creative/random")
    top_k = st.sidebar.slider("Top-K Sampling", 1, 100, 40,
                             help="Number of top predictions to sample from")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎼 Generated Music")
        
        # Generation button
        if st.button("🎵 Generate Music", key="generate_main"):
            with st.spinner("Generating music... 🎹"):
                generator = get_music_generator()
                
                # Record generation time
                start_time = time.time()
                sequence = generator.generate_music(
                    genre=genre,
                    scale=scale,
                    key=key,
                    length=length,
                    temperature=temperature,
                    top_k=top_k
                )
                generation_time = time.time() - start_time
                
                # Store in session state
                st.session_state.generated_sequence = sequence
                st.session_state.generation_params = {
                    'genre': genre,
                    'scale': scale,
                    'key': key_name,
                    'length': length,
                    'tempo': tempo,
                    'generation_time': generation_time
                }
        
        # Display generated music if available
        if hasattr(st.session_state, 'generated_sequence'):
            sequence = st.session_state.generated_sequence
            params = st.session_state.generation_params
            
            # Success message
            st.success(f"✅ Generated {len(sequence)} notes in {params['generation_time']:.2f} seconds!")
            
            # Create visualization
            fig = create_piano_roll_visualization(sequence, 
                f"Generated Music - {params['genre'].title()} in {params['key']} {params['scale'].title()}")
            st.pyplot(fig)
            
            # Convert to MIDI
            generator = get_music_generator()
            midi_data = generator.sequence_to_midi(sequence, params['tempo'])
            
            # Download link
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generated_music_{params['genre']}_{params['key']}_{params['scale']}_{timestamp}.mid"
            
            st.markdown(
                download_midi(midi_data, filename),
                unsafe_allow_html=True
            )
            
            # Sequence analysis
            st.subheader("📊 Sequence Analysis")
            
            # Filter out rests
            notes_only = [n for n in sequence if n > 0]
            
            if notes_only:
                col_a, col_b, col_c, col_d = st.columns(4)
                
                with col_a:
                    st.metric("Total Notes", len(notes_only))
                
                with col_b:
                    st.metric("Note Range", f"{max(notes_only) - min(notes_only)}")
                
                with col_c:
                    rest_percentage = ((len(sequence) - len(notes_only)) / len(sequence)) * 100
                    st.metric("Rest %", f"{rest_percentage:.1f}%")
                
                with col_d:
                    avg_pitch = sum(notes_only) / len(notes_only)
                    st.metric("Avg Pitch", f"{avg_pitch:.1f}")
    
    with col2:
        st.subheader("🎯 Quick Examples")
        
        # Example buttons
        examples = [
            {"name": "Peaceful Piano", "genre": "classical", "scale": "major", "key": "C", "temp": 0.8},
            {"name": "Jazz Improv", "genre": "jazz", "scale": "mixolydian", "key": "F", "temp": 1.2},
            {"name": "Blues Riff", "genre": "rock", "scale": "blues", "key": "E", "temp": 1.0},
            {"name": "Folk Melody", "genre": "pop", "scale": "pentatonic", "key": "G", "temp": 0.7}
        ]
        
        for example in examples:
            if st.button(f"🎵 {example['name']}", key=f"example_{example['name']}"):
                # Set parameters
                st.sidebar.selectbox("Genre", ["pop", "jazz", "classical", "rock"], 
                                   index=["pop", "jazz", "classical", "rock"].index(example['genre']))
                
                # Generate with example parameters
                with st.spinner(f"Generating {example['name']}..."):
                    generator = get_music_generator()
                    key_midi = 60 + key_names.index(example['key'])
                    
                    sequence = generator.generate_music(
                        genre=example['genre'],
                        scale=example['scale'],
                        key=key_midi,
                        length=96,
                        temperature=example['temp']
                    )
                    
                    st.session_state.generated_sequence = sequence
                    st.session_state.generation_params = {
                        'genre': example['genre'],
                        'scale': example['scale'],
                        'key': example['key'],
                        'length': 96,
                        'tempo': 120,
                        'generation_time': 0.5
                    }
                    st.rerun()
        
        # Info section
        st.subheader("ℹ️ About")
        st.info("""
        This AI music generator uses a Transformer neural network to create musical sequences. 
        
        **Features:**
        - Multiple musical scales and genres
        - Adjustable creativity levels
        - Real-time visualization
        - MIDI file export
        - Musical theory constraints
        
        **Tips:**
        - Lower temperature = more predictable
        - Higher temperature = more creative
        - Try different scales for various moods
        """)
        
        # Model info
        st.subheader("🤖 Model Info")
        generator = get_music_generator()
        
        if generator.model is not None:
            total_params = sum(p.numel() for p in generator.model.parameters())
            st.text(f"Parameters: {total_params:,}")
            st.text(f"Device: {generator.device}")
            st.text(f"Vocab Size: {generator.vocab_size}")
        
        # Statistics
        if hasattr(st.session_state, 'generated_sequence'):
            st.subheader("📈 Generation Stats")
            params = st.session_state.generation_params
            
            st.text(f"Genre: {params['genre'].title()}")
            st.text(f"Key: {params['key']} {params['scale'].title()}")
            st.text(f"Tempo: {params['tempo']} BPM")
            st.text(f"Length: {params['length']} notes")
            st.text(f"Time: {params['generation_time']:.2f}s")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🎵 AI Music Generator | Built with Streamlit & PyTorch</p>
        <p>Combine creativity with artificial intelligence to compose unique musical pieces</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()