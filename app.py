import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
import io
import base64
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

app = Flask(__name__)


# FRAKTAL Kompresi
class FractalImageCompression:
    def __init__(self, range_size=8, domain_size=16, step=8):
        self.range_size = range_size
        self.domain_size = domain_size
        self.step = step
        self.transforms = self._generate_transformations()
        
    def _generate_transformations(self):
        transforms = []
        for angle in [0, 90, 180, 270]:
            transforms.append(('rotate', angle))
        for angle in [0, 90, 180, 270]:
            transforms.append(('flip_rotate', angle))
        return transforms
    
    def downscale_block(self, block, target_size):
        img = Image.fromarray(block)
        return np.array(img.resize((target_size, target_size), Image.Resampling.BOX))
    
    def apply_transformation(self, block, transform):
        transform_type, angle = transform
        k = angle // 90
        if transform_type == 'rotate':
            return np.rot90(block, k)
        else: 
            flipped = np.fliplr(block)
            return np.rot90(flipped, k)
    
    def find_best_match(self, range_block, domain_blocks):
        best_error = float('inf')
        best_match = None
        range_flat = range_block.flatten().astype(np.float64)
        
        for domain_idx, domain_block in enumerate(domain_blocks):
            domain_scaled = self.downscale_block(domain_block, self.range_size)
            for transform in self.transforms:
                domain_transformed = self.apply_transformation(domain_scaled, transform)
                domain_flat = domain_transformed.flatten().astype(np.float64)
                
                n = len(domain_flat)
                sum_d = np.sum(domain_flat)
                sum_r = np.sum(range_flat)
                sum_dr = np.sum(domain_flat * range_flat)
                sum_d2 = np.sum(domain_flat**2)
                
                det = (n * sum_d2 - sum_d**2)
                if det == 0: s = 0.0
                else: s = (n * sum_dr - sum_d * sum_r) / det
                
                o = (sum_r - s * sum_d) / n
                if s < -1.0: s = -1.0
                if s > 1.0: s = 1.0
                
                approximation = s * domain_flat + o
                error = np.mean((range_flat - approximation)**2)
                
                if error < best_error:
                    best_error = error
                    best_match = {
                        'domain_idx': domain_idx, 'transform': transform,
                        'scale': s, 'offset': o, 'error': error,
                        'visual_block': approximation.reshape(self.range_size, self.range_size)
                    }
        return best_match
    
    def compress(self, image):
        h, w = image.shape
        fractal_codes = []
        domain_blocks = []
        domain_positions = []
        
        # 1. Buat Library Domain
        for i in range(0, h - self.domain_size + 1, self.step):
            for j in range(0, w - self.domain_size + 1, self.step):
                domain_block = image[i:i+self.domain_size, j:j+self.domain_size]
                domain_blocks.append(domain_block)
                domain_positions.append((i, j))
        
        first_block_debug = None
        range_count = 0
        
        # 2. Pencarian Range
        for i in range(0, h - self.range_size + 1, self.step):
            for j in range(0, w - self.range_size + 1, self.step):
                range_block = image[i:i+self.range_size, j:j+self.range_size]
                best_match = self.find_best_match(range_block, domain_blocks)
                
                # Tangkap data debug blok pertama untuk visualisasi
                if range_count == 0:
                    first_block_debug = {
                        'range_block': range_block,
                        'best_domain_raw': domain_blocks[best_match['domain_idx']],
                        'best_domain_transformed': best_match['visual_block'],
                        'info': best_match, 'pos': (i, j)
                    }
                
                fractal_codes.append({
                    'range_pos': (i, j),
                    'domain_idx': best_match['domain_idx'],
                    'domain_pos': domain_positions[best_match['domain_idx']],
                    'transform': best_match['transform'],
                    'scale': best_match['scale'], 'offset': best_match['offset']
                })
                range_count += 1
                
        return fractal_codes, first_block_debug

    def decompress(self, fractal_codes, image_shape, iterations=8):
        h, w = image_shape
        reconstructed = np.random.randint(0, 256, (h, w)).astype(np.float64)
        history = [reconstructed.copy().astype(np.uint8)]
        
        for _ in range(iterations):
            new_image = np.zeros_like(reconstructed)
            for code in fractal_codes:
                di, dj = code['domain_pos']
                domain = reconstructed[di:di+self.domain_size, dj:dj+self.domain_size]
                domain_scaled = self.downscale_block(domain, self.range_size)
                domain_transformed = self.apply_transformation(domain_scaled, code['transform'])
                approximation = code['scale'] * domain_transformed + code['offset']
                ri, rj = code['range_pos']
                new_image[ri:ri+self.range_size, rj:rj+self.range_size] = approximation
            
            reconstructed = new_image
            history.append(np.clip(reconstructed, 0, 255).astype(np.uint8))
            
        return np.clip(reconstructed, 0, 255).astype(np.uint8), history

# VISUALIZATION
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def generate_partition_grid(image, range_size):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image, cmap='gray')
    ax.set_title(f"Grid Partisi ({range_size}x{range_size})")
    # Gambar sebagian grid agar tidak penuh
    step = max(range_size, image.shape[0] // 10)
    for i in range(0, image.shape[0], step):
        ax.axhline(i, color='red', alpha=0.3, linewidth=1)
    for j in range(0, image.shape[1], step):
        ax.axvline(j, color='red', alpha=0.3, linewidth=1)
    rect = patches.Rectangle((0,0), range_size, range_size, linewidth=2, edgecolor='yellow', facecolor='none')
    ax.add_patch(rect)
    ax.axis('off')
    return fig_to_base64(fig)

def generate_block_debug(debug_data):
    if not debug_data: return None
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(debug_data['range_block'], cmap='gray', vmin=0, vmax=255); axes[0].set_title("Target (Range)")
    axes[1].imshow(debug_data['best_domain_raw'], cmap='gray', vmin=0, vmax=255); axes[1].set_title("Sumber (Domain)")
    axes[2].imshow(debug_data['best_domain_transformed'], cmap='gray', vmin=0, vmax=255); axes[2].set_title("Hasil Prediksi")
    for ax in axes: ax.axis('off')
    return fig_to_base64(fig)

def generate_evolution(history):
    indices = [0, 1, 2, 4, 8]
    valid_indices = [i for i in indices if i < len(history)]
    fig, axes = plt.subplots(1, len(valid_indices), figsize=(15, 3))
    for i, idx in enumerate(valid_indices):
        axes[i].imshow(history[idx], cmap='gray')
        axes[i].set_title(f"Iterasi {idx}")
        axes[i].axis('off')
    return fig_to_base64(fig)

def generate_final_comparison(original, reconstructed):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original, cmap='gray'); axes[0].set_title("Asli")
    axes[1].imshow(reconstructed, cmap='gray'); axes[1].set_title("Hasil Fraktal")
    for ax in axes: ax.axis('off')
    return fig_to_base64(fig)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    if 'file' not in request.files: return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    resize_val = int(request.form.get('size', 128))
    
    # 1. Load & Resize
    img = Image.open(file.stream).convert('L')
    img = img.resize((resize_val, resize_val))
    original_np = np.array(img)
    
    # 2. Compress
    start_t = time.time()
    fic = FractalImageCompression(range_size=8, domain_size=16, step=8)
    codes, debug_data = fic.compress(original_np)
    encode_time = time.time() - start_t
    
    # 3. Decompress
    result_np, history = fic.decompress(codes, original_np.shape, iterations=8)
    
    # 4. Hitung Statistik
    mse = np.mean((original_np.astype(float) - result_np.astype(float))**2)
    psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse != 0 else 100
    
    original_size_kb = original_np.size / 1024
    compressed_size_kb = (len(codes) * 28) / 1024
    ratio = original_np.size / (len(codes) * 28) if len(codes) > 0 else 0
    
    # 5. Generate Data untuk Frontend
    
    # Data Visualisasi
    viz_grid = generate_partition_grid(original_np, 8)
    viz_debug = generate_block_debug(debug_data)
    viz_evo = generate_evolution(history)
    viz_final = generate_final_comparison(original_np, result_np)
    
    # Data Sampel Matriks (8x8 pojok kiri)
    matrix_sample = original_np[0:8, 0:8].tolist()
    
    # Data Sampel Kode (5 baris)
    codes_sample = []
    for c in codes[:5]:
        codes_sample.append({
            'r': str(c['range_pos']),
            'd': str(c['domain_idx']),
            't': str(c['transform'][1]),
            's': f"{c['scale']:.2f}",
            'o': f"{c['offset']:.1f}"
        })

    response_data = {
        'metrics': {
            'time': f"{encode_time:.2f}s",
            'psnr': f"{psnr:.2f} dB",
            'mse': f"{mse:.2f}",
            'size_raw': f"{original_size_kb:.1f} KB",
            'size_frac': f"{compressed_size_kb:.1f} KB",
            'ratio': f"{ratio:.1f} : 1"
        },
        'visuals': {
            'grid': viz_grid,
            'debug': viz_debug,
            'evolution': viz_evo,
            'final': viz_final
        },
        'data': {
            'matrix': matrix_sample,
            'codes': codes_sample,
            'debug_info': {
                'pos': str(debug_data['pos']),
                'scale': f"{debug_data['info']['scale']:.2f}",
                'offset': f"{debug_data['info']['offset']:.2f}",
                'error': f"{debug_data['info']['error']:.4f}"
            }
        }
    }
    
    return jsonify(response_data)

# if __name__ == '__main__':
#     app.run(debug=True)