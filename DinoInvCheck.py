import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

# ======== 1. Carica il modello DINOv2 ========
model_name = "dinov2_vitb14"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️  Using device: {device}")

model = torch.hub.load("facebookresearch/dinov2", model_name)
model.eval()
model = model.to(device)

# ======== 2. Preprocessing ========
transform = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
])

# ======== 3. Funzione per estrarre feature ========
def extract_features(image):
    """
    Estrae features DINOv2 da un'immagine PIL
    """
    img_t = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = model(img_t)
    return feats.squeeze(0).cpu().numpy()

# ======== 4. Calcola cosine similarity ========
def cosine_similarity(feat1, feat2):
    """
    Calcola la similarità coseno tra due feature vectors
    """
    # Normalize
    feat1_norm = feat1 / (np.linalg.norm(feat1) + 1e-8)
    feat2_norm = feat2 / (np.linalg.norm(feat2) + 1e-8)
    # Cosine similarity
    similarity = np.dot(feat1_norm, feat2_norm)
    return similarity

# ======== 5. Test rotation invariance ========
def test_rotation_invariance(image_path, output_dir=None):
    """
    Testa la rotation invariance delle features DINOv2 per una singola immagine

    Returns:
        dict con le similarità per ogni rotazione
    """
    # Carica immagine originale
    image = Image.open(image_path).convert("RGB")

    # Estrai features dall'originale
    original_features = extract_features(image)

    # Definisci gli angoli di rotazione (8 rotazioni)
    angles = [0, 45, 90, 135, 180, 225, 270, 315]

    # Salva le immagini ruotate se richiesto
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        image.save(output_dir / "original.jpg")

    results = {}

    for angle in angles:
        if angle == 0:
            # Angolo 0 = immagine originale, similarity = 1.0
            results[angle] = 1.0
            continue

        # Ruota l'immagine
        rotated_image = image.rotate(angle, expand=True, resample=Image.BICUBIC)

        # Salva se richiesto
        if output_dir:
            rotated_image.save(output_dir / f"rotated_{angle}.jpg")

        # Estrai features
        rotated_features = extract_features(rotated_image)

        # Calcola similarità coseno
        similarity = cosine_similarity(original_features, rotated_features)
        results[angle] = float(similarity)  # Convert to native Python float

    return results

# ======== 6. Main - processa tutte le immagini del test set ========
def main():
    test_images_dir = Path("/home/lapemaya/PycharmProjects/Progetto Drone/datasetGenerale/train/images")
    output_dir = Path("/home/lapemaya/PycharmProjects/Progetto Drone/runs/rotation_invariance_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ottieni tutte le immagini
    image_files = list(test_images_dir.glob("*.jpg"))
    print(f"📊 Found {len(image_files)} images in test set")

    if len(image_files) == 0:
        print("❌ No images found!")
        return

    # Statistiche globali
    all_results = []
    angles = [0, 45, 90, 135, 180, 225, 270, 315]  # Include 0 per mostrare similarity = 1.0
    angle_similarities = {angle: [] for angle in angles}

    # Processa ogni immagine
    print("\n🔄 Processing images...")
    for i, image_path in enumerate(tqdm(image_files)):
        try:
            # Salva le rotazioni solo per le prime 5 immagini
            save_dir = output_dir / f"image_{i:03d}" if i < 5 else None

            # Testa rotation invariance
            results = test_rotation_invariance(image_path, output_dir=save_dir)

            # Aggiungi ai risultati
            all_results.append({
                "image": image_path.name,
                "similarities": results
            })

            # Aggiungi alle statistiche per angolo
            for angle in angles:
                angle_similarities[angle].append(results[angle])

        except Exception as e:
            print(f"\n⚠️  Error processing {image_path.name}: {e}")
            continue

    # ======== 7. Calcola statistiche ========
    print("\n" + "="*80)
    print("📈 ROTATION INVARIANCE ANALYSIS")
    print("="*80)

    print(f"\n✅ Successfully processed {len(all_results)} / {len(image_files)} images")

    print("\n📊 Cosine Similarity Statistics by Rotation Angle:")
    print("-" * 80)
    print(f"{'Angle':<10} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-" * 80)

    overall_similarities = []
    overall_similarities_no_zero = []  # For computing mean without the trivial 0° case

    for angle in angles:
        similarities = angle_similarities[angle]
        if len(similarities) > 0:
            mean_sim = np.mean(similarities)
            std_sim = np.std(similarities)
            min_sim = np.min(similarities)
            max_sim = np.max(similarities)

            print(f"{angle}°{'':<7} {mean_sim:<12.6f} {std_sim:<12.6f} {min_sim:<12.6f} {max_sim:<12.6f}")
            overall_similarities.extend(similarities)

            # Exclude 0° from overall mean computation
            if angle != 0:
                overall_similarities_no_zero.extend(similarities)

    print("-" * 80)

    # Statistiche complessive (excluding 0° for more meaningful interpretation)
    if len(overall_similarities_no_zero) > 0:
        overall_mean = np.mean(overall_similarities_no_zero)
        overall_std = np.std(overall_similarities_no_zero)
        overall_min = np.min(overall_similarities_no_zero)
        overall_max = np.max(overall_similarities_no_zero)

        print(f"{'Overall*':<10} {overall_mean:<12.6f} {overall_std:<12.6f} {overall_min:<12.6f} {overall_max:<12.6f}")
        print("* Overall statistics exclude 0° rotation (trivial case)")
        print("="*80)

        print("\n💡 Interpretation:")
        print(f"   • Average cosine similarity across rotations (excluding 0°): {overall_mean:.6f}")
        print(f"   • Range: 1.0 = identical features, 0.0 = completely different")

        if overall_mean > 0.9:
            invariance_level = "EXCELLENT (highly rotation invariant)"
        elif overall_mean > 0.8:
            invariance_level = "GOOD (reasonably rotation invariant)"
        elif overall_mean > 0.7:
            invariance_level = "MODERATE (some rotation sensitivity)"
        else:
            invariance_level = "LOW (highly rotation sensitive)"

        print(f"   • Rotation Invariance Level: {invariance_level}")

        # Trova gli angoli più problematici (escludendo 0°)
        angle_means = {angle: np.mean(angle_similarities[angle]) for angle in angles if angle != 0}
        worst_angle = min(angle_means, key=angle_means.get)  # Lowest similarity = worst
        best_angle = max(angle_means, key=angle_means.get)   # Highest similarity = best

        print(f"\n   • Most challenging rotation: {worst_angle}° (sim: {angle_means[worst_angle]:.6f})")
        print(f"   • Least challenging rotation: {best_angle}° (sim: {angle_means[best_angle]:.6f})")

    # ======== 8. Visualizza distribuzione ========
    if len(overall_similarities) > 0:
        # Plot 1: Box plot per angolo
        plt.figure(figsize=(10, 6))
        data_to_plot = [angle_similarities[angle] for angle in angles]
        plt.boxplot(data_to_plot, labels=[f"{a}°" for a in angles])
        plt.xlabel("Rotation Angle", fontsize=12)
        plt.ylabel("Cosine Similarity", fontsize=12)
        plt.title("Cosine Similarity Distribution by Rotation Angle", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)  # Similarity range is [0, 1]

        plot1_path = output_dir / "rotation_boxplot.png"
        plt.savefig(plot1_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Box plot saved to: {plot1_path}")
        plt.close()

        # Plot 2: Bar chart con media per angolo
        plt.figure(figsize=(10, 6))
        mean_values = [np.mean(angle_similarities[angle]) for angle in angles]
        bars = plt.bar([f"{a}°" for a in angles], mean_values, alpha=0.7, edgecolor='black')

        # Colora la barra a 0° in modo diverso per evidenziarla
        bars[0].set_color('green')
        bars[0].set_alpha(0.5)

        plt.xlabel("Rotation Angle", fontsize=12)
        plt.ylabel("Mean Cosine Similarity", fontsize=12)
        plt.title("Mean Cosine Similarity by Rotation Angle", fontsize=14)
        plt.grid(True, alpha=0.3, axis='y')
        plt.ylim(0, 1.05)  # Similarity range is [0, 1]

        # Aggiungi i valori sopra le barre
        for i, (angle, mean_val) in enumerate(zip(angles, mean_values)):
            plt.text(i, mean_val + 0.02, f'{mean_val:.3f}', ha='center', va='bottom', fontsize=9)

        plot2_path = output_dir / "rotation_barchart.png"
        plt.savefig(plot2_path, dpi=150, bbox_inches='tight')
        print(f"📊 Bar chart saved to: {plot2_path}")
        plt.close()

    # ======== 9. Salva risultati in JSON ========
    results_dict = {
        "summary": {
            "total_images": len(image_files),
            "processed_images": len(all_results),
            "overall_mean_similarity": float(overall_mean) if len(overall_similarities_no_zero) > 0 else None,
            "overall_std_similarity": float(overall_std) if len(overall_similarities_no_zero) > 0 else None,
            "note": "Overall statistics exclude 0° rotation (trivial case with similarity = 1.0)",
            "angle_statistics": {
                str(angle): {
                    "mean": float(np.mean(angle_similarities[angle])),
                    "std": float(np.std(angle_similarities[angle])),
                    "min": float(np.min(angle_similarities[angle])),
                    "max": float(np.max(angle_similarities[angle]))
                }
                for angle in angles if len(angle_similarities[angle]) > 0
            }
        }
    }

    json_path = output_dir / "rotation_invariance_results.json"
    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"💾 Results saved to: {json_path}")
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()
