import sys
from pathlib import Path

import streamlit as st
import torch
import yaml
from joblib import load as joblib_load
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data.chestmnist_dataset import CHESTMNIST_LABELS, NUM_CLASSES
from src.data.transforms import get_autoencoder_transforms, get_chestmnist_transforms, get_rgb_transforms
from src.models.autoencoder import ConvAutoencoder
from src.models.multimodal_model import MultimodalFusionModel
from src.models.simple_cnn import SimpleCNN
from src.models.transfer_model import build_transfer_model
from src.models.vit_model import build_vit_model


def load_config():
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_classifier(model_name, config):
    if model_name == "simple_cnn":
        return SimpleCNN(num_classes=NUM_CLASSES, in_channels=1)
    if model_name == "transfer":
        return build_transfer_model(config["models"]["transfer_name"], NUM_CLASSES, pretrained=False)
    if model_name == "vit":
        return build_vit_model(config["models"]["vit_name"], NUM_CLASSES, pretrained=False)
    raise ValueError(f"Modèle inconnu: {model_name}")


@st.cache_resource
def load_models(supervised_path, ae_path, multimodal_path, vectorizer_path):
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classifier, class_ckpt = None, None
    if supervised_path and Path(supervised_path).exists():
        class_ckpt = torch.load(supervised_path, map_location=device)
        classifier = build_classifier(class_ckpt.get("model_name", "simple_cnn"), config).to(device)
        classifier.load_state_dict(class_ckpt["model_state_dict"])
        classifier.eval()

    autoencoder, ae_ckpt = None, None
    if ae_path and Path(ae_path).exists():
        ae_ckpt = torch.load(ae_path, map_location=device)
        autoencoder = ConvAutoencoder(in_channels=1).to(device)
        autoencoder.load_state_dict(ae_ckpt["model_state_dict"])
        autoencoder.eval()

    multimodal, multimodal_ckpt, vectorizer = None, None, None
    if multimodal_path and vectorizer_path and Path(multimodal_path).exists() and Path(vectorizer_path).exists():
        multimodal_ckpt = torch.load(multimodal_path, map_location=device)
        vectorizer = joblib_load(vectorizer_path)
        label_columns = multimodal_ckpt.get("label_columns", config["openi"]["label_columns"])
        tfidf_dim = multimodal_ckpt.get("tfidf_dim") or len(vectorizer.vocabulary_)
        multimodal = MultimodalFusionModel(
            tfidf_dim=tfidf_dim,
            num_classes=len(label_columns),
            pretrained_image=False,
        ).to(device)
        multimodal.load_state_dict(multimodal_ckpt["model_state_dict"])
        multimodal.eval()

    return config, device, classifier, class_ckpt, autoencoder, ae_ckpt, multimodal, multimodal_ckpt, vectorizer


st.set_page_config(page_title="Tri radiologique - démo", layout="wide")
st.title("Démonstrateur de tri radiologique")
st.warning(
    "Prototype pédagogique uniquement. Ce n'est pas un dispositif médical et il ne doit pas être utilisé "
    "pour décider d'une prise en charge."
)

config = load_config()
default_model = ROOT / config["paths"]["output_dir"] / "best_simple_cnn.pt"
default_ae = ROOT / config["paths"]["output_dir"] / "best_autoencoder.pt"
default_multimodal = ROOT / config["paths"]["output_dir"] / "best_openi_multimodal.pt"
default_vectorizer = ROOT / config["paths"]["output_dir"] / "openi_tfidf_vectorizer.joblib"

with st.sidebar:
    st.header("Modèles")
    supervised_path = st.text_input("Checkpoint classification", str(default_model))
    ae_path = st.text_input("Checkpoint autoencodeur", str(default_ae))
    multimodal_path = st.text_input("Checkpoint multimodal OpenI", str(default_multimodal))
    vectorizer_path = st.text_input("Vectorizer TF-IDF OpenI", str(default_vectorizer))

uploaded = st.file_uploader("Uploader une radiographie thoracique", type=["png", "jpg", "jpeg"])
report_text = st.text_area(
    "Compte-rendu optionnel",
    placeholder="Texte libre utilisé si un modèle multimodal OpenI est chargé.",
)

if uploaded is not None:
    raw_image = Image.open(uploaded)
    image = raw_image.convert("L")
    image_rgb = raw_image.convert("RGB")
    st.image(image, caption="Image chargée", width=320)

    config, device, classifier, _, autoencoder, ae_ckpt, multimodal, multimodal_ckpt, vectorizer = load_models(
        supervised_path,
        ae_path,
        multimodal_path,
        vectorizer_path,
    )
    image_size = config["chestmnist"]["size"]

    if classifier is None:
        st.info("Aucun checkpoint supervisé trouvé. Lancez d'abord un entraînement ChestMNIST.")
    else:
        transform = get_chestmnist_transforms(image_size, train=False)
        x = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.sigmoid(classifier(x)).cpu().squeeze(0).numpy()

        st.subheader("Probabilités par pathologie")
        labels = list(CHESTMNIST_LABELS.values())
        rows = sorted(zip(labels, probs), key=lambda item: item[1], reverse=True)
        st.dataframe(
            [{"pathologie": label, "probabilité": round(float(prob), 4)} for label, prob in rows],
            use_container_width=True,
        )

    if autoencoder is None:
        st.info("Aucun autoencodeur trouvé. Lancez `train_autoencoder.py` pour afficher un score d'anomalie.")
    else:
        ae_transform = get_autoencoder_transforms(image_size, train=False)
        x_ae = ae_transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            recon = autoencoder(x_ae)
            score = float(((x_ae - recon) ** 2).mean().cpu())
        threshold = ae_ckpt.get("anomaly_threshold") if ae_ckpt else None
        st.subheader("Score d'anomalie")
        st.metric("Erreur de reconstruction MSE", f"{score:.6f}")
        if threshold is not None:
            st.write(f"Seuil validation p95 : `{threshold:.6f}`")
            st.write("Interprétation :", "image atypique pour l'AE" if score >= threshold else "image non atypique selon l'AE")

    if report_text.strip():
        st.subheader("Prédiction multimodale OpenI")
        if multimodal is None or vectorizer is None:
            st.info("Aucun checkpoint multimodal complet trouvé. Lancez `train_multimodal.py` pour utiliser image + texte.")
        else:
            mm_transform = get_rgb_transforms(config["openi"]["image_size"], train=False)
            x_img = mm_transform(image_rgb).unsqueeze(0).to(device)
            x_txt = torch.tensor(vectorizer.transform([report_text]).toarray(), dtype=torch.float32).to(device)
            with torch.no_grad():
                mm_probs = torch.sigmoid(multimodal(x_img, x_txt)).cpu().squeeze(0).numpy()
            labels = multimodal_ckpt.get("label_columns", config["openi"]["label_columns"])
            st.dataframe(
                [{"label": label, "probabilité": round(float(prob), 4)} for label, prob in zip(labels, mm_probs)],
                use_container_width=True,
            )
