# Version figée du worker officiel RunPod ComfyUI.
# Ne pas utiliser "latest", pour éviter qu'une mise à jour casse le worker.
ARG WORKER_VERSION=5.8.6

FROM runpod/worker-comfyui:${WORKER_VERSION}-base-cuda12.8.1

# Configuration générale
ENV PYTHONUNBUFFERED=1 \
    COMFY_LOG_LEVEL=INFO \
    REFRESH_WORKER=false \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1


# Création explicite des dossiers que nous allons utiliser.
RUN mkdir -p \
    /opt/pollen \
    /comfyui/models/checkpoints \
    /comfyui/models/diffusion_models \
    /comfyui/models/unet \
    /comfyui/models/vae \
    /comfyui/models/clip \
    /comfyui/models/text_encoders \
    /comfyui/models/loras \
    /comfyui/models/controlnet


# ------------------------------------------------------------
# CUSTOM NODES
# ------------------------------------------------------------
# Custom nodes nécessaires au workflow :
# - ControlAltAI-Nodes fournit FluxResolutionNode
# - RES4LYF fournit le sampler res_3m
RUN comfy-node-install \
    https://github.com/gseth/ControlAltAI-Nodes \
    https://github.com/ClownsharkBatwing/RES4LYF
RUN uv pip install -r /comfyui/custom_nodes/RES4LYF/requirements.txt \
    && python -c "import pywt"

# ------------------------------------------------------------
# PREVIEW COMFYUI
# ------------------------------------------------------------
# Active la génération des previews intermédiaires par ComfyUI.
# Le worker Base44/RunPod devra ensuite transmettre ces images.
RUN sed -i \
    's/--disable-metadata/--disable-metadata --preview-method auto --preview-size 384/g' \
    /start.sh \
    && grep -q -- "--preview-method auto" /start.sh


# Ajout du script qui connectera le cache Hugging Face à ComfyUI.
COPY bootstrap.py /opt/pollen/bootstrap.py


# Vérifie pendant le build que le script Python est syntaxiquement valide.
RUN python -m py_compile /opt/pollen/bootstrap.py


# Remplace la commande de démarrage officielle :
# 1. bootstrap.py crée les liens vers les modèles
# 2. bootstrap.py lance ensuite /start.sh
CMD ["python", "/opt/pollen/bootstrap.py"]
