# Version figée du worker officiel RunPod ComfyUI.
# Ne pas utiliser "latest", pour éviter qu'une mise à jour casse le worker.
ARG WORKER_VERSION=5.8.6

FROM runpod/worker-comfyui:${WORKER_VERSION}-base-cuda12.8.1

# Configuration générale
ENV PYTHONUNBUFFERED=1 \
    COMFY_LOG_LEVEL=INFO \
    REFRESH_WORKER=false \
    HF_HUB_OFFLINE=0 \
    TRANSFORMERS_OFFLINE=1 \
    POLLEN_PREVIEW_ENABLED=true \
    POLLEN_PREVIEW_INTERVAL_MS=750 \
    POLLEN_PREVIEW_MAX_BYTES=500000 \
    POLLEN_LORA_CACHE_MAX_ITEMS=5


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
    /comfyui/models/controlnet \
    /comfyui/models/ultralytics/bbox \
    /comfyui/models/ultralytics/segm \
    /comfyui/models/sams \
    /comfyui/models/upscale_models \
    /opt/pollen/face-cache


# ------------------------------------------------------------
# CUSTOM NODES
# ------------------------------------------------------------
# Custom nodes nécessaires au workflow :
# - ControlAltAI-Nodes fournit FluxResolutionNode
# - RES4LYF fournit les samplers utilisés par Qwen Image
# - Impact Pack fournit FaceDetailer et SAMLoader
# - Impact Subpack fournit les détecteurs Ultralytics
# - rgthree fournit le Power LoRA Loader
RUN comfy-node-install \
    https://github.com/gseth/ControlAltAI-Nodes \
    https://github.com/ClownsharkBatwing/RES4LYF \
    https://github.com/ltdrdata/ComfyUI-Impact-Pack \
    https://github.com/ltdrdata/ComfyUI-Impact-Subpack \
    https://github.com/rgthree/rgthree-comfy
RUN set -eu; \
    for requirements in \
        /comfyui/custom_nodes/RES4LYF/requirements.txt \
        /comfyui/custom_nodes/ComfyUI-Impact-Pack/requirements.txt \
        /comfyui/custom_nodes/ComfyUI-Impact-Subpack/requirements.txt \
        /comfyui/custom_nodes/rgthree-comfy/requirements.txt; do \
        if [ -f "$requirements" ]; then uv pip install -r "$requirements"; fi; \
    done; \
    uv pip install huggingface_hub; \
    touch /comfyui/custom_nodes/skip_download_model; \
    COMFYUI_PATH=/comfyui COMFYUI_MODEL_PATH=/comfyui/models \
        python /comfyui/custom_nodes/ComfyUI-Impact-Pack/install.py; \
    python -c "import pywt; import huggingface_hub"

# ------------------------------------------------------------
# PREVIEW COMFYUI
# ------------------------------------------------------------
# Active la génération des previews intermédiaires par ComfyUI.
# Le worker Base44/RunPod devra ensuite transmettre ces images.
RUN sed -i \
    's/--disable-metadata/--disable-metadata --preview-method auto --preview-size 384/g' \
    /start.sh \
    && sed -i \
    's|python -u /handler.py|python -u /opt/pollen/preview_handler.py|g' \
    /start.sh \
    && grep -q -- "--preview-method auto" /start.sh \
    && grep -q -- "/opt/pollen/preview_handler.py" /start.sh


# Ajout du bootstrap modèles, du cache de LoRAs et des previews progressives.
COPY bootstrap.py face_asset_cache.py preview_bridge.py preview_handler.py /opt/pollen/


# Vérifie pendant le build que les scripts Python sont syntaxiquement valides.
RUN python -m py_compile \
    /opt/pollen/bootstrap.py \
    /opt/pollen/face_asset_cache.py \
    /opt/pollen/preview_bridge.py \
    /opt/pollen/preview_handler.py


# Remplace la commande de démarrage officielle :
# 1. bootstrap.py crée les liens vers les modèles
# 2. bootstrap.py lance ensuite /start.sh
CMD ["python", "/opt/pollen/bootstrap.py"]
