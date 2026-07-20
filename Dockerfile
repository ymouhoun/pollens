# Version figée du worker officiel RunPod ComfyUI.
# Ne pas utiliser "latest", pour éviter qu'une mise à jour casse le worker.
ARG WORKER_VERSION=5.8.6

FROM runpod/worker-comfyui:${WORKER_VERSION}-base-cuda12.8.1

LABEL org.opencontainers.image.title="pollens-worker" \
      org.opencontainers.image.version="0.2.10"

# Configuration générale
ENV PYTHONUNBUFFERED=1 \
    COMFY_LOG_LEVEL=INFO \
    REFRESH_WORKER=false \
    HF_HUB_OFFLINE=0 \
    TRANSFORMERS_OFFLINE=1 \
    POLLEN_PREVIEW_ENABLED=true \
    POLLEN_PREVIEW_INTERVAL_MS=750 \
    POLLEN_STATUS_INTERVAL_MS=500 \
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
# Empêche Impact Pack de télécharger SAM dans l'image : le worker le récupère
# depuis le dépôt Hugging Face partagé lors du premier Face Detail.
RUN touch /comfyui/custom_nodes/skip_download_model

RUN comfy-node-install \
    https://github.com/gseth/ControlAltAI-Nodes \
    https://github.com/ClownsharkBatwing/RES4LYF \
    https://github.com/ltdrdata/ComfyUI-Impact-Pack \
    https://github.com/ltdrdata/ComfyUI-Impact-Subpack \
    https://github.com/rgthree/rgthree-comfy

# comfy-node-install clone les nodes mais ne garantit pas l'installation de
# leurs dépendances Python. Impact Pack et Impact Subpack ont besoin notamment
# d'OpenCV, tandis que RES4LYF a besoin d'OpenCV, Matplotlib et PyWavelets.
#
# On installe directement les requirements des deux packs Impact, sans appeler
# leur install.py (celui-ci n'est pas adapté au build non interactif RunPod).
# Pour RES4LYF, on utilise opencv-python-headless, déjà requis par Impact Pack,
# afin d'éviter d'installer simultanément les variantes GUI et headless.
RUN set -eu; \
    test -f /comfyui/custom_nodes/comfyui-impact-pack/requirements.txt; \
    test -f /comfyui/custom_nodes/comfyui-impact-subpack/requirements.txt; \
    uv pip install \
        -r /comfyui/custom_nodes/comfyui-impact-pack/requirements.txt \
        -r /comfyui/custom_nodes/comfyui-impact-subpack/requirements.txt; \
    uv pip install \
        huggingface_hub \
        PyWavelets \
        matplotlib \
        "numpy>=1.26.4" \
        opencv-python-headless

# Échec immédiat du build si les imports qui ont cassé le worker sont absents.
RUN python -c "import cv2; import pywt; import matplotlib; import huggingface_hub"

# Conserve le FaceDetailer d'Impact Pack et ajoute seulement une relance
# ciblée lorsqu'un patch raffiné est détecté comme presque entièrement noir.
#
# Le wrapper est enregistré depuis Impact Pack lui-même. Un dossier custom
# node séparé ne serait pas fiable ici : ComfyUI parcourt os.listdir() sans
# garantir qu'Impact Pack sera importé avant les modules qui en dépendent.
COPY pollen_face_detailer_retry.py \
    /comfyui/custom_nodes/comfyui-impact-pack/modules/impact/pollen_face_detailer_retry.py

RUN sed -i \
    '/^__all__ =/i\from impact.pollen_face_detailer_retry import PollenFaceDetailerAutoRetry\nNODE_CLASS_MAPPINGS["PollenFaceDetailerAutoRetry"] = PollenFaceDetailerAutoRetry\nNODE_DISPLAY_NAME_MAPPINGS["PollenFaceDetailerAutoRetry"] = "FaceDetailer (Pollen auto retry)"\n' \
    /comfyui/custom_nodes/comfyui-impact-pack/__init__.py \
    && grep -q 'NODE_CLASS_MAPPINGS\["PollenFaceDetailerAutoRetry"\]' \
    /comfyui/custom_nodes/comfyui-impact-pack/__init__.py

# Valide les imports réels au build, pas seulement la syntaxe Python. Cela
# évite qu'un node absent ne soit découvert qu'au moment d'une requête.
RUN cd /comfyui \
    && PYTHONPATH=/comfyui:/comfyui/custom_nodes/comfyui-impact-pack/modules \
    python -c "from impact.pollen_face_detailer_retry import PollenFaceDetailerAutoRetry; assert PollenFaceDetailerAutoRetry.INPUT_TYPES()"

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
    /opt/pollen/preview_handler.py \
    /comfyui/custom_nodes/comfyui-impact-pack/modules/impact/pollen_face_detailer_retry.py


# Remplace la commande de démarrage officielle :
# 1. bootstrap.py crée les liens vers les modèles
# 2. bootstrap.py lance ensuite /start.sh
CMD ["python", "/opt/pollen/bootstrap.py"]
