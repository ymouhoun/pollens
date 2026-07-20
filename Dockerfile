# Version figée du worker officiel RunPod ComfyUI.
# Ne pas utiliser "latest", pour éviter qu'une mise à jour casse le worker.
ARG WORKER_VERSION=5.8.6
ARG CUDA_VERSION=12.8.1

# llama-cpp-python doit correspondre à Python 3.12, utilisé par worker-comfyui
# 5.8.6. Le wheel contient les kernels des trois familles GPU autorisées par
# Pollens et est construit une seule fois dans une couche Docker réutilisable.
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS llm-enhancer-wheel-builder

ARG DEBIAN_FRONTEND=noninteractive
ARG LLAMA_CPP_PYTHON_VERSION=0.3.34

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
        python3 \
        python3-dev \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/llama-build \
    && /opt/llama-build/bin/python -m pip install --no-cache-dir \
        pip==25.1.1 \
        setuptools==80.9.0 \
        wheel==0.45.1 \
        scikit-build-core==0.11.6

ENV CMAKE_ARGS="-DGGML_CUDA=ON -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES=90-real;100-real;120-real;120-virtual" \
    FORCE_CMAKE=1 \
    CMAKE_BUILD_PARALLEL_LEVEL=8

RUN mkdir -p /wheelhouse \
    && /opt/llama-build/bin/python -m pip wheel \
        --no-cache-dir \
        --no-build-isolation \
        --no-deps \
        --no-binary=llama-cpp-python \
        --wheel-dir=/wheelhouse \
        llama-cpp-python==${LLAMA_CPP_PYTHON_VERSION}

COPY verify_llama_cuda_wheel.py /opt/verify_llama_cuda_wheel.py
RUN /opt/llama-build/bin/python /opt/verify_llama_cuda_wheel.py /wheelhouse

FROM runpod/worker-comfyui:${WORKER_VERSION}-base-cuda12.8.1

LABEL org.opencontainers.image.title="pollens-worker" \
      org.opencontainers.image.version="0.3.0"

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
    POLLEN_LORA_CACHE_MAX_ITEMS=5 \
    LLM_ENHANCER_WORKER_PYTHON=/opt/llm-enhancer-venv/bin/python \
    LLM_ENHANCER_GPU_TIMEOUT_SECONDS=900 \
    LLM_ENHANCER_STRICT_BACKEND_VERSION=1 \
    LLM_ENHANCER_LLAMA_CPP_VERSION=0.3.34 \
    COMFY_EXTRA_ARGS=--highvram


# Création explicite des dossiers que nous allons utiliser.
RUN mkdir -p \
    /opt/pollen \
    /comfyui/models/checkpoints \
    /comfyui/models/diffusion_models \
    /comfyui/models/unet \
    /comfyui/models/vae \
    /comfyui/models/clip \
    /comfyui/models/text_encoders \
    /comfyui/models/llm_gguf \
    /comfyui/models/loras \
    /comfyui/models/controlnet \
    /comfyui/models/ultralytics/bbox \
    /comfyui/models/ultralytics/segm \
    /comfyui/models/sams \
    /comfyui/models/upscale_models \
    /opt/pollen/face-cache

# Runtime isolé du Prompt Enhancer. ComfyUI ne charge jamais llama-cpp-python
# dans son propre processus : le custom node dialogue avec ce venv par un
# worker JSON-lines persistant.
RUN python -m venv /opt/llm-enhancer-venv
COPY prompt_enhancer_V2/requirements-runtime.lock /tmp/llm-enhancer-requirements.lock
RUN /opt/llm-enhancer-venv/bin/python -m pip install --no-cache-dir \
        -r /tmp/llm-enhancer-requirements.lock
COPY --from=llm-enhancer-wheel-builder /wheelhouse/ /opt/llm-enhancer-wheelhouse/
RUN /opt/llm-enhancer-venv/bin/python -m pip install --no-cache-dir --no-deps \
        /opt/llm-enhancer-wheelhouse/llama_cpp_python-*.whl \
    && /opt/llm-enhancer-venv/bin/python -c \
        "from importlib.metadata import files, version; assert version('llama-cpp-python') == '0.3.34'; assert any('libggml-cuda' in str(path) for path in (files('llama-cpp-python') or ()))"

# Ne pas importer llama_cpp pendant docker build : le driver libcuda.so.1 est
# injecté par RunPod uniquement lorsqu'un worker possède réellement un GPU.
# gpu_worker.py refait le contrôle dynamique llama_supports_gpu_offload() au
# démarrage du worker, après détection du GPU et de son compute capability.


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

# Source runtime du node LLM Prompt Enhancer et presets Pollens.
COPY prompt_enhancer_V2 /comfyui/custom_nodes/prompt_enhancer_V2

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

# Vérifie que la version installée d'Impact Pack expose toutes les API utilisées
# par le wrapper. Ne pas importer Impact Pack isolément ici : son initialisation
# dépend du cycle de chargement des custom nodes de ComfyUI.
RUN grep -q '^class FaceDetailer' \
        /comfyui/custom_nodes/comfyui-impact-pack/modules/impact/impact_pack.py \
    && grep -q '^class DetailerForEachAutoRetry' \
        /comfyui/custom_nodes/comfyui-impact-pack/modules/impact/impact_pack.py \
    && grep -q '^class BlackPatchRetryHook' \
        /comfyui/custom_nodes/comfyui-impact-pack/modules/impact/hooks.py

# ------------------------------------------------------------
# PREVIEW COMFYUI
# ------------------------------------------------------------
# Active la génération des previews intermédiaires par ComfyUI.
# Le worker Base44/RunPod devra ensuite transmettre ces images.
RUN sed -i \
    's/--disable-metadata/--disable-metadata ${COMFY_EXTRA_ARGS} --preview-method auto --preview-size 384/g' \
    /start.sh \
    && sed -i \
    's|python -u /handler.py|python -u /opt/pollen/preview_handler.py|g' \
    /start.sh \
    && grep -q -- '${COMFY_EXTRA_ARGS}' /start.sh \
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
    /comfyui/custom_nodes/comfyui-impact-pack/modules/impact/pollen_face_detailer_retry.py \
    /comfyui/custom_nodes/prompt_enhancer_V2/enhancer_node.py \
    /comfyui/custom_nodes/prompt_enhancer_V2/gpu_runtime.py \
    /comfyui/custom_nodes/prompt_enhancer_V2/gpu_worker.py \
    /comfyui/custom_nodes/prompt_enhancer_V2/preset_store.py \
    /comfyui/custom_nodes/prompt_enhancer_V2/prompt_logic.py

RUN cd /comfyui \
    && PYTHONPATH=/comfyui:/comfyui/custom_nodes python -c \
        "import prompt_enhancer_V2 as plugin; assert 'LLMPromptEnhancer' in plugin.NODE_CLASS_MAPPINGS"


# Remplace la commande de démarrage officielle :
# 1. bootstrap.py crée les liens vers les modèles
# 2. bootstrap.py lance ensuite /start.sh
CMD ["python", "/opt/pollen/bootstrap.py"]
