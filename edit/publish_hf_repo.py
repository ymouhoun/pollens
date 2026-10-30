"""Crée (ou met à jour) le dépôt Hugging Face privé de l'endpoint d'édition.

À lancer sur le pod RunPod, où les quatre modèles sont déjà téléchargés :
l'envoi part du datacenter, sans repasser par le Mac.

    /workspace/ComfyUI/venv/bin/python edit/publish_hf_repo.py

Le token HF doit avoir le droit d'écriture (`hf auth login` avant).
Un seul commit : les poids, puis le manifeste. Relancer le script ne renvoie
que ce qui a changé (empreintes vérifiées par huggingface_hub).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

REPO_ID = os.environ.get("POLLENS_EDIT_REPO", "ymouhoun/pollens-edit")
MODELS = Path(os.environ.get("COMFY_MODELS_DIR", "/workspace/ComfyUI/models"))
MANIFEST = Path(__file__).resolve().parent / "comfy-models.json"


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    operations = []
    missing = []
    for item in manifest["files"]:
        local = MODELS / item["target"]
        if not local.is_file():
            missing.append(str(local))
            continue
        size_gb = local.stat().st_size / 1e9
        print(f"  {item['source']}  ←  {local}  ({size_gb:.2f} Go)")
        operations.append(CommitOperationAdd(item["source"], str(local)))
    if missing:
        print("Fichiers introuvables :\n  " + "\n  ".join(missing))
        print("Téléchargez-les d'abord (voir edit/README.md), puis relancez.")
        return 1

    operations.append(CommitOperationAdd("comfy-models.json", str(MANIFEST)))

    api = HfApi()
    api.create_repo(REPO_ID, private=True, exist_ok=True)
    print(f"Envoi vers {REPO_ID} (privé)… les premières minutes servent au calcul des empreintes.")
    api.create_commit(
        repo_id=REPO_ID,
        operations=operations,
        commit_message="Poids de l'endpoint d'édition : Qwen-Image-2.1, Qwen3-VL 8B, VAE 2.1, SAM 3.1",
    )
    files = sorted(f for f in api.list_repo_files(REPO_ID))
    print("Commit OK. Contenu du dépôt :\n  " + "\n  ".join(files))
    return 0


if __name__ == "__main__":
    sys.exit(main())
