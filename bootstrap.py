import json
import os
import sys
from pathlib import Path


CACHE_ROOT = Path("/runpod-volume/huggingface-cache/hub")
COMFY_MODELS_ROOT = Path("/comfyui/models")
MANIFEST_FILENAME = "comfy-models.json"


def log(message: str) -> None:
    print(f"[pollen-bootstrap] {message}", flush=True)


def safe_relative_path(value: str) -> Path:
    """
    Refuse les chemins absolus ou contenant '..'.
    Cela empêche un manifeste incorrect d'écrire ailleurs
    que dans les dossiers prévus.
    """
    path = Path(value)

    if path.is_absolute():
        raise RuntimeError(f"Absolute path is forbidden: {value}")

    if not path.parts or ".." in path.parts:
        raise RuntimeError(f"Unsafe relative path: {value}")

    return path


def detect_model_id() -> str:
    """
    Utilise HF_MODEL_ID si renseigné dans RunPod.

    En secours, détecte automatiquement le seul dépôt présent
    dans le cache. Chaque endpoint RunPod ne devant avoir qu'un
    seul Cached Model, il devrait n'y avoir qu'un candidat.
    """
    configured_model_id = os.environ.get("HF_MODEL_ID")

    if configured_model_id:
        return configured_model_id.strip()

    if not CACHE_ROOT.exists():
        raise RuntimeError(
            f"Hugging Face cache directory not found: {CACHE_ROOT}"
        )

    candidates = sorted(
        path
        for path in CACHE_ROOT.glob("models--*")
        if path.is_dir()
    )

    if len(candidates) != 1:
        names = [path.name for path in candidates]
        raise RuntimeError(
            "Unable to detect a unique cached model. "
            f"Found {len(candidates)} candidates: {names}. "
            "Set HF_MODEL_ID explicitly in the RunPod endpoint."
        )

    # models--organisation--repo devient organisation/repo
    encoded_name = candidates[0].name.removeprefix("models--")
    parts = encoded_name.split("--", maxsplit=1)

    if len(parts) != 2:
        raise RuntimeError(
            f"Unexpected Hugging Face cache directory: {candidates[0]}"
        )

    return f"{parts[0]}/{parts[1]}"


def resolve_snapshot(model_id: str) -> Path:
    """
    Résout :
    organisation/repo

    vers :
    /runpod-volume/huggingface-cache/hub/
      models--organisation--repo/snapshots/<revision>
    """
    encoded_model_id = model_id.replace("/", "--")
    repository_root = CACHE_ROOT / f"models--{encoded_model_id}"
    snapshots_root = repository_root / "snapshots"
    main_ref = repository_root / "refs" / "main"

    if not repository_root.is_dir():
        raise RuntimeError(
            f"Cached repository not found: {repository_root}"
        )

    # Méthode normale : lire la révision correspondant à main.
    if main_ref.is_file():
        revision = main_ref.read_text(encoding="utf-8").strip()
        snapshot = snapshots_root / revision

        if snapshot.is_dir():
            return snapshot

    # Secours si refs/main n'existe pas.
    if snapshots_root.is_dir():
        snapshots = sorted(
            path
            for path in snapshots_root.iterdir()
            if path.is_dir()
        )

        if len(snapshots) == 1:
            return snapshots[0]

        if snapshots:
            # Prend la révision la plus récemment modifiée.
            return max(snapshots, key=lambda path: path.stat().st_mtime)

    raise RuntimeError(
        f"No cached snapshot found for model: {model_id}"
    )


def create_model_links(snapshot: Path) -> None:
    manifest_path = snapshot / MANIFEST_FILENAME

    if not manifest_path.is_file():
        raise RuntimeError(
            f"Manifest not found: {manifest_path}"
        )

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    files = manifest.get("files")

    if not isinstance(files, list) or not files:
        raise RuntimeError(
            f"'files' is missing or empty in {manifest_path}"
        )

    for item in files:
        if not isinstance(item, dict):
            raise RuntimeError(
                f"Invalid manifest entry: {item!r}"
            )

        source_relative = safe_relative_path(item["source"])
        target_relative = safe_relative_path(item["target"])

        source = snapshot / source_relative
        target = COMFY_MODELS_ROOT / target_relative

        if not source.is_file():
            raise RuntimeError(
                f"Model file not found in cached repository: {source}"
            )

        target.parent.mkdir(parents=True, exist_ok=True)

        # Path.exists() renvoie False pour certains liens cassés ;
        # is_symlink() permet donc de couvrir les deux situations.
        if target.exists() or target.is_symlink():
            if target.is_symlink():
                try:
                    if target.resolve() == source.resolve():
                        log(f"Already linked: {target}")
                        continue
                except FileNotFoundError:
                    pass

            raise RuntimeError(
                f"Target already exists and will not be overwritten: {target}"
            )

        os.symlink(source, target)

        log(f"Linked {target} -> {source}")


def main() -> None:
    model_id = detect_model_id()
    log(f"Using cached model: {model_id}")

    snapshot = resolve_snapshot(model_id)
    log(f"Resolved snapshot: {snapshot}")

    create_model_links(snapshot)
    log("All model links created successfully")

    # Remplace le processus Python par le démarrage officiel du worker.
    # Cela permet à RunPod de gérer correctement arrêt et redémarrage.
    os.execv("/start.sh", ["/start.sh"])


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        log(f"FATAL: {error}")
        sys.exit(1)