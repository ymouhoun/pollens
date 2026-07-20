# Pollens LLM Prompt Enhancer

Node ComfyUI d'amélioration de prompts par LLM GGUF. Cette version est dédiée
à l'enhancer texte : le captioner a été retiré du package actif.

## Garanties de cette version

- Identifiant ComfyUI conservé : `LLMPromptEnhancer`.
- Entrées et sorties historiques conservées, ainsi que les 13 presets.
- Modèle conservé en mémoire entre les générations.
- Inférence exécutée dans un worker GPU persistant et isolé de ComfyUI.
- Aucun fallback CPU : une incompatibilité arrête explicitement le workflow.
- Cibles certifiées : H200 (`sm_90`), B200 (`sm_100`) et RTX PRO 6000
  Blackwell (`sm_120`).

L'artefact Docker certifié exige :

- `llama-cpp-python==0.3.34` ;
- driver NVIDIA `>= 570.124.06` ;
- un wheel contenant du SASS pour `sm_90`, `sm_100`, `sm_120` et du PTX
  `compute_120`.

Hors de cette image, une autre version de `llama-cpp-python` n'est pas rejetée
sur son seul numéro : le worker tente un véritable chargement et une génération
GPU. Cela permet d'utiliser un environnement existant comme la version 0.3.20,
tout en détectant réellement un wheel sans kernels compatibles.

## Architecture

Le processus ComfyUI ne charge jamais `llama-cpp-python`. Il lance un worker
Python séparé qui vérifie le GPU, charge le GGUF, le conserve en VRAM et traite
les requêtes suivantes sans rechargement. Un abort natif CUDA tue uniquement
le worker et remonte une erreur claire à ComfyUI.

La logique des presets, de composition du prompt et de nettoyage de la sortie
est indépendante de CUDA et couverte par les tests unitaires.

## Archive ComfyUI propre

Pour une installation manuelle sur un pod qui possède déjà un backend CUDA :

```bash
python3 scripts/build_release.py
```

Le fichier `dist/prompt_enhancer_V2.zip` contient un unique dossier racine
`prompt_enhancer_V2/`. Il faut extraire ce dossier dans `ComfyUI/custom_nodes/`.
Les sources de test, Docker et `legacy/` ne sont pas incluses et ne doivent pas
être décompressées directement à la racine de `custom_nodes/`.

## Construire le wheel CUDA universel

Depuis le dossier du plugin :

```bash
docker build \
  -f docker/Dockerfile \
  --target wheel-export \
  --output type=local,dest=./dist \
  .
```

La construction utilise CUDA 12.8.1 et échoue si `cuobjdump` ne confirme pas
la présence des trois architectures. Il n'existe plus de compilation au
démarrage du pod. L'image fournie construit un wheel CPython 3.10, correspondant
à l'environnement historique du plugin. Si Pollens change de version Python,
le wheel doit être reconstruit avec cette même version.

## Intégration dans l'image Pollens

Le worker peut vivre dans son propre environnement Python afin de ne pas
modifier les dépendances de ComfyUI :

```dockerfile
RUN python3 -m venv /opt/llm-enhancer-venv
COPY requirements-runtime.lock /tmp/requirements-runtime.lock
RUN /opt/llm-enhancer-venv/bin/pip install \
      --no-cache-dir -r /tmp/requirements-runtime.lock
COPY dist/llama_cpp_python-*.whl /tmp/llama_cpp_python.whl
RUN /opt/llm-enhancer-venv/bin/pip install \
      --no-cache-dir --no-deps /tmp/llama_cpp_python.whl

COPY . /opt/ComfyUI/custom_nodes/llm-prompt-enhancer
ENV LLM_ENHANCER_WORKER_PYTHON=/opt/llm-enhancer-venv/bin/python
ENV LLM_ENHANCER_GPU_TIMEOUT_SECONDS=900
ENV LLM_ENHANCER_STRICT_BACKEND_VERSION=1
ENV LLM_ENHANCER_LLAMA_CPP_VERSION=0.3.34
```

Adaptez `/opt/ComfyUI` au chemin utilisé par Pollens. Le GGUF reste dans
`ComfyUI/models/llm_gguf/`.

## Certification sur un GPU réel

Construire l'image de certification :

```bash
docker build \
  -f docker/Dockerfile \
  --target certification \
  -t pollens-llm-enhancer:certification \
  .
```

Puis exécuter exactement cette commande sur un pod H200, B200 et RTX PRO 6000 :

```bash
docker run --rm --gpus all \
  -v /chemin/vers/models:/models:ro \
  pollens-llm-enhancer:certification \
  --model /models/qwen_8b_q8_0.gguf
```

Le test charge réellement le modèle avec flash attention, effectue une
génération et renvoie un objet JSON contenant `"certified": true`, le GPU,
l'architecture, le driver, la version de `llama-cpp-python` et la durée.
La même image doit passer sur les trois machines avant publication.

## Presets

Les presets sont les fichiers `presets/system_prompt_<nom>.md`. Un negative
prompt spécifique peut être déclaré dans le fichier :

```html
<!-- NEGATIVE: blurry, watermark, text -->
```

Le marqueur est retiré du system prompt et exposé sur la sortie
`negative_prompt`.

## Comportement en cas d'erreur

Une absence de GPU, un GPU non certifié, un driver trop ancien, un wheel CPU ou
un kernel CUDA incompatible provoquent une erreur de workflow. Le prompt
d'origine n'est pas renvoyé silencieusement et aucune génération CPU n'est
tentée. Une version différente du backend produit un avertissement et doit
réussir le test GPU réel ; elle ne devient bloquante que lorsque
`LLM_ENHANCER_STRICT_BACKEND_VERSION=1`.

Pour un essai de développement uniquement, le verrou de version peut être
désactivé avec `LLM_ENHANCER_ALLOW_UNCERTIFIED_BUILD=1`. Ce réglage ne doit pas
être utilisé dans une image Pollens publiée.

L'ancienne implémentation incluant le captioner et l'installateur dynamique est
conservée dans `legacy/` pour référence, mais n'est plus importée.
