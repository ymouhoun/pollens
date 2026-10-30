# Endpoint d'édition produit (`pollens-edit`)

Phase 1 de l'édition produit : SAM 3.1 trouve la pièce à remplacer, Qwen3-VL 8B rédige les textes à partir du packshot,
Qwen-Image-2.1 insère le produit, et seule la zone masquée est recollée dans l'image d'origine.
La phase 2 (harmonisation par le fine-tune) tourne sur l'endpoint du modèle de l'image.

Les images arrivent par **adresse R2 signée** (`{name, url}`), téléchargées par le worker (`remote_images.py`) :
ce code est commun aux deux images Docker. **Avant tout, fusionner la branche `worker-shared` dans `main`** : elle
doit reconstruire les endpoints de génération, dont Editorial, qui en a besoin pour l'harmonisation : après la fusion, créer une release GitHub qui cible `main`. Les générations
actuelles n'en sont pas affectées (elles envoient toujours leurs images en base64).

| Élément | Valeur |
|---|---|
| Branche GitHub | `worker-edit` (jamais `main` : un push ici ne doit pas reconstruire les 7 endpoints de génération) |
| Dockerfile | `Dockerfile.edit` (base `runpod/worker-comfyui:5.10.0-base-cuda12.8.1`, ComfyUI figé en `v0.42.5`) |
| Custom nodes | `pollens_nodes` (maison, étalonnage de la pièce, torch seul) ; le reste est natif |
| Dépôt HF | `ymouhoun/pollens-edit` (privé), manifeste `edit/comfy-models.json` |
| Poids | Qwen-Image-2.1 int8 (6,8 Go), Qwen3-VL 8B int8 (8,7 Go), VAE 2.1 (0,6 Go), SAM 3.1 (1,7 Go) |
| Licence | Qwen-Image-2.1 et son encodeur : Qwen Research License (non commercial tant que l'accord n'est pas signé). SAM 3.1 : SAM License, usage commercial autorisé |

## 1. Créer le dépôt Hugging Face (depuis le pod RunPod, une fois)

Les quatre fichiers doivent être dans `/workspace/ComfyUI/models/` (c'est déjà le cas sur le pod de test).

```bash
cd /workspace && git clone -b worker-edit https://github.com/ymouhoun/pollens.git pollens-worker-edit
```
```bash
/workspace/ComfyUI/venv/bin/hf auth login
```
```bash
/workspace/ComfyUI/venv/bin/python /workspace/pollens-worker-edit/edit/publish_hf_repo.py
```

Le token doit avoir le droit d'écriture. Le script affiche les quatre fichiers, puis « Commit OK » et le contenu du dépôt.

## 2. Créer l'endpoint RunPod

Chemins à vérifier sur l'interface du jour.

1. Serverless → **New Endpoint** → **Import Git Repository** → `ymouhoun/pollens`.
2. **Branch** : `worker-edit`. **Dockerfile Path** : `Dockerfile.edit`.
3. Réglages, alignés sur l'endpoint Editorial :
   - type **Queue**, Active workers 0, Max workers 1, Idle timeout 60 s, Execution timeout 1200 s, FlashBoot activé ;
   - **Container disk 80 Go** ;
   - GPU : RTX PRO 6000 Blackwell en priorité (48 Go suffisent, 96 Go laissent de la marge à 2048 px) ;
   - **Model / Cached model** : `ymouhoun/pollens-edit` ;
   - variables : `HF_MODEL_ID=ymouhoun/pollens-edit`, `HF_TOKEN` (le même token en lecture que les autres endpoints).
4. Attendre le build `Completed` (premier build plus long : ComfyUI est remplacé et testé pendant la construction).
5. Logs du premier worker : `Linked /comfyui/models/checkpoints/sam3.1_multiplex_fp16.safetensors -> …` et les trois autres liens.

## 3. Brancher l'app

- Base44, secret **`RUNPOD_ENDPOINT_ID_EDIT`** = l'ID du nouvel endpoint.
- Le secret `MEDIA_SIGNING_SECRET` doit être posé (c'est déjà le cas depuis R2) : la fonction signe les adresses des images.
- Fonctions : `runpodEditSubmit` (nouvelle), `runpodStatus` (renvoie maintenant `texts`) et `mediaUploadUrl` (nouvelles variantes d'image). Publish, puis si besoin demander au Builder : « Redeploy runpodEditSubmit and runpodStatus from GitHub main without changing source ».

## Mises à jour

- Code du worker d'édition : commit sur `worker-edit`, puis **créer une release GitHub** qui cible `worker-edit` : un simple push ne met pas l'endpoint à jour (RunPod ne reconstruit qu'à la publication d'une release). Même règle pour `main` et les endpoints de génération.
- Poids : relancer `publish_hf_repo.py` après avoir changé les fichiers sur le pod, puis rafraîchir le Cached Model de l'endpoint (retirer, sauvegarder, remettre, sauvegarder) et recycler les workers.
