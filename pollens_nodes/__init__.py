"""Nœuds maison Pollens pour ComfyUI.

PollensMatchGrade : recale la couleur de la pièce éditée sur l'étalonnage de l'image d'origine.
Qwen-Image-2.1 reprend la couleur du packshot avec sa lumière de studio, neutre et saturée ; le reste de
l'image garde l'étalonnage du fine-tune. Le nœud mesure, dans l'espace Lab, la pièce d'origine (masque SAM
sur l'image d'origine) et la pièce éditée (masque SAM sur la sortie de 2.1), puis rapproche la seconde
de la première : luminosité et contraste, saturation (chroma), et, en option, dominante de teinte.
La teinte de la pièce éditée est conservée par défaut : c'est elle que l'édition a corrigée.
Les détails (texture, coutures, dentelle) ne bougent pas : la correction est une transformation
globale des couleurs, appliquée seulement dans le masque de la pièce éditée, bord adouci.
Aucune dépendance hors de torch.
"""
import torch
import torch.nn.functional as F

_M = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                   [0.2126729, 0.7151522, 0.0721750],
                   [0.0193339, 0.1191920, 0.9503041]])
_WHITE = torch.tensor([0.95047, 1.0, 1.08883])
_EPS, _KAPPA = 216 / 24389, 24389 / 27


def _to_lab(rgb):
    c = rgb.clamp(0, 1)
    lin = torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = lin @ _M.to(rgb).T / _WHITE.to(rgb)
    f = torch.where(xyz > _EPS, xyz.clamp(min=1e-8) ** (1 / 3), (_KAPPA * xyz + 16) / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return torch.stack([L, a, b], -1)


def _to_rgb(lab):
    L, a, b = lab.unbind(-1)
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200
    f = torch.stack([fx, fy, fz], -1)
    xyz = torch.where(f ** 3 > _EPS, f ** 3, (116 * f - 16) / _KAPPA) * _WHITE.to(lab)
    lin = xyz @ torch.linalg.inv(_M).to(lab).T
    lin = lin.clamp(0, 1)
    return torch.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin.clamp(min=1e-8) ** (1 / 2.4) - 0.055).clamp(0, 1)


def _fit(mask, h, w):
    m = mask if mask.dim() == 3 else mask.unsqueeze(0)
    if m.shape[-2:] != (h, w):
        m = F.interpolate(m.unsqueeze(1).float(), size=(h, w), mode="bilinear", align_corners=False).squeeze(1)
    return m.float().clamp(0, 1)


def _erode(m, r):
    if r <= 0:
        return m
    return -F.max_pool2d(-m.unsqueeze(1), 2 * r + 1, stride=1, padding=r).squeeze(1)


def _feather(m, r):
    if r <= 0:
        return m
    sigma = r / 2
    x = torch.arange(-r, r + 1, dtype=m.dtype, device=m.device)
    k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    k = k / k.sum()
    t = m.unsqueeze(1)
    t = F.conv2d(F.pad(t, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    t = F.conv2d(F.pad(t, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))
    return t.squeeze(1).clamp(0, 1)


def _stats(x, w):
    s = w.sum().clamp(min=1e-6)
    mu = (x * w).sum() / s
    sd = (((x - mu) ** 2) * w).sum().div(s).sqrt()
    return mu, sd


class PollensMatchGrade:
    CATEGORY = "Pollens"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "rapport")
    FUNCTION = "run"
    DESCRIPTION = "Recale la couleur de la pièce éditée sur l'étalonnage de la pièce d'origine (Lab, dans le masque)."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "edited": ("IMAGE", {"tooltip": "Sortie de 2.1, à la taille de l'image d'origine."}),
            "original": ("IMAGE", {"tooltip": "Image d'origine (étalonnage de référence)."}),
            "original_mask": ("MASK", {"tooltip": "Pièce dans l'image d'origine (SAM)."}),
            "edited_mask": ("MASK", {"tooltip": "Pièce dans l'image éditée (SAM sur la sortie de 2.1)."}),
            "luminance": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                    "tooltip": "Luminosité et contraste de la pièce recalés sur l'origine (0 = inchangés)."}),
            "saturation": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05,
                                     "tooltip": "Saturation moyenne recalée sur l'origine (0 = inchangée)."}),
            "dominante": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                    "tooltip": "Teinte moyenne ramenée vers l'origine. 0 par défaut : l'édition peut corriger la couleur."}),
            "erode": ("INT", {"default": 6, "min": 0, "max": 64, "tooltip": "Retrait des masques pour la mesure (évite les bords)."}),
            "feather": ("INT", {"default": 8, "min": 0, "max": 64, "tooltip": "Adoucissement du bord de la zone corrigée."}),
        }}

    def run(self, edited, original, original_mask, edited_mask, luminance, saturation, dominante, erode, feather):
        b, h, w, _ = edited.shape
        orig = original
        if orig.shape[1:3] != (h, w):
            orig = F.interpolate(orig.movedim(-1, 1), size=(h, w), mode="bilinear", align_corners=False).movedim(1, -1)
        mo = _erode(_fit(original_mask, h, w), erode)
        me = _fit(edited_mask, h, w)
        me_stats = _erode(me, erode)
        if mo.sum() < 500 or me_stats.sum() < 500:
            return (edited, "Masque trop petit : couleur inchangée.")
        out, reports = [], []
        for i in range(b):
            lab_e = _to_lab(edited[i].float())
            lab_o = _to_lab(orig[min(i, orig.shape[0] - 1)].float())
            wo = mo[min(i, mo.shape[0] - 1)]
            we = me_stats[min(i, me_stats.shape[0] - 1)]
            L, A, B = lab_e.unbind(-1)
            muLe, sdLe = _stats(L, we)
            muLo, sdLo = _stats(lab_o[..., 0], wo)
            k = (sdLo / sdLe.clamp(min=1e-3)).clamp(0.5, 2.0)
            L2 = L + luminance * ((muLo + (L - muLe) * k) - L)
            Ce = torch.sqrt(A ** 2 + B ** 2)
            Co = torch.sqrt(lab_o[..., 1] ** 2 + lab_o[..., 2] ** 2)
            muCe, _ = _stats(Ce, we)
            muCo, _ = _stats(Co, wo)
            r = (muCo / muCe.clamp(min=1e-3)).clamp(0.4, 1.6)
            s = 1 + saturation * (r - 1)
            A2, B2 = A * s, B * s
            if dominante > 0:
                muAo, _ = _stats(lab_o[..., 1], wo)
                muBo, _ = _stats(lab_o[..., 2], wo)
                muA2, _ = _stats(A2, we)
                muB2, _ = _stats(B2, we)
                A2 = A2 + dominante * (muAo - muA2)
                B2 = B2 + dominante * (muBo - muB2)
            fixed = _to_rgb(torch.stack([L2, A2, B2], -1)).to(edited)
            m = _feather(me[min(i, me.shape[0] - 1)].unsqueeze(0), feather)[0].unsqueeze(-1).to(edited)
            out.append(edited[i] * (1 - m) + fixed * m)
            reports.append(
                f"luminosité {muLe.item():.1f} → origine {muLo.item():.1f}, contraste ×{k.item():.2f} ; "
                f"saturation pièce {muCe.item():.1f} / origine {muCo.item():.1f} → ×{s.item():.2f} appliqué")
        return (torch.stack(out), "\n".join(reports))


NODE_CLASS_MAPPINGS = {"PollensMatchGrade": PollensMatchGrade}
NODE_DISPLAY_NAME_MAPPINGS = {"PollensMatchGrade": "Pollens · Étalonnage de la pièce"}
