# labels.py — crop groups derived from agro_context (the single source of truth)
import agro_context

CROP_GROUPS: dict[str, list[int]] = {}
for idx, info in agro_context.DISEASE_INFO.items():
    CROP_GROUPS.setdefault(info["crop"], []).append(int(idx))
CROP_GROUPS = {crop: sorted(ix) for crop, ix in CROP_GROUPS.items()}

CROPS = sorted(CROP_GROUPS)
