from .encoders.modules import GeneralConditioner

UNCONDITIONAL_CONFIG = {
    "target": "vwm.modules.GeneralConditioner",
    "params": {"emb_models": list()}
}
