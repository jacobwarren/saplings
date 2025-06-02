from __future__ import annotations

"""Validator plugins for Saplings."""


from saplings.api.registry import RegistrationMode, register_plugin
from saplings.plugins.validators.code_validator import CodeValidator
from saplings.plugins.validators.factual_validator import FactualValidator

# Register the plugins with SKIP mode to avoid warnings
register_plugin(CodeValidator, mode=RegistrationMode.SKIP)
register_plugin(FactualValidator, mode=RegistrationMode.SKIP)

__all__ = ["CodeValidator", "FactualValidator"]
