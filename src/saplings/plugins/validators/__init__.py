"""Validator plugins for Saplings."""

from saplings.core.plugin import register_plugin
from saplings.plugins.validators.code_validator import CodeValidator
from saplings.plugins.validators.factual_validator import FactualValidator

# Register the plugins
register_plugin(CodeValidator)
register_plugin(FactualValidator)

__all__ = ["CodeValidator", "FactualValidator"]
