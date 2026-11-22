"""Interfaces: an interaction form applied to an action abstraction.

    per_action   the model calls the action tools itself, one action a reply
    program      the model writes a program that calls them (``sandbox/``)
    sequence     a program that only acts: a series of actions, nothing seen in between
    free         every abstraction's tools side by side (RUI-Free)

``load_profile("rui.pose/v1+aids@owlv2-sam2")`` resolves an id into the
:class:`Profile` a Harness hands to the model: a named interface
(``ruikit/configs/interfaces.yaml``) with Harness components and a
perception backend composed onto it (``compose``).
"""

from ruikit.interfaces.base import Interface, Profile
from ruikit.interfaces.catalog import FORMS, interface_for, list_profiles, load_profile

__all__ = ["FORMS", "Interface", "Profile", "interface_for", "list_profiles", "load_profile"]
