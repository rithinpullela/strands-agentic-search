"""A minimal analogue of Apache Commons ``StringSubstitutor``.

OpenSearch templates its prompts with ``${parameters.<key>}`` placeholders and
supports the ``:-`` default syntax (``${parameters.<key>:-some default}``).
The QueryPlanningTool constructs substitutors with prefix ``${parameters.`` and
suffix ``}`` in two places:

* ``MLModelTool`` substitutes the system/user prompts before the model call.
* ``executeQueryPlanning`` substitutes the fallback query when the model
  returns nothing usable.

We replicate the subset of behavior the tool relies on: simple ``${prefix.key}``
replacement and the ``:-default`` fallback (used when the key is missing).
"""

from __future__ import annotations

import re
from typing import Mapping

_DEFAULT_PREFIX = "${parameters."
_DEFAULT_SUFFIX = "}"


def substitute(
    template: str,
    values: Mapping[str, object],
    *,
    prefix: str = _DEFAULT_PREFIX,
    suffix: str = _DEFAULT_SUFFIX,
) -> str:
    """Replace ``{prefix}key{suffix}`` placeholders in ``template``.

    Supports the ``key:-default`` form: if ``key`` is absent from ``values``,
    the literal text after ``:-`` is used. Unresolved placeholders without a
    default are left untouched (matching Commons' ``disableSubstitutionInValues``
    default of leaving unknown variables in place).
    """
    pattern = re.compile(
        re.escape(prefix) + r"(.*?)" + re.escape(suffix),
        flags=re.DOTALL,
    )

    def repl(match: re.Match[str]) -> str:
        body = match.group(1)
        if ":-" in body:
            key, default = body.split(":-", 1)
        else:
            key, default = body, None
        key = key.strip()
        if key in values and values[key] is not None:
            return str(values[key])
        if default is not None:
            return default
        return match.group(0)

    return pattern.sub(repl, template)
