"""Core utilities for tsl.

This file must not depend on any other bzl files.
"""

def if_google(google_value, oss_value = []):
    """Returns one of the arguments based on the non-configurable build env.

    Specifically, it does not return a `select`, and can be used to e.g.
    compute elements of list attributes.
    """
    _ = (google_value, oss_value)  # buildifier: disable=unused-variable
    return oss_value  # copybara:comment_replace return google_value

def internal_load_visibility(internal_targets):
    """Returns internal_targets in g3, but returns public in OSS."""
    return if_google(internal_targets, ["public"])
