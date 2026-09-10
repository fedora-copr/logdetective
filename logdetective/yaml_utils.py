"""YAML loading helpers."""

import yaml


class DuplicateKeyMapping(dict):
    """Mapping that retains evidence of keys repeated in the YAML source."""

    def __init__(self):
        super().__init__()
        self.duplicate_keys = []


class DuplicateKeySafeLoader(yaml.SafeLoader):  # pylint: disable=too-many-ancestors
    """Safe YAML loader that records duplicate mapping keys."""

    def construct_yaml_map(self, node):
        """Construct a mapping without discarding duplicate-key metadata."""
        mapping = DuplicateKeyMapping()
        yield mapping
        value = self.construct_mapping(node)
        mapping.update(value)
        mapping.duplicate_keys.extend(value.duplicate_keys)

    def construct_mapping(self, node, deep=False):
        self.flatten_mapping(node)
        mapping = DuplicateKeyMapping()
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in mapping
            except TypeError as exc:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found unhashable key",
                    key_node.start_mark,
                ) from exc
            if duplicate:
                mapping.duplicate_keys.append(key)
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


DuplicateKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    DuplicateKeySafeLoader.construct_yaml_map,
)
