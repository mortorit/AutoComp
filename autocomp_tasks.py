from dataclasses import dataclass
from itertools import product, permutations
import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple


COLOR_VOCAB = (
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown",
    "black", "white", "gray", "silver", "gold", "teal", "maroon", "navy",
    "olive", "lime", "cyan", "magenta",
)

SPATIAL_RELATION_VOCAB = (
    "under", "above", "to the left of", "to the right of",
)

SIZE_RELATION_VOCAB = (
    "larger than", "smaller than", "bigger than",
)

SHAPE_VOCAB = (
    "cube", "sphere", "pyramid", "cone", "cylinder",
)

OBJECT_VOCAB = (
    "Sneakers", "Chair", "Hat", "Car", "Lamp", "Glasses", "Bottle", "Desk",
    "Cup", "Cabinet", "Shelf", "Handbag", "Bracelet", "Plate", "Picture",
    "Frame", "Helmet", "Book", "Gloves", "Storage Box", "Boat",
    "Leather Shoes", "Bench", "Potted Plant", "Bowl", "Basin", "Flag",
    "Pillow", "Boots", "Vase", "Microphone", "Necklace", "Ring", "SUV",
    "Wine Glass", "Belt", "Monitor", "Backpack", "Umbrella", "Speaker",
    "Watch", "Tie", "Trash Bin", "Slippers", "Bicycle", "Stool", "Barrel",
    "Bucket", "Van", "Couch", "Sandals", "Basket", "Drum", "Pen", "Pencil",
    "Bus", "High Heels", "Motorcycle", "Guitar", "Carpet", "Cell Phone",
    "Camera", "Truck", "Traffic Cone", "Cymbal", "Lifesaver", "Towel", "Toy",
    "Candle", "Sailboat", "Laptop", "Bed", "Faucet", "Tent", "Mirror",
    "Sink", "Knife", "Paddle", "Pickup Truck", "Fork", "Traffic Sign",
    "Balloon", "Tripod", "Spoon", "Clock", "Pot", "Cake", "Dining Table",
    "Hanger", "Napkin", "Keyboard", "Lantern", "Fan", "Baseball Glove",
    "Airplane", "PC Mouse", "Train", "Luggage", "Nightstand", "Tea Pot",
    "Telephone", "Trolley", "Headphones", "Sports Car", "Dessert", "Scooter",
    "Stroller", "Crane", "Remote", "Refrigerator", "Oven", "Baseball Bat",
    "Jug", "Piano", "Pizza", "Skateboard", "Surfboard", "Gas Stove", "Donut",
    "Bow Tie", "Toilet", "Kite", "Shovel", "Computer Case", "Toilet Paper",
    "Cleaning Products", "Chopsticks", "Microwave", "Baseball",
    "Cutting Board", "Coffee Table", "Side Table", "Scissors", "Marker",
    "Ladder", "Snowboard", "Cookies", "Radiator", "Fire Hydrant",
    "Fire Extinguisher", "Fire Truck", "Billiards", "Converter", "Bathtub",
    "Wheelchair", "Briefcase", "Paint Brush", "Heavy Truck", "Extractor Fan",
    "Extension Cord", "Tennis Racket", "Folder", "Earphone", "Mask", "Kettle",
    "Ship", "Swing", "Coffee Machine", "Slide", "Carriage", "Projector",
    "Frisbee", "Washing Machine", "Dryer", "Printer", "Saxophone",
    "Tissue Box", "Toothbrush", "Ice Cream", "Hot Air Balloon", "Cello",
    "Scale", "Trophy", "Blender", "Wallet", "Purse", "Tape", "Tablet",
    "Cosmetics", "Trumpet", "Golf Ball", "Parking Meter", "Key", "Hurdle",
    "Fishing Rod", "Medal", "Flute", "Brush", "Megaphone", "Broom",
    "Trombone", "Router", "Modem", "Poker Card", "Toaster", "Notepaper",
    "Pliers", "CD", "DVD", "Hammer", "Flask", "Screwdriver", "Soap",
    "Recorder", "Board Eraser", "Tape Measure", "Ruler", "Showerhead",
    "Globe", "Chips", "Stapler", "Formula 1 Car", "Dishwasher", "Hoverboard",
    "Rice Cooker", "Tuba", "Calculator", "Electric Drill", "Hair Dryer",
    "Egg Tart", "Treadmill", "Lighter", "Game Board", "Mop", "Target",
    "Pencil Case", "Binoculars", "Barbell", "Noodles", "Comb", "Dumpling",
    "Chainsaw", "Eraser", "Lipstick", "Cosmetics Mirror",
)

PLURAL_NOUNS = {
    "sneakers", "glasses", "gloves", "leather shoes", "boots", "slippers",
    "sandals", "high heels", "scissors", "chopsticks", "headphones", "pliers",
    "binoculars", "earphone",
}

VOWEL_LETTERS = "aeiou"


@dataclass(frozen=True)
class TaskSpec:
    name: str
    display_name: str
    binding_type: str
    object_type: str
    object_vocab: Tuple[str, ...]
    attribute_vocab: Tuple[str, ...]
    repo_suffix: str

    @property
    def is_attribute_binding(self) -> bool:
        return self.binding_type == "attribute"

    @property
    def is_relational_binding(self) -> bool:
        return self.binding_type == "relational"

    def num_attributes(self, num_objects: int) -> int:
        return num_objects if self.is_attribute_binding else num_objects - 1


TASK_SPECS: Dict[str, TaskSpec] = {
    "color": TaskSpec(
        name="color",
        display_name="Color",
        binding_type="attribute",
        object_type="object",
        object_vocab=OBJECT_VOCAB,
        attribute_vocab=COLOR_VOCAB,
        repo_suffix="C",
    ),
    "shape_color": TaskSpec(
        name="shape_color",
        display_name="Shape-Color",
        binding_type="attribute",
        object_type="shape",
        object_vocab=SHAPE_VOCAB,
        attribute_vocab=COLOR_VOCAB,
        repo_suffix="SC",
    ),
    "position": TaskSpec(
        name="position",
        display_name="Position",
        binding_type="relational",
        object_type="object",
        object_vocab=OBJECT_VOCAB,
        attribute_vocab=SPATIAL_RELATION_VOCAB,
        repo_suffix="P",
    ),
    "relative_size": TaskSpec(
        name="relative_size",
        display_name="Relative Size",
        binding_type="relational",
        object_type="object",
        object_vocab=OBJECT_VOCAB,
        attribute_vocab=SIZE_RELATION_VOCAB,
        repo_suffix="RS",
    ),
}

TASK_ALIASES = {
    "color": "color",
    "shape": "shape_color",
    "shape_color": "shape_color",
    "shape_colour": "shape_color",
    "shapecolor": "shape_color",
    "shapecolour": "shape_color",
    "position": "position",
    "spatial": "position",
    "relative_size": "relative_size",
    "rel_size": "relative_size",
    "relsize": "relative_size",
    "size": "relative_size",
}

TASK_CHOICES = tuple(TASK_SPECS.keys())

CAPTION_SOURCE_ALIASES = {
    "minimal": "handmade",
    "handmade": "handmade",
    "contextual": "llm",
    "llm": "llm",
}

CAPTION_SOURCE_CHOICES = tuple(CAPTION_SOURCE_ALIASES.keys())


def normalize_task_type(task_type: str) -> str:
    key = task_type.strip().lower().replace("-", "_").replace(" ", "_").replace(".", "")
    if key not in TASK_ALIASES:
        raise ValueError(
            f"Unknown task_type '{task_type}'. Expected one of: {', '.join(TASK_CHOICES)}"
        )
    return TASK_ALIASES[key]


def get_task_spec(task_type: str) -> TaskSpec:
    return TASK_SPECS[normalize_task_type(task_type)]


def normalize_caption_source(source: str) -> str:
    key = source.strip().lower().replace("-", "_").replace(" ", "_")
    if key not in CAPTION_SOURCE_ALIASES:
        raise ValueError(
            "Unknown caption source "
            f"'{source}'. Expected one of: {', '.join(CAPTION_SOURCE_CHOICES)}"
        )
    return CAPTION_SOURCE_ALIASES[key]


def caption_file_suffix(source: str) -> str:
    return normalize_caption_source(source)


def get_article(obj: str, next_word: str) -> str:
    if obj.lower() in PLURAL_NOUNS:
        return "a pair of"
    return "an" if next_word.lower().startswith(tuple(VOWEL_LETTERS)) else "a"


def get_item_attributes(item: Dict[str, Any]) -> List[str]:
    for key in ("attributes", "colors", "relations"):
        value = item.get(key)
        if value:
            return list(value)
    raise KeyError("Item is missing an 'attributes', 'colors', or 'relations' field.")


def _first_present(item: Dict[str, Any], field_names: Iterable[str]) -> Any:
    for field in field_names:
        value = item.get(field)
        if value is not None:
            return value
    return None


def get_caption(item: Dict[str, Any], source: str) -> str:
    source = normalize_caption_source(source)
    field_names = (
        ("minimal_caption", "handmade_caption", "caption")
        if source == "handmade"
        else ("contextual_caption", "llm_caption", "caption")
    )
    caption = _first_present(item, field_names)
    if not caption:
        raise KeyError(f"Item is missing a caption for source '{source}'.")
    return caption


def get_image(item: Dict[str, Any], source: str) -> Any:
    source = normalize_caption_source(source)
    field_names = (
        ("minimal_image", "handmade_image", "image")
        if source == "handmade"
        else ("contextual_image", "llm_image", "image")
    )
    return _first_present(item, field_names)


def _unique_permutations(values: Sequence[str]) -> List[Tuple[str, ...]]:
    return list(dict.fromkeys(permutations(values)))


def _replace_elements(caption: str, replacements: Dict[str, str]) -> str:
    result = caption.lower()
    placeholders = {
        original.lower(): f"__AUTOCOMP_PLACEHOLDER_{idx}__"
        for idx, original in enumerate(replacements)
    }
    for original in sorted(placeholders, key=len, reverse=True):
        result = re.sub(
            rf"\b{re.escape(original)}\b",
            placeholders[original],
            result,
            flags=re.IGNORECASE,
        )
    for original, new_value in replacements.items():
        result = result.replace(placeholders[original.lower()], new_value.lower())
    return result


def _fix_attribute_articles(caption: str, objects: Sequence[str], attributes: Sequence[str]) -> str:
    fixed = caption
    for obj, attr in sorted(zip(objects, attributes), key=lambda pair: len(pair[0]), reverse=True):
        article = get_article(obj, attr)
        pattern = rf"\b(?:a pair of|an|a)\s+({re.escape(attr.lower())}\s+{re.escape(obj.lower())})\b"
        fixed = re.sub(pattern, f"{article} " + r"\1", fixed, flags=re.IGNORECASE)
    return fixed


def _fix_relational_articles(caption: str, objects: Sequence[str]) -> str:
    fixed = caption
    for obj in sorted(set(objects), key=len, reverse=True):
        article = get_article(obj, obj)
        pattern = rf"\b(?:a pair of|an|a)\s+({re.escape(obj.lower())})\b"
        fixed = re.sub(pattern, f"{article} " + r"\1", fixed, flags=re.IGNORECASE)
    return fixed


def substitute_caption(
    caption: str,
    task_type: str,
    orig_objects: Sequence[str],
    orig_attributes: Sequence[str],
    new_objects: Sequence[str],
    new_attributes: Sequence[str],
) -> str:
    spec = get_task_spec(task_type)
    replacements: Dict[str, str] = {}
    replacements.update(dict(zip(orig_objects, new_objects)))
    replacements.update(dict(zip(orig_attributes, new_attributes)))
    substituted = _replace_elements(caption, replacements)

    if spec.is_attribute_binding:
        substituted = _fix_attribute_articles(substituted, new_objects, new_attributes)
    else:
        substituted = _fix_relational_articles(substituted, new_objects)

    return substituted[:1].upper() + substituted[1:]


def generate_swap_negatives(item: Dict[str, Any], task_type: str, caption_source: str = "minimal") -> List[str]:
    spec = get_task_spec(task_type)
    caption = get_caption(item, caption_source)
    objects = list(item["objects"])
    attributes = get_item_attributes(item)

    negatives = []
    if spec.is_attribute_binding:
        for attr_perm in _unique_permutations(attributes):
            if attr_perm == tuple(attributes):
                continue
            negatives.append(
                substitute_caption(caption, spec.name, objects, attributes, objects, list(attr_perm))
            )
    else:
        for obj_perm in _unique_permutations(objects):
            if obj_perm == tuple(objects):
                continue
            negatives.append(
                substitute_caption(caption, spec.name, objects, attributes, list(obj_perm), attributes)
            )
    return negatives


def generate_confusion_candidates(
    item: Dict[str, Any],
    task_type: str,
    caption_source: str = "minimal",
) -> Tuple[List[str], int]:
    spec = get_task_spec(task_type)
    caption = get_caption(item, caption_source)
    objects = list(item["objects"])
    attributes = get_item_attributes(item)

    object_configs = list(product(objects, repeat=len(objects)))
    attribute_configs = list(product(attributes, repeat=len(attributes)))
    positive_config = (tuple(objects), tuple(attributes))

    candidates: List[str] = []
    positive_label = -1
    for obj_config, attr_config in product(object_configs, attribute_configs):
        current_config = (tuple(obj_config), tuple(attr_config))
        if current_config == positive_config:
            positive_label = len(candidates)
            candidates.append(caption)
            continue
        candidates.append(
            substitute_caption(
                caption,
                spec.name,
                objects,
                attributes,
                list(obj_config),
                list(attr_config),
            )
        )
    return candidates, positive_label


def generate_confusion_negatives(
    item: Dict[str, Any],
    task_type: str,
    caption_source: str = "minimal",
) -> List[str]:
    candidates, positive_label = generate_confusion_candidates(item, task_type, caption_source)
    if positive_label < 0:
        return candidates
    return candidates[:positive_label] + candidates[positive_label + 1:]
