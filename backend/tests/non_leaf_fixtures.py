"""
Deterministic synthetic probes for the content-level input suitability gate.

IMPORTANT SCOPE NOTE
--------------------
These images are programmatic illustrations, not photographs. They exercise the gate's
decision branches and the downstream API/UI safety behaviour. They are NOT evidence that
the gate generalises to real portraits, animals, documents or screenshots. The
authoritative non-leaf evidence in this repository is the 1,143 real
``Background_without_leaves`` photographs, evaluated by
``ml/scripts/evaluate_input_gate.py`` and reported in
``docs/audit_reports/non_leaf_input_safety_report.md``.

Each fixture is drawn on a 300x300 canvas and encoded losslessly unless stated otherwise,
so decisions are stable across runs.
"""
import io

from PIL import Image, ImageDraw, ImageFilter

_SIZE = 300


def _encode(image: Image.Image, format: str = "PNG") -> bytes:
    buf = io.BytesIO()
    image.save(buf, format=format)
    return buf.getvalue()


def _blank(background=(240, 240, 238)) -> Image.Image:
    return Image.new("RGB", (_SIZE, _SIZE), background)


def _leaf(draw: ImageDraw.ImageDraw, box, fill=(58, 128, 58), veins=True) -> None:
    draw.ellipse(box, fill=fill)
    if veins:
        x0, y0, x1, y1 = box
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        draw.line([(x0 + 8, cy), (x1 - 8, cy)], fill=(34, 92, 34), width=2)
        for offset in range(-3, 4):
            ax = cx + offset * (x1 - x0) // 10
            draw.line([(ax, y0 + 10), (cx, cy)], fill=(34, 92, 34), width=1)
            draw.line([(ax, y1 - 10), (cx, cy)], fill=(34, 92, 34), width=1)


# ---------------------------------------------------------------------------
# Non-leaf categories
# ---------------------------------------------------------------------------
def create_portrait_bytes() -> bytes:
    """A person: skin tones, hair, clothing, neutral indoor wall. No plant pigment."""
    image = _blank((56, 56, 60))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, _SIZE, 150], fill=(74, 72, 70))       # wall
    draw.rectangle([0, 150, _SIZE, _SIZE], fill=(96, 92, 86))   # lower wall
    draw.rectangle([96, 232, 204, _SIZE], fill=(58, 72, 108))   # shirt
    draw.rectangle([132, 210, 168, 250], fill=(214, 162, 132))  # neck
    draw.ellipse([92, 58, 208, 214], fill=(222, 172, 140))      # face
    draw.ellipse([86, 40, 214, 132], fill=(38, 28, 22))         # hair
    draw.ellipse([122, 130, 140, 144], fill=(32, 28, 26))       # eyes
    draw.ellipse([160, 130, 178, 144], fill=(32, 28, 26))
    draw.arc([132, 152, 168, 182], 200, 340, fill=(150, 92, 82), width=3)  # mouth
    return _encode(image)


def create_animal_bytes() -> bytes:
    """An animal: brown fur, muzzle, ears, neutral background. No plant pigment."""
    image = _blank((206, 202, 192))
    draw = ImageDraw.Draw(image)
    draw.ellipse([48, 120, 252, 290], fill=(146, 100, 60))       # body
    draw.ellipse([96, 56, 204, 178], fill=(158, 110, 66))        # head
    draw.polygon([(104, 70), (86, 24), (138, 52)], fill=(120, 82, 48))  # ears
    draw.polygon([(196, 70), (214, 24), (162, 52)], fill=(120, 82, 48))
    draw.ellipse([126, 92, 142, 106], fill=(28, 24, 22))
    draw.ellipse([158, 92, 174, 106], fill=(28, 24, 22))
    draw.ellipse([138, 124, 162, 146], fill=(52, 42, 36))        # nose
    draw.rectangle([60, 260, 240, 292], fill=(120, 116, 108))    # floor
    return _encode(image)


def create_building_bytes() -> bytes:
    """A building facade: sky, masonry, windows, roof. No plant pigment."""
    image = _blank((124, 174, 224))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 190, _SIZE, _SIZE], fill=(148, 146, 138))  # ground
    draw.rectangle([24, 74, 276, 214], fill=(196, 192, 182))      # facade
    draw.polygon([(16, 74), (150, 28), (284, 74)], fill=(112, 72, 58))  # roof
    for row in range(2):
        for col in range(3):
            x = 48 + col * 76
            y = 96 + row * 56
            draw.rectangle([x, y, x + 44, y + 34], fill=(84, 98, 108))  # windows
            draw.line([(x + 22, y), (x + 22, y + 34)], fill=(150, 150, 150), width=2)
    draw.rectangle([128, 168, 172, 214], fill=(76, 62, 52))       # door
    return _encode(image)


def create_document_bytes() -> bytes:
    """A printed document page on a wooden desk. No plant pigment."""
    image = _blank((112, 86, 62))
    draw = ImageDraw.Draw(image)
    draw.rectangle([28, 20, 272, 280], fill=(250, 249, 245))      # page
    draw.rectangle([48, 40, 200, 52], fill=(56, 56, 60))          # heading
    for row in range(18):
        y = 72 + row * 10
        width = 196 if row % 5 else 130
        draw.rectangle([48, y, 48 + width, y + 4], fill=(96, 96, 100))
    draw.ellipse([196, 218, 252, 262], outline=(176, 46, 46), width=5)  # stamp
    return _encode(image)


def create_object_bytes() -> bytes:
    """A desk object: ceramic mug and notebook on wood. No plant pigment."""
    image = _blank((128, 96, 66))
    draw = ImageDraw.Draw(image)
    for y in range(0, _SIZE, 18):
        draw.line([(0, y), (_SIZE, y + 6)], fill=(116, 86, 58), width=2)  # grain
    draw.rectangle([36, 170, 200, 262], fill=(62, 76, 126))       # notebook
    draw.rectangle([36, 170, 200, 182], fill=(48, 58, 100))
    draw.rounded_rectangle([148, 92, 242, 190], radius=22, fill=(176, 174, 168))  # mug
    draw.ellipse([148, 84, 242, 116], fill=(120, 118, 112))
    draw.arc([216, 108, 272, 172], 270, 90, fill=(176, 174, 168), width=12)  # handle
    return _encode(image)


def create_screenshot_bytes() -> bytes:
    """A UI screenshot with panels and a small green primary button."""
    image = _blank((24, 28, 42))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, _SIZE, 34], fill=(34, 40, 58))          # title bar
    draw.rectangle([14, 44, 138, 268], fill=(38, 44, 62))         # sidebar
    draw.rectangle([152, 44, 286, 158], fill=(42, 48, 68))        # card
    draw.rectangle([152, 172, 286, 268], fill=(42, 48, 68))
    for row in range(10):
        y = 60 + row * 18
        draw.rectangle([24, y, 122, y + 5], fill=(150, 156, 172))
    for row in range(6):
        y = 58 + row * 14
        draw.rectangle([164, y, 266, y + 4], fill=(170, 176, 190))
    draw.rounded_rectangle([164, 204, 224, 228], radius=8, fill=(214, 255, 62))  # ~1.6%
    draw.rectangle([230, 204, 274, 228], fill=(58, 66, 90))
    return _encode(image)


def create_uniform_bytes() -> bytes:
    """A constant-colour canvas (handled by the existing physical-integrity gate)."""
    return _encode(_blank((128, 128, 128)))


# ---------------------------------------------------------------------------
# Plant / specimen edge cases
# ---------------------------------------------------------------------------
def create_leaf_closeup_bytes(color=(58, 128, 58)) -> bytes:
    """A single leaf filling most of the frame on a neutral background."""
    image = _blank((34, 34, 36))
    draw = ImageDraw.Draw(image)
    _leaf(draw, [26, 20, 274, 290], fill=color)
    return _encode(image)


def create_multiple_leaves_bytes() -> bytes:
    """Several leaves in one frame (a legitimate specimen photograph)."""
    image = _blank((40, 40, 42))
    draw = ImageDraw.Draw(image)
    _leaf(draw, [8, 18, 128, 170], fill=(52, 118, 52))
    _leaf(draw, [150, 30, 288, 196], fill=(64, 136, 62))
    _leaf(draw, [58, 176, 232, 294], fill=(70, 142, 66))
    return _encode(image)


def create_complex_background_bytes() -> bytes:
    """A leaf against soil, pebbles and a tool handle (field-style clutter)."""
    image = _blank((118, 86, 58))
    draw = ImageDraw.Draw(image)
    for y in range(0, _SIZE, 12):
        draw.line([(0, y), (_SIZE, y)], fill=(96, 70, 48), width=3)
    for x, y in ((40, 60), (92, 210), (240, 70), (210, 250), (60, 140)):
        draw.ellipse([x, y, x + 22, y + 16], fill=(150, 146, 134))
    draw.rectangle([250, 190, 290, 300], fill=(58, 72, 108))      # tool handle
    _leaf(draw, [46, 40, 214, 250], fill=(60, 126, 58))
    return _encode(image)


def create_sparse_vegetation_bytes() -> bytes:
    """Plant material present but only ~4% of the frame (distant or tiny subject)."""
    image = _blank((200, 196, 190))
    draw = ImageDraw.Draw(image)
    _leaf(draw, [18, 18, 74, 74])
    _leaf(draw, [232, 238, 288, 292])
    return _encode(image)


def create_scattered_vegetation_bytes() -> bytes:
    """Diffuse foliage: several ~1.5% patches, no dominant region (context scene)."""
    image = _blank((198, 196, 192))
    draw = ImageDraw.Draw(image)
    boxes = [
        [12, 12, 56, 56],
        [244, 20, 288, 64],
        [120, 122, 164, 166],
        [24, 238, 68, 282],
        [236, 236, 280, 280],
        [18, 120, 62, 164],
    ]
    for box in boxes:
        _leaf(draw, box, fill=(56, 124, 56), veins=False)
    return _encode(image)


def create_tiny_leaf_bytes() -> bytes:
    """A single, very small leaf (~1.5% of the frame)."""
    image = _blank((204, 202, 198))
    draw = ImageDraw.Draw(image)
    _leaf(draw, [132, 132, 176, 176], veins=False)
    return _encode(image)


def create_dark_leaf_bytes() -> bytes:
    """A leaf photograph underexposed to near-black."""
    image = Image.open(io.BytesIO(create_leaf_closeup_bytes()))
    arr = image.point(lambda value: int(value * 0.10))
    return _encode(arr)


def create_blurry_leaf_bytes() -> bytes:
    """A leaf photographed heavily out of focus."""
    image = Image.open(io.BytesIO(create_leaf_closeup_bytes()))
    return _encode(image.filter(ImageFilter.GaussianBlur(radius=9)))
