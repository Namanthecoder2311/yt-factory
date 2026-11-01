"""Procedural 3D Wigglo (original mascot) dancing in a meadow, rendered with Blender Cycles.

    blender -b -P kids/wigglo_scene.py -- --out /tmp/frames --seconds 10 --res 1280x720

Everything is built from code (no downloaded assets), so the character is ours and identical in
every episode. Motion: inchworm wiggle along the body, head bob and tilt, blinking, antenna sway,
pom-pom bounce, slow camera orbit.
"""
import math
import sys
import time

import bpy

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
opts = dict(zip(argv[::2], argv[1::2]))
OUT = opts.get("--out", "/tmp/frames")
SECONDS = float(opts.get("--seconds", 10))
RES_X, RES_Y = (int(v) for v in opts.get("--res", "1280x720").split("x"))
FPS = 24
FRAMES = int(SECONDS * FPS)

# ------------------------------------------------------------------ scene setup
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.render.resolution_x, scene.render.resolution_y = RES_X, RES_Y
scene.render.fps = FPS
scene.frame_start, scene.frame_end = 1, FRAMES
scene.render.filepath = OUT.rstrip("/") + "/f_"
scene.render.image_settings.file_format = "PNG"
scene.cycles.samples = int(opts.get("--samples", 32))
scene.cycles.use_denoising = True
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"

prefs = bpy.context.preferences.addons["cycles"].preferences
for backend in ("OPTIX", "CUDA"):
    try:
        prefs.compute_device_type = backend
        prefs.get_devices()
        gpus = [d for d in prefs.devices if d.type == backend]
        if gpus:
            for d in prefs.devices:
                d.use = d.type == backend
            scene.cycles.device = "GPU"
            print(f"render device: {backend} x{len(gpus)}", flush=True)
            break
    except Exception as e:  # backend not available on this machine
        print("no", backend, e, flush=True)
else:
    print("render device: CPU", flush=True)


def material(name, color, rough=0.45, sss=0.0, emit=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    p = m.node_tree.nodes["Principled BSDF"]
    p.inputs["Base Color"].default_value = (*color, 1)
    p.inputs["Roughness"].default_value = rough
    if sss:
        p.inputs["Subsurface Weight"].default_value = sss
        p.inputs["Subsurface Radius"].default_value = (0.4, 0.3, 0.2)
    if emit:
        p.inputs["Emission Color"].default_value = (*color, 1)
        p.inputs["Emission Strength"].default_value = emit
    return m


def attach(child, parent):
    """Parent while keeping the child's world placement (positions below are world-space)."""
    bpy.context.view_layer.update()
    mw = child.matrix_world.copy()
    child.parent = parent
    child.matrix_world = mw


def sphere(name, loc, radius, mat, scale=(1, 1, 1), parent=None):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=radius, location=loc)
    o = bpy.context.object
    o.name = name
    o.scale = scale
    bpy.ops.object.shade_smooth()
    o.data.materials.append(mat)
    if parent:
        attach(o, parent)
    return o


def empty(name, loc, parent=None):
    o = bpy.data.objects.new(name, None)
    bpy.context.collection.objects.link(o)
    o.location = loc
    if parent:
        attach(o, parent)
    return o


# palette (matches the published DP/banner)
LIME = material("lime", (0.45, 0.80, 0.22), 0.5, sss=0.15)
BELLY = material("belly", (0.70, 0.90, 0.45), 0.5, sss=0.1)
WHITE = material("white", (0.97, 0.97, 0.97), 0.2)
PUPIL = material("pupil", (0.18, 0.09, 0.04), 0.15)
SHINE = material("shine", (1, 1, 1), 0.1, emit=4)
PINK = material("cheek", (0.98, 0.55, 0.60), 0.6)
MOUTH = material("mouth", (0.55, 0.08, 0.12), 0.4)
ORANGE = material("beanie", (0.98, 0.50, 0.12), 0.8)
YELLOW = material("pompom", (1.0, 0.85, 0.25), 0.9)
GRASS = material("grass", (0.30, 0.65, 0.20), 0.9)

# ------------------------------------------------------------------ world, ground, lights
world = bpy.data.worlds.new("sky")
scene.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.55, 0.78, 1.0, 1)
world.node_tree.nodes["Background"].inputs[1].default_value = 0.9

bpy.ops.mesh.primitive_plane_add(size=60, location=(0, 0, 0))
bpy.context.object.data.materials.append(GRASS)

bpy.ops.object.light_add(type="SUN", location=(4, -4, 8))
sun = bpy.context.object
sun.data.energy = 3.5
sun.rotation_euler = (math.radians(50), 0, math.radians(35))
bpy.ops.object.light_add(type="AREA", location=(-3, -5, 4))
fill = bpy.context.object
fill.data.energy = 400
fill.data.size = 5
fill.rotation_euler = (math.radians(60), 0, math.radians(-30))

flower_mats = [material(f"petal{i}", c, 0.6) for i, c in enumerate(
    [(1, 0.45, 0.6), (1, 0.85, 0.2), (0.6, 0.5, 1), (1, 1, 1), (1, 0.55, 0.25)])]
for i in range(70):  # deterministic flower field
    a = i * 2.39996
    r = 3 + (i % 17) * 0.9
    x, y = r * math.cos(a), r * math.sin(a) + 4
    sphere(f"flower{i}", (x, y, 0.12), 0.12 + (i % 3) * 0.03, flower_mats[i % 5], (1, 1, 0.6))

# ------------------------------------------------------------------ Wigglo
root = empty("Wigglo", (0, 0, 0))
SEG = 5
segments = []
for i in range(SEG):  # body, tail -> neck
    r = 0.36 + 0.05 * i
    s = sphere(f"seg{i}", (-1.9 + i * 0.52, 0, r), r, LIME, parent=root)
    sphere(f"belly{i}", (-1.9 + i * 0.52, -r * 0.55, r * 0.8), r * 0.62, BELLY, (1, 0.6, 1), parent=s)
    for side in (-1, 1):
        sphere(f"foot{i}{side}", (-1.9 + i * 0.52, side * r * 0.55, 0.08), 0.09, LIME, parent=s)
    segments.append(s)

head = sphere("head", (0.85, 0, 1.05), 0.62, LIME, parent=root)
for side in (-1, 1):  # eyes, pupils, highlights, cheeks
    eye = sphere(f"eye{side}", (1.27, side * 0.24, 1.18), 0.17, WHITE, (0.7, 1, 1.15), parent=head)
    sphere(f"pupil{side}", (1.37, side * 0.24, 1.17), 0.11, PUPIL, (0.6, 1, 1.1), parent=eye)
    sphere(f"shine{side}", (1.43, side * 0.21, 1.24), 0.035, SHINE, parent=eye)
    sphere(f"cheek{side}", (1.22, side * 0.43, 0.92), 0.11, PINK, (0.5, 1, 0.7), parent=head)
mouth = sphere("mouth", (1.39, 0, 0.84), 0.11, MOUTH, (0.4, 1.2, 0.55), parent=head)
beanie = sphere("beanie", (0.82, 0, 1.45), 0.56, ORANGE, (1.05, 1.05, 0.62), parent=head)
pom = sphere("pom", (0.82, 0, 1.86), 0.14, ORANGE, parent=beanie)
antennae = []
for side in (-1, 1):
    pivot = empty(f"ant{side}", (0.75, side * 0.28, 1.6), parent=head)
    bpy.ops.mesh.primitive_cylinder_add(radius=0.03, depth=0.5, location=(0.75, side * 0.36, 1.85),
                                        rotation=(side * math.radians(-18), 0, 0))
    stalk = bpy.context.object
    stalk.data.materials.append(LIME)
    attach(stalk, pivot)
    sphere(f"antpom{side}", (0.75, side * 0.45, 2.12), 0.12, YELLOW, parent=pivot)
    antennae.append(pivot)
eyes = [bpy.data.objects["eye-1"], bpy.data.objects["eye1"]]

# ------------------------------------------------------------------ animation
# Offsets are added to each part's rest pose (local space after parenting).
bpy.context.view_layer.update()
rest = {o.name: (o.location.copy(), o.scale.copy()) for o in [*segments, head, pom, mouth, *eyes]}
BLINKS = {int(f * FPS) for f in (2.2, 5.1, 7.6, 9.3) if f < SECONDS}
for f in range(1, FRAMES + 1):
    t = (f - 1) / FPS
    beat = t * 2.0 * math.pi * 1.0                     # one wiggle per second (song tempo 120)
    for i, s in enumerate(segments):
        loc, scl = rest[s.name]
        lift = max(0.0, math.sin(beat - i * 0.9)) * 0.22
        s.location = (loc.x, loc.y, loc.z + lift)
        s.scale = (scl.x * (1 + lift * 0.4), scl.y * (1 - lift * 0.15), scl.z * (1 - lift * 0.2))
        s.keyframe_insert("location", frame=f)
        s.keyframe_insert("scale", frame=f)
    loc, _ = rest[head.name]
    head.location = (loc.x, loc.y, loc.z + max(0.0, math.sin(beat - SEG * 0.9)) * 0.18)
    head.rotation_euler = (math.sin(beat * 0.5) * 0.12, math.sin(beat) * 0.05, math.sin(beat * 0.5) * 0.15)
    head.keyframe_insert("location", frame=f)
    head.keyframe_insert("rotation_euler", frame=f)
    for k, a in enumerate(antennae):
        a.rotation_euler = (math.sin(beat * 1.5 + k) * 0.25, math.sin(beat + k) * 0.15, 0)
        a.keyframe_insert("rotation_euler", frame=f)
    loc, _ = rest[pom.name]
    pom.location = (loc.x, loc.y, loc.z + abs(math.sin(beat)) * 0.1)
    pom.keyframe_insert("location", frame=f)
    closed = any(abs(f - b) <= 2 for b in BLINKS)
    for e in eyes:
        _, scl = rest[e.name]
        e.scale = (scl.x, scl.y, scl.z * (0.1 if closed else 1))
        e.keyframe_insert("scale", frame=f)
    _, scl = rest[mouth.name]
    mouth.scale = (scl.x, scl.y, scl.z * (1 + 0.6 * max(0.0, math.sin(beat * 2))))  # "singing"
    mouth.keyframe_insert("scale", frame=f)

# camera: slow orbit, kid's eye level
bpy.ops.object.camera_add()
cam = bpy.context.object
scene.camera = cam
cam.data.lens = 40
target = empty("target", (0, 0, 0.9))
track = cam.constraints.new("TRACK_TO")
track.target = target
for f in (1, FRAMES):
    ang = math.radians(-70 + 40 * (f - 1) / max(1, FRAMES - 1))
    cam.location = (6.2 * math.cos(ang), 6.2 * math.sin(ang), 1.8)
    cam.keyframe_insert("location", frame=f)

# ------------------------------------------------------------------ render
t0 = time.time()
bpy.ops.render.render(animation=True)
spent = time.time() - t0
print(f"RENDER_STATS frames={FRAMES} seconds={spent:.1f} per_frame={spent / FRAMES:.2f}", flush=True)
