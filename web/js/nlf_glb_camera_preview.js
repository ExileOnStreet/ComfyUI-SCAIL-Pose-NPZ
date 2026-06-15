import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const NODE_NAMES = new Set([
    "PreviewNLFPoseGLBWithCamera",
    "PreviewWorldNLFPoseWithCamera",
]);
// Preview camera works in scene-sized GLB units; downstream converters can
// multiply translation by this scale to recover NLF world units (typically mm).
const CAMERA_UNIT_SCALE_TO_NLF = 1000.0;
const WORLD_PERSON_COLORS = [0x60a5fa, 0xf97316, 0x34d399, 0xf472b6, 0xfacc15, 0x93c5fd];
const GRID_BASE_SIZE = 8;

// Adapted interaction/data-flow ideas from:
// - ComfyUI_Rabbit-Camera-Perspective (custom DOM widget lifecycle)
// - ComfyUI-qwenmultiangle (camera state sync between 3D viewport and node widgets)
let threeDepsPromise = null;

function loadThreeDeps() {
    if (!threeDepsPromise) {
        threeDepsPromise = Promise.all([
            import("https://esm.sh/three@0.160.0"),
            import("https://esm.sh/three@0.160.0/examples/jsm/controls/OrbitControls.js"),
            import("https://esm.sh/three@0.160.0/examples/jsm/loaders/GLTFLoader.js"),
        ]).then(([THREE, controlsMod, loaderMod]) => {
            return {
                THREE,
                OrbitControls: controlsMod.OrbitControls,
                GLTFLoader: loaderMod.GLTFLoader,
            };
        });
    }
    return threeDepsPromise;
}

function matrix4ToRowMajorArray(elements) {
    return [
        [elements[0], elements[4], elements[8], elements[12]],
        [elements[1], elements[5], elements[9], elements[13]],
        [elements[2], elements[6], elements[10], elements[14]],
        [elements[3], elements[7], elements[11], elements[15]],
    ];
}

function roundNumber(value, digits = 6) {
    if (!Number.isFinite(value)) {
        return value;
    }
    const factor = 10 ** digits;
    return Math.round(value * factor) / factor;
}

function roundNumbersDeep(value, digits = 6) {
    if (typeof value === "number") {
        return roundNumber(value, digits);
    }
    if (Array.isArray(value)) {
        return value.map((item) => roundNumbersDeep(item, digits));
    }
    if (value && typeof value === "object") {
        const out = {};
        for (const [k, v] of Object.entries(value)) {
            out[k] = roundNumbersDeep(v, digits);
        }
        return out;
    }
    return value;
}

function buildViewUrl(item) {
    const params = new URLSearchParams({
        filename: String(item.filename || ""),
        subfolder: String(item.subfolder || ""),
        type: String(item.type || "output"),
    });
    return api.apiURL(`/view?${params.toString()}`);
}

function createAxisLabelSprite(THREE, text, colorHex) {
    const canvas = document.createElement("canvas");
    canvas.width = 128;
    canvas.height = 64;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
        return null;
    }

    const color = `#${Number(colorHex).toString(16).padStart(6, "0")}`;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "rgba(2, 6, 23, 0.72)";
    ctx.strokeStyle = "rgba(148, 163, 184, 0.45)";
    ctx.lineWidth = 2;
    const w = 92;
    const h = 40;
    const x = (canvas.width - w) / 2;
    const y = (canvas.height - h) / 2;
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, 8);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = color;
    ctx.font = "bold 28px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, canvas.width / 2, canvas.height / 2 + 1);

    const texture = new THREE.CanvasTexture(canvas);
    texture.needsUpdate = true;
    texture.colorSpace = THREE.SRGBColorSpace;

    const material = new THREE.SpriteMaterial({
        map: texture,
        transparent: true,
        depthWrite: false,
        depthTest: false,
    });
    const sprite = new THREE.Sprite(material);
    sprite.scale.set(0.34, 0.17, 1.0);
    return sprite;
}

function createSceneAxisGroup(THREE) {
    const group = new THREE.Group();
    const axes = new THREE.AxesHelper(1.0);
    group.add(axes);

    const xLabel = createAxisLabelSprite(THREE, "+X", 0xff4d4f);
    const yLabel = createAxisLabelSprite(THREE, "+Y", 0x22c55e);
    const zLabel = createAxisLabelSprite(THREE, "+Z", 0x3b82f6);

    if (xLabel) {
        xLabel.position.set(1.18, 0, 0);
        group.add(xLabel);
    }
    if (yLabel) {
        yLabel.position.set(0, 1.18, 0);
        group.add(yLabel);
    }
    if (zLabel) {
        zLabel.position.set(0, 0, 1.18);
        group.add(zLabel);
    }

    group.position.set(0, 0, 0);
    group.scale.setScalar(1.0);
    return group;
}

async function createViewer(node, cameraInfoWidget) {
    const { THREE, OrbitControls, GLTFLoader } = await loadThreeDeps();

    const container = document.createElement("div");
    container.style.cssText = "position:relative;width:100%;height:100%;min-height:420px;border:1px solid #2c2f36;border-radius:8px;overflow:hidden;background:#0d1117;";

    const toolbar = document.createElement("div");
    toolbar.style.cssText = "position:absolute;top:8px;left:8px;right:8px;z-index:2;display:flex;gap:8px;align-items:center;";

    const resetBtn = document.createElement("button");
    resetBtn.textContent = "Reset View";
    resetBtn.style.cssText = "padding:4px 8px;border:1px solid #444;border-radius:6px;background:#1f2937;color:#e5e7eb;cursor:pointer;font-size:12px;";

    const status = document.createElement("div");
    status.textContent = "Run node to load preview";
    status.style.cssText = "margin-left:auto;padding:4px 8px;border-radius:6px;background:rgba(15,23,42,.85);color:#93c5fd;font-size:12px;";

    const canvasWrap = document.createElement("div");
    canvasWrap.style.cssText = "position:absolute;inset:0;";

    const controlPanel = document.createElement("div");
    controlPanel.style.cssText = "position:absolute;left:8px;right:8px;bottom:8px;z-index:2;display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:8px;border:1px solid #334155;border-radius:8px;background:rgba(2,6,23,.82);";

    function createRangeRow(label, min, max, step, value) {
        const row = document.createElement("div");
        row.style.cssText = "display:flex;align-items:center;gap:8px;min-width:0;";

        const name = document.createElement("span");
        name.textContent = label;
        name.style.cssText = "width:52px;flex:0 0 52px;color:#cbd5e1;font-size:11px;";

        const input = document.createElement("input");
        input.type = "range";
        input.min = String(min);
        input.max = String(max);
        input.step = String(step);
        input.value = String(value);
        input.style.cssText = "flex:1 1 auto;min-width:0;";

        const val = document.createElement("span");
        val.textContent = String(value);
        val.style.cssText = "width:58px;flex:0 0 58px;text-align:right;color:#93c5fd;font-size:11px;";

        row.appendChild(name);
        row.appendChild(input);
        row.appendChild(val);
        return { row, input, val };
    }

    const fovRow = createRangeRow("FOV", 20, 120, 1, 55);
    const nearRow = createRangeRow("Near", 0.01, 10, 0.01, 0.01);
    const farRow = createRangeRow("Far", 50, 20000, 10, 10000);

    const animPlayBtn = document.createElement("button");
    animPlayBtn.textContent = "Pause";
    animPlayBtn.style.cssText = "padding:4px 8px;border:1px solid #444;border-radius:6px;background:#1f2937;color:#e5e7eb;cursor:pointer;font-size:12px;";

    const speedRow = createRangeRow("Speed", 0.0, 3.0, 0.05, 1.0);
    const timeRow = createRangeRow("Time", 0.0, 0.0, 0.001, 0.0);
    timeRow.input.disabled = true;

    const animRow = document.createElement("div");
    animRow.style.cssText = "display:flex;align-items:center;gap:8px;grid-column:1 / -1;";
    animRow.appendChild(animPlayBtn);
    animRow.appendChild(speedRow.row);
    animRow.appendChild(timeRow.row);

    controlPanel.appendChild(fovRow.row);
    controlPanel.appendChild(nearRow.row);
    controlPanel.appendChild(farRow.row);
    controlPanel.appendChild(animRow);

    toolbar.appendChild(resetBtn);
    toolbar.appendChild(status);
    container.appendChild(canvasWrap);
    container.appendChild(toolbar);
    container.appendChild(controlPanel);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.domElement.style.cssText = "width:100%;height:100%;display:block;";
    canvasWrap.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b1220);

    const camera = new THREE.PerspectiveCamera(55, 1, 0.01, 10000);
    camera.position.set(0, 1.2, 4.0);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.set(0, 1.0, 0);

    const hemi = new THREE.HemisphereLight(0xffffff, 0x223344, 0.9);
    scene.add(hemi);
    const dir = new THREE.DirectionalLight(0xffffff, 1.0);
    dir.position.set(4, 8, 4);
    scene.add(dir);
    const grid = new THREE.GridHelper(GRID_BASE_SIZE, 32, 0x3b82f6, 0x1f2937);
    grid.position.y = 0;
    if (grid.material) {
        if (Array.isArray(grid.material)) {
            for (const mat of grid.material) {
                mat.transparent = true;
                mat.opacity = 0.92;
            }
        } else {
            grid.material.transparent = true;
            grid.material.opacity = 0.92;
        }
    }
    scene.add(grid);
    const sceneAxisGroup = createSceneAxisGroup(THREE);
    scene.add(sceneAxisGroup);

    const viewfinderOverlay = document.createElement("div");
    viewfinderOverlay.style.cssText = "position:absolute;z-index:1;pointer-events:none;border:2px solid rgba(245,158,11,.95);border-radius:4px;box-shadow:0 0 0 1px rgba(15,23,42,.9) inset,0 0 14px rgba(245,158,11,.2);";

    const viewfinderLabel = document.createElement("div");
    viewfinderLabel.style.cssText = "position:absolute;left:0;top:-24px;padding:2px 6px;border-radius:4px;background:rgba(2,6,23,.82);border:1px solid #334155;color:#fbbf24;font-size:11px;white-space:nowrap;";
    viewfinderOverlay.appendChild(viewfinderLabel);
    canvasWrap.appendChild(viewfinderOverlay);

    const loader = new GLTFLoader();
    let currentObject = null;
    let mixer = null;
    let animationClips = [];
    let animationDuration = 0;
    let isAnimationPlaying = true;
    let isScrubbing = false;
    let playbackSpeed = 1.0;
    let frameWidth = 2.0;
    let frameHeight = 2.0;
    let renderFrameRect = { x: 0, y: 0, w: 64, h: 64 };
    let worldFrames = [];
    let worldFps = 24.0;
    let worldCurrentTime = 0.0;
    let worldCurrentFrameIndex = -1;
    let worldJointRadius = 24.0;
    let worldEdgeRadius = 14.0;
    let rafId = null;
    const clock = new THREE.Clock();

    function clearCurrentSceneObject() {
        if (mixer) {
            mixer.stopAllAction();
            if (currentObject) {
                mixer.uncacheRoot(currentObject);
            }
            mixer = null;
            animationClips = [];
            animationDuration = 0;
        }
        if (currentObject) {
            scene.remove(currentObject);
            disposeObject3D(currentObject);
            currentObject = null;
        }
    }

    function updateCameraFrameOverlay() {
        const fw = Math.max(1.0, frameWidth);
        const fh = Math.max(1.0, frameHeight);
        const aspect = fw / fh;

        const viewportW = Math.max(64, canvasWrap.clientWidth || 64);
        const viewportH = Math.max(64, canvasWrap.clientHeight || 64);

        let drawW = viewportW;
        let drawH = drawW / aspect;
        if (drawH > viewportH) {
            drawH = viewportH;
            drawW = drawH * aspect;
        }

        const left = Math.round((viewportW - drawW) * 0.5);
        const top = Math.round((viewportH - drawH) * 0.5);
        const width = Math.max(1, Math.round(drawW));
        const height = Math.max(1, Math.round(drawH));

        renderFrameRect = {
            x: left,
            y: top,
            w: width,
            h: height,
        };

        viewfinderOverlay.style.left = `${left}px`;
        viewfinderOverlay.style.top = `${top}px`;
        viewfinderOverlay.style.width = `${width}px`;
        viewfinderOverlay.style.height = `${height}px`;
        viewfinderLabel.textContent = `Frame ${Math.round(fw)} x ${Math.round(fh)}`;
    }

    function setStatus(text, isError = false) {
        status.textContent = text;
        status.style.color = isError ? "#fca5a5" : "#93c5fd";
    }

    function applyCameraInfoToWidget() {
        if (!cameraInfoWidget) {
            return;
        }
        camera.updateMatrixWorld(true);
        const payload = {
            camera_to_world: matrix4ToRowMajorArray(camera.matrixWorld.elements),
            camera_unit_scale: CAMERA_UNIT_SCALE_TO_NLF,
            fov_degrees: camera.fov,
            aspect: camera.aspect,
            near: camera.near,
            far: camera.far,
            target: [controls.target.x, controls.target.y, controls.target.z],
            frame_width: frameWidth,
            frame_height: frameHeight,
        };
        const text = JSON.stringify(roundNumbersDeep(payload, 6));
        cameraInfoWidget.value = text;
        if (typeof cameraInfoWidget.callback === "function") {
            cameraInfoWidget.callback(text);
        }
    }

    function syncCameraBarsFromCamera() {
        fovRow.input.value = String(camera.fov);
        fovRow.val.textContent = camera.fov.toFixed(1);
        nearRow.input.value = String(camera.near);
        nearRow.val.textContent = camera.near.toFixed(2);
        farRow.input.value = String(camera.far);
        farRow.val.textContent = camera.far.toFixed(0);
    }

    function syncAnimBars() {
        speedRow.val.textContent = playbackSpeed.toFixed(2);
        speedRow.input.value = String(playbackSpeed);
        timeRow.val.textContent = timeRow.input.value;
        animPlayBtn.textContent = isAnimationPlaying ? "Pause" : "Play";
    }

    function applyWidgetCameraInfoToView() {
        if (!cameraInfoWidget || !cameraInfoWidget.value) {
            return false;
        }
        try {
            const data = JSON.parse(String(cameraInfoWidget.value));
            let applied = false;
            if (Array.isArray(data.camera_to_world) && data.camera_to_world.length === 4) {
                const m = data.camera_to_world;
                const mat = new THREE.Matrix4();
                mat.set(
                    m[0][0], m[0][1], m[0][2], m[0][3],
                    m[1][0], m[1][1], m[1][2], m[1][3],
                    m[2][0], m[2][1], m[2][2], m[2][3],
                    m[3][0], m[3][1], m[3][2], m[3][3],
                );
                const pos = new THREE.Vector3();
                const quat = new THREE.Quaternion();
                const scl = new THREE.Vector3();
                mat.decompose(pos, quat, scl);
                camera.position.copy(pos);
                camera.quaternion.copy(quat);
                applied = true;
            }
            if (typeof data.fov_degrees === "number") {
                camera.fov = data.fov_degrees;
                camera.updateProjectionMatrix();
            }
            if (typeof data.near === "number") {
                camera.near = Math.max(0.001, Number(data.near));
            }
            if (typeof data.far === "number") {
                camera.far = Math.max(camera.near + 0.001, Number(data.far));
            }
            camera.updateProjectionMatrix();
            if (Array.isArray(data.target) && data.target.length === 3) {
                controls.target.set(Number(data.target[0]), Number(data.target[1]), Number(data.target[2]));
            }
            if (typeof data.frame_width === "number") {
                frameWidth = Math.max(0.1, Number(data.frame_width));
            }
            if (typeof data.frame_height === "number") {
                frameHeight = Math.max(1.0, Number(data.frame_height));
            } else if (typeof data.frame_length === "number") {
                // Backward compatibility for older saved camera JSON.
                frameHeight = Math.max(1.0, Number(data.frame_length));
            }
            controls.update();
            updateCameraFrameOverlay();
            syncCameraBarsFromCamera();
            return applied;
        } catch (e) {
            // Keep viewport usable if widget contains user-edited invalid JSON.
            return false;
        }
    }

    function fitObjectToView(obj) {
        const box = new THREE.Box3().setFromObject(obj);
        if (!box.isEmpty()) {
            const center = box.getCenter(new THREE.Vector3());

            const size = box.getSize(new THREE.Vector3());
            const radius = Math.max(size.length() * 0.5, 0.5);
            controls.target.copy(center);

            camera.near = Math.max(radius / 500, 0.01);
            camera.far = Math.max(radius * 100, 100.0);
            camera.position.set(
                center.x + radius * 0.2,
                center.y + radius * 0.5,
                center.z + radius * 2.2,
            );
            camera.updateProjectionMatrix();
            controls.update();
            syncCameraBarsFromCamera();
            applyCameraInfoToWidget();
        }
    }

    function alignGridToBox(box) {
        if (box.isEmpty()) {
            return;
        }

        const size = box.getSize(new THREE.Vector3());
        const footprint = Math.max(1.0, size.x, size.z);
        const scaleXZ = Math.max(1.0, footprint / GRID_BASE_SIZE);

        grid.scale.set(scaleXZ, 1.0, scaleXZ);
        grid.position.y = box.min.y;
        grid.visible = true;
        grid.updateMatrixWorld(true);
        const axisScale = THREE.MathUtils.clamp(footprint * 0.2, 1.0, 3000.0);
        sceneAxisGroup.scale.setScalar(axisScale);
        sceneAxisGroup.updateMatrixWorld(true);
        updateCameraFrameOverlay();
    }

    function alignGridToObjectFloor(obj) {
        obj.updateMatrixWorld(true);
        const box = new THREE.Box3().setFromObject(obj);
        alignGridToBox(box);
    }

    function disposeObject3D(root) {
        root.traverse((child) => {
            if (child.geometry && typeof child.geometry.dispose === "function") {
                child.geometry.dispose();
            }
            if (child.material) {
                if (Array.isArray(child.material)) {
                    for (const mat of child.material) {
                        if (mat && typeof mat.dispose === "function") {
                            mat.dispose();
                        }
                    }
                } else if (typeof child.material.dispose === "function") {
                    child.material.dispose();
                }
            }
        });
    }

    function loadGlb(url) {
        setStatus("Loading GLB...");
        loader.load(
            url,
            (gltf) => {
                worldFrames = [];
                worldCurrentTime = 0.0;
                worldCurrentFrameIndex = -1;
                clearCurrentSceneObject();
                currentObject = gltf.scene;
                scene.add(currentObject);
                alignGridToObjectFloor(currentObject);
                const appliedCameraInfo = applyWidgetCameraInfoToView();
                if (!appliedCameraInfo) {
                    fitObjectToView(currentObject);
                }

                animationClips = Array.isArray(gltf.animations) ? gltf.animations : [];
                if (animationClips.length > 0) {
                    mixer = new THREE.AnimationMixer(currentObject);
                    animationDuration = 0;
                    for (const clip of animationClips) {
                        const action = mixer.clipAction(clip);
                        action.reset();
                        action.play();
                        animationDuration = Math.max(animationDuration, Number(clip.duration) || 0);
                    }
                    isAnimationPlaying = true;
                    isScrubbing = false;
                    timeRow.input.disabled = false;
                    timeRow.input.min = "0";
                    timeRow.input.max = String(Math.max(animationDuration, 0.001));
                    timeRow.input.step = "0.001";
                    timeRow.input.value = "0.000";
                    timeRow.val.textContent = "0.000";
                    setStatus(`GLB loaded. ${animationClips.length} animation clip(s) playing.`);
                } else {
                    timeRow.input.disabled = true;
                    timeRow.input.min = "0";
                    timeRow.input.max = "0";
                    timeRow.input.value = "0.000";
                    timeRow.val.textContent = "0.000";
                    setStatus("GLB loaded (no animation clips). Drag to orbit camera.");
                }
                syncAnimBars();
            },
            undefined,
            (err) => {
                console.error("Failed to load GLB preview", err);
                setStatus("Failed to load GLB preview", true);
            }
        );
    }

    function createCylinderBetween(startArr, endArr, radius, colorHex) {
        const start = new THREE.Vector3(Number(startArr[0]), Number(startArr[1]), Number(startArr[2]));
        const end = new THREE.Vector3(Number(endArr[0]), Number(endArr[1]), Number(endArr[2]));
        if (!Number.isFinite(start.x) || !Number.isFinite(start.y) || !Number.isFinite(start.z)) {
            return null;
        }
        if (!Number.isFinite(end.x) || !Number.isFinite(end.y) || !Number.isFinite(end.z)) {
            return null;
        }

        const dir = new THREE.Vector3().subVectors(end, start);
        const len = dir.length();
        if (len < 1e-6) {
            return null;
        }

        const geometry = new THREE.CylinderGeometry(radius, radius, len, 10, 1, false);
        const material = new THREE.MeshStandardMaterial({ color: colorHex, roughness: 0.55, metalness: 0.05 });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.copy(start).add(end).multiplyScalar(0.5);
        mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
        return mesh;
    }

    function buildWorldFrameGroup(framePeople) {
        const group = new THREE.Group();
        const sphereSegments = 10;

        for (let personIdx = 0; personIdx < framePeople.length; personIdx++) {
            const person = framePeople[personIdx] || {};
            const points = Array.isArray(person.points) ? person.points : [];
            const edges = Array.isArray(person.edges) ? person.edges : [];
            const colorHex = WORLD_PERSON_COLORS[personIdx % WORLD_PERSON_COLORS.length];

            for (const p of points) {
                if (!Array.isArray(p) || p.length !== 3) {
                    continue;
                }
                const x = Number(p[0]);
                const y = Number(p[1]);
                const z = Number(p[2]);
                if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
                    continue;
                }

                const geometry = new THREE.SphereGeometry(worldJointRadius, sphereSegments, sphereSegments);
                const material = new THREE.MeshStandardMaterial({ color: colorHex, roughness: 0.45, metalness: 0.08 });
                const sphere = new THREE.Mesh(geometry, material);
                sphere.position.set(x, y, z);
                group.add(sphere);
            }

            for (const e of edges) {
                if (!Array.isArray(e) || e.length !== 2) {
                    continue;
                }
                const a = Number(e[0]);
                const b = Number(e[1]);
                if (!Number.isInteger(a) || !Number.isInteger(b)) {
                    continue;
                }
                if (a < 0 || b < 0 || a >= points.length || b >= points.length) {
                    continue;
                }
                const cyl = createCylinderBetween(points[a], points[b], worldEdgeRadius, colorHex);
                if (cyl) {
                    group.add(cyl);
                }
            }
        }

        return group;
    }

    function setWorldFrame(frameIndex, fitOnFirstFrame = false) {
        if (!Array.isArray(worldFrames) || worldFrames.length === 0) {
            return;
        }
        const clamped = Math.max(0, Math.min(worldFrames.length - 1, Math.floor(frameIndex)));
        if (clamped === worldCurrentFrameIndex) {
            return;
        }

        if (currentObject) {
            scene.remove(currentObject);
            disposeObject3D(currentObject);
            currentObject = null;
        }

        const framePeople = Array.isArray(worldFrames[clamped]) ? worldFrames[clamped] : [];
        currentObject = buildWorldFrameGroup(framePeople);
        scene.add(currentObject);
        const frameBox = new THREE.Box3().setFromObject(currentObject);
        if (!frameBox.isEmpty()) {
            const size = frameBox.getSize(new THREE.Vector3());
            const span = Math.max(1.0, size.x, size.y, size.z);
            const axisScale = THREE.MathUtils.clamp(span * 0.2, 1.0, 3000.0);
            sceneAxisGroup.scale.setScalar(axisScale);
            sceneAxisGroup.updateMatrixWorld(true);
        }

        if (fitOnFirstFrame) {
            const appliedCameraInfo = applyWidgetCameraInfoToView();
            if (!appliedCameraInfo) {
                fitObjectToView(currentObject);
            }
        }

        worldCurrentFrameIndex = clamped;
    }

    function loadWorldPreview(item) {
        clearCurrentSceneObject();
        worldFrames = Array.isArray(item.frames) ? item.frames : [];
        worldFps = Math.max(0.1, Number(item.fps) || 24.0);
        worldJointRadius = Math.max(0.01, Number(item.joint_radius) || 24.0);
        worldEdgeRadius = Math.max(0.01, Number(item.edge_radius) || Number(item.cylinder_radius) || 14.0);
        worldCurrentTime = 0.0;
        worldCurrentFrameIndex = -1;

        // For world-joint preview, keep the floor grid on the global plane instead
        // of snapping to the lowest joint point.
        grid.position.y = 0;
        grid.scale.set(1.0, 1.0, 1.0);
        grid.visible = true;
        grid.updateMatrixWorld(true);

        if (worldFrames.length === 0) {
            animationDuration = 0;
            timeRow.input.disabled = true;
            timeRow.input.min = "0";
            timeRow.input.max = "0";
            timeRow.input.value = "0.000";
            timeRow.val.textContent = "0.000";
            setStatus("No world-joint frames to preview.", true);
            syncAnimBars();
            return;
        }

        animationDuration = worldFrames.length > 1 ? (worldFrames.length / worldFps) : 0;
        isAnimationPlaying = worldFrames.length > 1;
        isScrubbing = false;
        timeRow.input.disabled = worldFrames.length <= 1;
        timeRow.input.min = "0";
        timeRow.input.max = String(Math.max(animationDuration, 0.001));
        timeRow.input.step = "0.001";
        timeRow.input.value = "0.000";
        timeRow.val.textContent = "0.000";

        setWorldFrame(0, true);
        setStatus(`World joints loaded. ${worldFrames.length} frame(s).`);
        syncAnimBars();
    }

    function resize() {
        const w = Math.max(64, canvasWrap.clientWidth || 64);
        const h = Math.max(64, canvasWrap.clientHeight || 64);
        renderer.setSize(w, h, false);
        camera.aspect = Math.max(1.0, frameWidth) / Math.max(1.0, frameHeight);
        camera.updateProjectionMatrix();
        updateCameraFrameOverlay();
    }

    const resizeObserver = new ResizeObserver(() => resize());
    resizeObserver.observe(canvasWrap);

    controls.addEventListener("change", () => {
        updateCameraFrameOverlay();
        applyCameraInfoToWidget();
    });

    fovRow.input.addEventListener("input", () => {
        camera.fov = Number(fovRow.input.value);
        fovRow.val.textContent = camera.fov.toFixed(1);
        camera.updateProjectionMatrix();
        applyCameraInfoToWidget();
    });

    nearRow.input.addEventListener("input", () => {
        const near = Math.max(0.001, Number(nearRow.input.value));
        camera.near = Math.min(near, camera.far - 0.001);
        nearRow.val.textContent = camera.near.toFixed(2);
        nearRow.input.value = String(camera.near);
        camera.updateProjectionMatrix();
        applyCameraInfoToWidget();
    });

    farRow.input.addEventListener("input", () => {
        const far = Math.max(camera.near + 0.001, Number(farRow.input.value));
        camera.far = far;
        farRow.val.textContent = camera.far.toFixed(0);
        farRow.input.value = String(camera.far);
        camera.updateProjectionMatrix();
        applyCameraInfoToWidget();
    });

    animPlayBtn.addEventListener("click", () => {
        if (animationDuration <= 0) {
            return;
        }
        isAnimationPlaying = !isAnimationPlaying;
        syncAnimBars();
    });

    speedRow.input.addEventListener("input", () => {
        playbackSpeed = Math.max(0.0, Number(speedRow.input.value));
        speedRow.val.textContent = playbackSpeed.toFixed(2);
    });

    timeRow.input.addEventListener("pointerdown", () => {
        isScrubbing = true;
    });
    timeRow.input.addEventListener("pointerup", () => {
        isScrubbing = false;
    });
    timeRow.input.addEventListener("input", () => {
        const t = Number(timeRow.input.value);
        timeRow.val.textContent = t.toFixed(3);
        if (mixer) {
            mixer.setTime(t);
        }
        if (worldFrames.length > 0) {
            worldCurrentTime = Math.max(0.0, Math.min(Math.max(animationDuration, 0.0), t));
            const idx = Math.max(0, Math.min(worldFrames.length - 1, Math.floor(worldCurrentTime * worldFps + 1e-6)));
            setWorldFrame(idx, false);
        }
    });

    resetBtn.addEventListener("click", () => {
        if (currentObject) {
            fitObjectToView(currentObject);
        } else {
            camera.position.set(0, 1.2, 4.0);
            controls.target.set(0, 1.0, 0);
            controls.update();
            updateCameraFrameOverlay();
            syncCameraBarsFromCamera();
            applyCameraInfoToWidget();
        }
        setStatus("Camera reset");
    });

    function animate() {
        rafId = requestAnimationFrame(animate);
        const dt = clock.getDelta();
        if (mixer && isAnimationPlaying && !isScrubbing) {
            mixer.update(dt * playbackSpeed);
        }
        if (mixer && animationDuration > 0 && !isScrubbing) {
            const t = ((mixer.time % animationDuration) + animationDuration) % animationDuration;
            timeRow.input.value = t.toFixed(3);
            timeRow.val.textContent = t.toFixed(3);
        }
        if (!mixer && worldFrames.length > 0) {
            if (isAnimationPlaying && !isScrubbing && animationDuration > 0) {
                worldCurrentTime = ((worldCurrentTime + dt * playbackSpeed) % animationDuration + animationDuration) % animationDuration;
                timeRow.input.value = worldCurrentTime.toFixed(3);
                timeRow.val.textContent = worldCurrentTime.toFixed(3);
            }
            const idx = Math.max(0, Math.min(worldFrames.length - 1, Math.floor(worldCurrentTime * worldFps + 1e-6)));
            setWorldFrame(idx, false);
        }
        controls.update();

        const fullW = Math.max(64, canvasWrap.clientWidth || 64);
        const fullH = Math.max(64, canvasWrap.clientHeight || 64);

        renderer.setScissorTest(false);
        renderer.setViewport(0, 0, fullW, fullH);
        renderer.clear(true, true, true);

        const rx = Math.max(0, Math.min(fullW - 1, renderFrameRect.x));
        const ryTop = Math.max(0, Math.min(fullH - 1, renderFrameRect.y));
        const rw = Math.max(1, Math.min(fullW - rx, renderFrameRect.w));
        const rh = Math.max(1, Math.min(fullH - ryTop, renderFrameRect.h));
        const ryBottom = fullH - (ryTop + rh);

        renderer.setViewport(rx, ryBottom, rw, rh);
        renderer.setScissor(rx, ryBottom, rw, rh);
        renderer.setScissorTest(true);
        renderer.render(scene, camera);
        renderer.setScissorTest(false);
    }

    resize();
    applyWidgetCameraInfoToView();
    syncCameraBarsFromCamera();
    syncAnimBars();
    updateCameraFrameOverlay();
    applyCameraInfoToWidget();
    animate();

    return {
        container,
        handleExecuted(output) {
            const worldData = output && (output.nlf_world_preview || (output.ui && output.ui.nlf_world_preview));
            if (Array.isArray(worldData) && worldData.length > 0) {
                const item = worldData[0] || {};
                if (typeof item.camera_info_json === "string" && item.camera_info_json.length > 0 && cameraInfoWidget) {
                    cameraInfoWidget.value = item.camera_info_json;
                }
                if (typeof item.frame_width === "number") {
                    frameWidth = Math.max(0.1, Number(item.frame_width));
                }
                if (typeof item.frame_height === "number") {
                    frameHeight = Math.max(1.0, Number(item.frame_height));
                }
                updateCameraFrameOverlay();
                loadWorldPreview(item);
                return;
            }

            const data = output && (output.nlf_glb_preview || (output.ui && output.ui.nlf_glb_preview));
            if (!Array.isArray(data) || data.length === 0) {
                return;
            }
            const item = data[0] || {};
            if (typeof item.camera_info_json === "string" && item.camera_info_json.length > 0 && cameraInfoWidget) {
                cameraInfoWidget.value = item.camera_info_json;
            }
            if (typeof item.frame_width === "number") {
                frameWidth = Math.max(0.1, Number(item.frame_width));
            }
            if (typeof item.frame_height === "number") {
                frameHeight = Math.max(1.0, Number(item.frame_height));
            } else if (typeof item.frame_length === "number") {
                frameHeight = Math.max(1.0, Number(item.frame_length));
            }
            updateCameraFrameOverlay();
            const url = buildViewUrl(item);
            loadGlb(url);
        },
        dispose() {
            if (rafId !== null) {
                cancelAnimationFrame(rafId);
            }
            controls.dispose();
            resizeObserver.disconnect();
            clearCurrentSceneObject();
            viewfinderOverlay.remove();
            renderer.dispose();
            container.remove();
        },
    };
}

app.registerExtension({
    name: "ComfyUI.SCAILPose.GLBPreviewCamera",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!NODE_NAMES.has(nodeData.name)) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = async function () {
            onNodeCreated?.apply(this, arguments);

            this.setSize([Math.max(this.size?.[0] || 700, 760), Math.max(this.size?.[1] || 560, 560)]);

            const cameraInfoWidget = this.widgets?.find((w) => w.name === "camera_info_json");
            if (cameraInfoWidget) {
                cameraInfoWidget.options = cameraInfoWidget.options || {};
                cameraInfoWidget.options.height = 70;
                cameraInfoWidget.computeSize = function (width) {
                    return [width, 70];
                };
            }

            let viewer;
            try {
                viewer = await createViewer(this, cameraInfoWidget);
            } catch (e) {
                console.error("Failed to initialize GLB preview camera widget", e);
                return;
            }

            const widget = this.addDOMWidget("glb_viewfinder", "HTML", viewer.container, {
                getMinHeight: () => 420,
                hideOnZoom: false,
                serialize: false,
            });

            const origOnExecuted = this.onExecuted;
            this.onExecuted = function (output) {
                origOnExecuted?.call(this, output);
                viewer.handleExecuted(output);
            };

            const baseOnRemove = widget.onRemove?.bind(widget);
            widget.onRemove = () => {
                baseOnRemove?.();
                viewer.dispose();
            };
        };
    },
});
