// ==========================================================================
// 1. GLOBAL STATE & THREE.JS SETUP
// ==========================================================================
let currentNamespace = "";

// Canvas & Scene Initialization
const container = document.getElementById('canvas-container');
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });

renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
container.appendChild(renderer.domElement);

// --- Particle Constellation (Simulating Pinecone Vector Space) ---
const particleCount = 1000;
const geometry = new THREE.BufferGeometry();
const positions = new Float32Array(particleCount * 3);

for (let i = 0; i < particleCount * 3; i++) {
    positions[i] = (Math.random() - 0.5) * 18;
}

geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
const particleMaterial = new THREE.PointsMaterial({
    color: 0x38bdf8, // Electric Cyan
    size: 0.03,
    transparent: true,
    opacity: 0.7
});

const particleSystem = new THREE.Points(geometry, particleMaterial);
scene.add(particleSystem);

// --- Central Rotating Index Core ---
const coreGeo = new THREE.IcosahedronGeometry(2.2, 2);
const coreMat = new THREE.MeshBasicMaterial({ color: 0x1e293b, wireframe: true });
const coreMesh = new THREE.Mesh(coreGeo, coreMat);
scene.add(coreMesh);

camera.position.z = 6;

// ==========================================================================
// 2. 3D ANIMATION LOOP & INTERACTION PULSE
// ==========================================================================
function animate() {
    requestAnimationFrame(animate);
    
    // Smooth background rotations
    particleSystem.rotation.y += 0.0008;
    particleSystem.rotation.x += 0.0003;
    coreMesh.rotation.y -= 0.002;
    
    renderer.render(scene, camera);
}
animate();

// Visual pulse triggered when processing vector actions
function pulseCore() {
    coreMesh.scale.set(1.35, 1.35, 1.35);
    setTimeout(() => {
        coreMesh.scale.set(1.0, 1.0, 1.0);
    }, 350);
}

// Window Resize Responsiveness
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

// ==========================================================================
// 3. FASTAPI BACKEND INTEGRATION
// ==========================================================================
const fileInput = document.getElementById('pdf-file-input');
const uploadStatus = document.getElementById('upload-status');
const namespaceBadge = document.getElementById('namespace-badge');

// --- File Upload & Ingestion Event ---
if (fileInput) {
    fileInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        uploadStatus.innerText = "⏳ Indexing PDF...";
        pulseCore();

        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await fetch("http://127.0.0.1:8000/api/upload", {
                method: "POST",
                body: formData
            });
            const data = await response.json();

            if (response.ok) {
                currentNamespace = data.namespace;
                uploadStatus.innerText = "✅ PDF Loaded";
                namespaceBadge.innerText = `Active: ${data.namespace}`;
                namespaceBadge.classList.replace("text-slate-400", "text-cyan-400");
            } else {
                uploadStatus.innerText = "❌ Upload Failed";
                alert(data.detail || "Error indexing file.");
            }
        } catch (err) {
            uploadStatus.innerText = "❌ Connection Error";
            console.error("FastAPI Backend connection error:", err);
        }
    });
}

// --- Query Execution Event ---
async function executeQuery() {
    const queryInput = document.getElementById('user-query');
    const query = queryInput.value.trim();
    const topKSelect = document.getElementById('topk-select');
    const topK = topKSelect ? parseInt(topKSelect.value) : 5;

    if (!query) return;
    if (!currentNamespace) {
        alert("Please upload and index a PDF document before asking questions!");
        return;
    }

    pulseCore();
    const answerContainer = document.getElementById('answer-container');
    const answerText = document.getElementById('answer-text');
    const sourcesList = document.getElementById('sources-list');

    answerContainer.classList.remove('hidden');
    answerText.innerText = "Searching spatial vectors & generating answer...";
    sourcesList.innerHTML = "";

    try {
        const response = await fetch("http://127.0.0.1:8000/api/query", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                query: query,
                namespace: currentNamespace,
                top_k: topK,
                threshold: 0.40
            })
        });

        const data = await response.json();
        answerText.innerText = data.answer;

        // Render Source Attributions
        if (data.sources && data.sources.length > 0) {
            data.sources.forEach((src, idx) => {
                const sourceDiv = document.createElement('div');
                sourceDiv.className = "text-xs bg-slate-900/80 p-3 rounded-xl border border-slate-800 text-slate-400";
                sourceDiv.innerHTML = `
                    <span class="text-cyan-400 font-semibold">Source ${idx + 1} (Page ${src.page})</span> | Confidence Score: ${src.score}<br>
                    <span class="italic text-slate-300">"${src.excerpt}"</span>
                `;
                sourcesList.appendChild(sourceDiv);
            });
        }
    } catch (err) {
        answerText.innerText = "Error communicating with 3D RAG server.";
        console.error("Query Error:", err);
    }
}