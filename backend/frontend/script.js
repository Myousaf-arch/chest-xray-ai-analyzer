const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const previewArea = document.getElementById('previewArea');
const imagePreview = document.getElementById('imagePreview');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultCard = document.getElementById('resultCard');
const resultTitle = document.getElementById('resultTitle');
const confidenceText = document.getElementById('confidenceText');

let fileToUpload = null;

// File Input triggers
dropzone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

// Drag and drop handlers
dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.style.borderColor = '#38bdf8';
});

dropzone.addEventListener('dragleave', () => {
    dropzone.style.borderColor = '#475569';
});

dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.style.borderColor = '#475569';
    if (e.dataTransfer.files.length > 0) {
        handleFileSelect(e.dataTransfer.files[0]);
    }
});

function handleFileSelect(file) {
    fileToUpload = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        previewArea.classList.remove('hidden');
        resultCard.classList.add('hidden');
    };
    reader.readAsDataURL(file);
}

// API Call to FastAPI Backend
analyzeBtn.addEventListener('click', async () => {
    if (!fileToUpload) return;

    analyzeBtn.disabled = true;
    analyzeBtn.innerText = "Analyzing X-Ray...";

    const formData = new FormData();
    formData.append('file', fileToUpload);

    try {
        const response = await fetch('http://127.0.0.1:8000/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            resultCard.classList.remove('hidden');
            resultTitle.innerText = data.result;
            confidenceText.innerText = `Confidence: ${data.confidence}`;

            if (data.is_pneumonia) {
                resultCard.className = "result-card pneumonia";
            } else {
                resultCard.className = "result-card normal";
            }
        } else {
            alert("Error analyzing image: " + data.error);
        }
    } catch (err) {
        alert("Server Error! Check karein FastAPI server http://127.0.0.1:8000 par run ho raha hai ya nahi.");
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.innerText = "Analyze X-Ray";
    }
});