// DOM Elements
const frames = document.querySelectorAll('.frame');
const progressFill = document.getElementById('progress-fill');
const progressStatus = document.getElementById('progress-status');
const modelCards = document.querySelectorAll('.model-card');
const outputModel = document.getElementById('output-model');
const outputImage = document.getElementById('output-image');
const stepIcons = document.querySelectorAll('.step-icon');
const selectedModelInput = document.getElementById('selected-model');
const uploadArea = document.getElementById('upload-area'); // Added missing variable
const fileInput = document.getElementById('sketch-upload');
const previewImage = document.getElementById('preview-image');

// Navigation buttons
const next1Button = document.getElementById('next-1');
const next2Button = document.getElementById('next-2');
const back2Button = document.getElementById('back-2');
const startOver3Button = document.getElementById('start-over-3');
const startOver4Button = document.getElementById('start-over-4');
const downloadAllButton = document.getElementById('download-all');

// Variables to store state
let selectedModel = '';
let hasUploadedSketch = false;
let originalOutputSrc = ''; // For comparison toggle

// Theme toggle functionality
const themeToggle = document.getElementById('theme-toggle');
const themeText = document.getElementById('theme-text');
const lightIcon = document.getElementById('light-icon');
const darkIcon = document.getElementById('dark-icon');
const prefersDarkScheme = window.matchMedia('(prefers-color-scheme: dark)');
const currentTheme = localStorage.getItem('theme');

// Toggle theme
if (themeToggle) {
    if (currentTheme === 'dark' || (!currentTheme && prefersDarkScheme.matches)) {
        document.body.classList.add('dark-theme');
        themeText.textContent = 'Light Mode';
        lightIcon.style.display = 'none';
        darkIcon.style.display = 'block';
    } else {
        document.body.classList.remove('dark-theme');
        themeText.textContent = 'Dark Mode';
        lightIcon.style.display = 'block';
        darkIcon.style.display = 'none';
    }

    themeToggle.addEventListener('click', () => {
        if (document.body.classList.contains('dark-theme')) {
            document.body.classList.remove('dark-theme');
            localStorage.setItem('theme', 'light');
            themeText.textContent = 'Dark Mode';
            lightIcon.style.display = 'block';
            darkIcon.style.display = 'none';
        } else {
            document.body.classList.add('dark-theme');
            localStorage.setItem('theme', 'dark');
            themeText.textContent = 'Light Mode';
            lightIcon.style.display = 'none';
            darkIcon.style.display = 'block';
        }
    });
}

// Download all button
downloadAllButton.addEventListener('click', () => {
    const pngLink = document.getElementById('download-png');
    const jpgLink = document.getElementById('download-jpg');
    pngLink.click();
    jpgLink.click();
});

// Set animation order for model cards
document.querySelectorAll('.model-card').forEach((card, index) => {
    card.style.setProperty('--order', index);
});

// Show a specific frame
function showFrame(frameNumber) {
    frames.forEach((frame, index) => {
        frame.classList.toggle('active', index + 1 === frameNumber);
    });

    if (frameNumber === 3) {
        simulateTranslation();
    }
}

// Reset to initial state
function startOver() {
    hasUploadedSketch = false;
    selectedModel = '';
    selectedModelInput.value = '';
    originalOutputSrc = '';

    previewImage.src = "{{ url_for('static', filename='placeholder/400/320') }}";
    previewImage.style.display = 'none';

    uploadArea.querySelector('svg').style.display = 'block';
    uploadArea.querySelector('h3').style.display = 'block';
    uploadArea.querySelectorAll('p').forEach(p => p.style.display = 'block');

    fileInput.value = '';
    progressFill.style.width = '0';
    progressStatus.textContent = "Processing your sketch...";
    modelCards.forEach(card => card.classList.remove('selected'));
    stepIcons.forEach((icon, index) => {
        icon.classList.toggle('active', index === 0);
    });

    showFrame(1);
}

// File input handling
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        const file = e.target.files[0];
        if (file.size > 5 * 1024 * 1024) { // 5MB limit
            showError("File exceeds 5MB limit");
            fileInput.value = '';
            return;
        }
        const reader = new FileReader();
        reader.onload = (event) => {
            previewImage.src = event.target.result;
            previewImage.style.display = 'block';
            uploadArea.querySelector('svg').style.display = 'none';
            uploadArea.querySelector('h3').style.display = 'none';
            uploadArea.querySelectorAll('p').forEach(p => p.style.display = 'none');
        };
        hasUploadedSketch = true;
        reader.readAsDataURL(file);
    }
});

// Model selection
modelCards.forEach(card => {
    card.addEventListener('click', () => {
        modelCards.forEach(c => c.classList.remove('selected'));
        card.classList.add('selected');
        selectedModel = card.dataset.model;
        outputModel.textContent = card.querySelector('h3').textContent;
        selectedModelInput.value = selectedModel;
    });
});

// Simulate translation with API call
function simulateTranslation() {
    progressFill.style.width = '0%';
    progressStatus.textContent = "Initializing...";

    let progress = 0;
    const interval = setInterval(() => {
        progress += 1;
        progressFill.style.width = `${progress}%`;

        if (progress === 25) {
            stepIcons[0].classList.remove('active');
            stepIcons[1].classList.add('active');
            progressStatus.textContent = "Processing features...";
        } else if (progress === 50) {
            stepIcons[1].classList.remove('active');
            stepIcons[2].classList.add('active');
            progressStatus.textContent = "Refining details...";
        } else if (progress === 75) {
            stepIcons[2].classList.remove('active');
            stepIcons[3].classList.add('active');
            progressStatus.textContent = "Finalizing image...";
        }

        if (progress >= 100) {
            clearInterval(interval);
        }
    }, 155);
}

// Navigation button event listeners
next1Button.addEventListener('click', () => {
    if (hasUploadedSketch) {
        showFrame(2);
    } else {
        showError("Please upload a sketch first!");
    }
});

next2Button.addEventListener('click', () => {
    if (selectedModel) {
        showFrame(3);
        const formData = new FormData();
        formData.append("file", fileInput.files[0]);
        formData.append("model", selectedModel);

        fetch("/generate", {
            method: "POST",
            body: formData
        })
        .then(response => {
            if (!response.ok) throw new Error("Server error");
            return response.json();
        })
        .then(data => {
            if (data.output_url) {
                const outputUrlPng = data.output_url.png;
                const outputUrlJpg = data.output_url.jpg;
                originalOutputSrc = outputUrlPng;
                outputImage.src = outputUrlPng;
                outputImage.style.display = "block";

                const pngFilename = outputUrlPng.split('/').pop().split('.')[0];
                const jpgFilename = outputUrlJpg.split('/').pop().split('.')[0];

                document.getElementById('download-png').href = `/download/png/${pngFilename}`;
                document.getElementById('download-jpg').href = `/download/jpg/${jpgFilename}`;

                showFrame(4);
            } else {
                throw new Error("Output URL not found in response");
            }
        })
        .catch(error => {
            console.error("Error:", error);
            showError(`An error occurred: ${error.message}`);
            setTimeout(() => showFrame(1), 2000); // Delay before resetting
        });
    } else {
        showError("Please select a translation model!");
    }
});

back2Button.addEventListener('click', () => showFrame(1));
startOver3Button.addEventListener('click', startOver);
startOver4Button.addEventListener('click', startOver);

// Drag and drop handling
document.addEventListener('dragover', (e) => e.preventDefault());
document.addEventListener('drop', (e) => {
    e.preventDefault();
    if (uploadArea.contains(e.target)) {
        if (e.dataTransfer.files.length > 0) {
            fileInput.files = e.dataTransfer.files;
            fileInput.dispatchEvent(new Event('change', { bubbles: true }));
        }
    }
});

// Error notification function
function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.classList.add('error-notification');
    errorDiv.innerHTML = `
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"></circle>
            <line x1="12" y1="8" x2="12" y2="12"></line>
            <line x1="12" y1="16" x2="12.01" y2="16"></line>
        </svg>
        <span>${message}</span>
    `;
    document.body.appendChild(errorDiv);

    setTimeout(() => {
        errorDiv.classList.add('show');
        setTimeout(() => {
            errorDiv.classList.remove('show');
            setTimeout(() => errorDiv.remove(), 300);
        }, 5000);
    }, 100);
}

// Comparison toggle (requires HTML element)
const comparisonToggle = document.getElementById('comparison-toggle');
if (comparisonToggle) {
    comparisonToggle.addEventListener('change', () => {
        outputImage.style.transition = 'opacity 0.3s ease';
        outputImage.style.opacity = '0';
        setTimeout(() => {
            outputImage.src = comparisonToggle.checked ? previewImage.src : originalOutputSrc;
            outputImage.style.opacity = '1';
        }, 300);
    });
}