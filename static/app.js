const API_BASE_URL = 'http://localhost:8000';

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const fileList = document.getElementById('fileList');
const uploadStatus = document.getElementById('uploadStatus');
const queryInput = document.getElementById('queryInput');
const queryButton = document.getElementById('queryButton');
const resultsSection = document.getElementById('resultsSection');
const answerText = document.getElementById('answerText');
const confidenceBar = document.getElementById('confidenceBar');
const confidenceText = document.getElementById('confidenceText');
const completenessText = document.getElementById('completenessText');
const sourcesContainer = document.getElementById('sourcesContainer');
const sourcesList = document.getElementById('sourcesList');
const missingInfoContainer = document.getElementById('missingInfoContainer');
const missingInfoList = document.getElementById('missingInfoList');
const suggestionsContainer = document.getElementById('suggestionsContainer');
const suggestionsList = document.getElementById('suggestionsList');
const dbStatus = document.getElementById('dbStatus');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    checkSystemStatus();
    loadDocumentsList();
});

function setupEventListeners() {
    // Upload area click
    uploadArea.addEventListener('click', () => fileInput.click());

    // File input change
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('drag-over');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        const files = e.dataTransfer.files;
        handleFiles(files);
    });

    // Query button
    queryButton.addEventListener('click', handleQuery);

    // Enter key in textarea (Ctrl+Enter to submit)
    queryInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
            handleQuery();
        }
    });

    // Refresh documents button
    const refreshDocsButton = document.getElementById('refreshDocsButton');
    if (refreshDocsButton) {
        refreshDocsButton.addEventListener('click', loadDocumentsList);
    }
}

function handleFileSelect(e) {
    const files = e.target.files;
    handleFiles(files);
}

async function handleFiles(files) {
    if (files.length === 0) return;

    // Display selected files
    fileList.innerHTML = '';
    Array.from(files).forEach(file => {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        fileItem.innerHTML = `
            <span class="file-name">${file.name}</span>
            <span class="file-size">${formatFileSize(file.size)}</span>
        `;
        fileList.appendChild(fileItem);
    });

    // Upload files
    await uploadFiles(files);
}

async function uploadFiles(files) {
    const formData = new FormData();
    Array.from(files).forEach(file => {
        formData.append('files', file);
    });

    showStatus('Uploading documents...', 'info');

    try {
        const response = await fetch(`${API_BASE_URL}/ingest?doc_type=generic`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Upload failed');
        }

        const result = await response.json();
        showStatus(`✅ ${result.message}`, 'success');

        // Refresh system status and documents list
        setTimeout(() => {
            checkSystemStatus();
            loadDocumentsList();
        }, 1000);

        // Clear file input
        fileInput.value = '';
        setTimeout(() => {
            fileList.innerHTML = '';
            uploadStatus.innerHTML = '';
        }, 3000);

    } catch (error) {
        showStatus(`❌ Error: ${error.message}`, 'error');
    }
}

async function handleQuery() {
    const query = queryInput.value.trim();
    if (!query) return;

    // Disable button and show loading
    queryButton.disabled = true;
    queryButton.querySelector('.btn-text').hidden = true;
    queryButton.querySelector('.btn-loading').hidden = false;

    // Hide previous results
    resultsSection.hidden = true;

    try {
        const response = await fetch(
            `${API_BASE_URL}/query?query=${encodeURIComponent(query)}`,
            { method: 'POST' }
        );

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Query failed');
        }

        const result = await response.json();
        displayResults(result);

    } catch (error) {
        alert(`Error: ${error.message}`);
    } finally {
        // Re-enable button
        queryButton.disabled = false;
        queryButton.querySelector('.btn-text').hidden = false;
        queryButton.querySelector('.btn-loading').hidden = true;
    }
}

function displayResults(result) {
    // Show results section
    resultsSection.hidden = false;
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    // Display answer
    answerText.textContent = result.answer || 'No answer generated.';

    // Display confidence
    const confidence = Math.round((result.confidence || 0) * 100);
    confidenceBar.style.setProperty('--confidence-width', `${confidence}%`);
    confidenceText.textContent = `${confidence}%`;

    // Display completeness
    if (result.is_complete) {
        completenessText.textContent = '✓ Complete';
        completenessText.className = 'completeness-badge complete';
    } else {
        completenessText.textContent = '⚠ Incomplete';
        completenessText.className = 'completeness-badge incomplete';
    }

    // Display sources
    if (result.sources && result.sources.length > 0) {
        sourcesContainer.hidden = false;
        sourcesList.innerHTML = result.sources
            .map(source => `<span class="source-tag">${source}</span>`)
            .join('');
    } else {
        sourcesContainer.hidden = true;
    }

    // Display missing info
    if (result.missing_info && result.missing_info.length > 0) {
        missingInfoContainer.hidden = false;
        missingInfoList.innerHTML = result.missing_info
            .map(info => `<li>${info}</li>`)
            .join('');
    } else {
        missingInfoContainer.hidden = true;
    }

    // Display enrichment suggestions
    if (result.enrichment_suggestions && result.enrichment_suggestions.length > 0) {
        suggestionsContainer.hidden = false;
        suggestionsList.innerHTML = result.enrichment_suggestions
            .map(suggestion => `
                <div class="suggestion-card">
                    <span class="suggestion-type ${suggestion.priority}">${suggestion.type}</span>
                    <p class="suggestion-action">${suggestion.action}</p>
                </div>
            `)
            .join('');
    } else {
        suggestionsContainer.hidden = true;
    }
}

async function loadDocumentsList() {
    const documentsContainer = document.getElementById('documentsContainer');

    try {
        const response = await fetch(`${API_BASE_URL}/documents`);
        if (!response.ok) throw new Error('Failed to load documents');

        const data = await response.json();

        if (data.documents && data.documents.length > 0) {
            documentsContainer.innerHTML = `
                <div class="documents-grid">
                    ${data.documents.map(doc => {
                        const fileIcon = getFileIcon(doc.file_type);
                        return `
                        <div class="document-card">
                            <div class="document-icon">${fileIcon}</div>
                            <div class="document-info">
                                <h4 class="document-name">${doc.source}</h4>
                                <div class="document-meta">
                                    <span class="meta-tag">${doc.chunk_count} chunk${doc.chunk_count !== 1 ? 's' : ''}</span>
                                </div>
                            </div>
                            <button class="btn-delete" onclick="deleteDocument('${doc.source}')" title="Delete document">
                                🗑️
                            </button>
                        </div>
                    `}).join('')}
                </div>
                <div class="documents-summary">
                    Total: ${data.total} document${data.total !== 1 ? 's' : ''}
                </div>
            `;
        } else {
            documentsContainer.innerHTML = '<p class="empty-state">No documents uploaded yet. Upload documents above to get started.</p>';
        }
    } catch (error) {
        console.error('Error loading documents:', error);
        documentsContainer.innerHTML = '<p class="empty-state error">Failed to load documents. Please try again.</p>';
    }
}

async function deleteDocument(source) {
    if (!confirm(`Are you sure you want to delete "${source}"?`)) {
        return;
    }

    try {
        const response = await fetch(`${API_BASE_URL}/qdrant/cleanup?source=${encodeURIComponent(source)}`, {
            method: 'DELETE'
        });

        if (!response.ok) throw new Error('Failed to delete document');

        const result = await response.json();
        alert(result.message);

        // Refresh documents list and system status
        await loadDocumentsList();
        await checkSystemStatus();
    } catch (error) {
        alert(`Error: ${error.message}`);
    }
}

async function checkSystemStatus() {
    try {
        // Check database health
        const healthResponse = await fetch(`${API_BASE_URL}/qdrant/health`);
        if (healthResponse.ok) {
            dbStatus.textContent = '✓ Connected';
            dbStatus.style.color = 'var(--success)';
        } else {
            dbStatus.textContent = '✗ Disconnected';
            dbStatus.style.color = 'var(--danger)';
        }
    } catch (error) {
        dbStatus.textContent = '✗ Error';
        dbStatus.style.color = 'var(--danger)';
    }
}

function showStatus(message, type) {
    uploadStatus.textContent = message;
    uploadStatus.className = `status-message ${type}`;
}

function getFileIcon(fileType) {
    const icons = {
        'pdf': '📕',
        'txt': '📄',
        'md': '📝',
        'json': '📋',
        'csv': '📊',
        'doc': '📘',
        'docx': '📘',
        'default': '📄'
    };
    return icons[fileType.toLowerCase()] || icons['default'];
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}
