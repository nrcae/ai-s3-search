document.addEventListener('DOMContentLoaded', () => {
    const searchForm = document.getElementById('searchForm');
    const resultsDiv = document.getElementById('results');
    const statusDiv = document.getElementById('status');
    const loader = document.getElementById('loader');
    const currentYearSpan = document.getElementById('currentYear');
    const modelNameDisplay = document.getElementById('modelNameDisplay');
    const topKInput = document.getElementById('topK');
    const queryInput = document.getElementById('query');
    const uploadForm = document.getElementById('uploadForm');
    const fileInput = document.getElementById('fileInput');
    const uploadStatusDiv = document.getElementById('uploadStatus');
    const uploadButton = document.getElementById('uploadButton');
    const sourceSelect = document.getElementById('sourceSelect');

    async function loadSources() {
        try {
            const response = await fetch('/sources');
            if (!response.ok) throw new Error('Failed to fetch sources');
            const data = await response.json();

            if (data.sources && Array.isArray(data.sources)) {
                data.sources.forEach(src => {
                    const option = document.createElement('option');
                    option.value = src;
                    option.textContent = src;
                    sourceSelect.appendChild(option);
                });
            }
        } catch (error) {
            console.error('Error loading sources:', error);
        }
    }

    if (currentYearSpan) {
        currentYearSpan.textContent = new Date().getFullYear();
    }

    const dateLocale = 'sv-SE';
    const timeOptions = { hour: '2-digit', minute: '2-digit', timeZone: 'UTC' };

    if (uploadForm) {
        uploadForm.addEventListener('submit', async function(event) {
            event.preventDefault();
            if (!fileInput.files || fileInput.files.length === 0) {
                displayUploadMessage('Please select a PDF file to upload.', 'error');
                return;
            }

            const file = fileInput.files[0];
            if (file.type !== "application/pdf") {
                displayUploadMessage('Invalid file type. Please select a PDF.', 'error');
                return;
            }

            const formData = new FormData();
            formData.append('file', file);

            uploadButton.disabled = true;
            uploadButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Uploading...';
            displayUploadMessage('Uploading file...', 'info', false);

            try {
                const response = await fetch('/upload_pdf', {
                    method: 'POST',
                    body: formData,
                });

                const result = await response.json();

                if (response.ok) {
                    displayUploadMessage(`Success: ${result.filename} uploaded. Indexing started.`, 'info');
                    fileInput.value = '';
                    fetchStatus();
                } else {
                    displayUploadMessage(`Error: ${result.detail || 'Upload failed.'}`, 'error');
                }
            } catch (error) {
                console.error('Upload error:', error);
                displayUploadMessage('An unexpected error occurred during upload.', 'error');
            } finally {
                uploadButton.disabled = false;
                uploadButton.innerHTML = '<i class="fas fa-upload"></i> Upload';
            }
        });
    }

    function displayUploadMessage(message, type, autoHide = true) {
        if (uploadStatusDiv) {
            uploadStatusDiv.textContent = message;
            uploadStatusDiv.className = `message ${type}`;
            uploadStatusDiv.style.display = 'block';

            if (autoHide) {
                setTimeout(() => {
                    uploadStatusDiv.style.display = 'none';
                }, 5000);
            }
        }
    }

    async function fetchStatus() {
        try {
            const response = await fetch('/status');
            if (!response.ok) {
                throw new Error(`Status check failed: ${response.status} ${response.statusText}`);
            }
            const data = await response.json();
            let humanReadableTime = 'N/A';
            const isoTimestampString = data.last_indexed_time;

            if (typeof isoTimestampString === 'string' && isoTimestampString.trim() !== '') {
                const d = new Date(isoTimestampString);
                if (d instanceof Date && !isNaN(d.valueOf())) {
                    const datePart = d.toLocaleDateString(dateLocale);
                    const timePart = d.toLocaleTimeString('en-GB', timeOptions);
                    humanReadableTime = `${datePart} ${timePart} UTC`;
                }
            }
            statusDiv.innerHTML = `Indexing: ${data.index_ready ? 'Ready' : 'Indexing...'}<br>Documents: ${data.index_size || 0}<br>Last Indexed Time: ${humanReadableTime}`;

            if (modelNameDisplay) {
                modelNameDisplay.textContent = `Model: ${data.embedding_model_name || 'N/A'}`;
            }

            statusDiv.classList.toggle('ready', data.index_ready);
            statusDiv.classList.toggle('not-ready', !data.index_ready);
            statusDiv.classList.remove('error');

        } catch (error) {
            statusDiv.textContent = 'Status: Error fetching status.';
            statusDiv.classList.add('error');
            statusDiv.classList.remove('ready', 'not-ready');
            console.error('Error fetching status:', error);
        }
    }

    function escapeHtml(unsafe) {
        if (typeof unsafe !== 'string') return '';
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    function highlightTerms(text, query) {
        if (!query || typeof text !== 'string') {
            return escapeHtml(text);
        }
        const queryTerms = query.trim().split(/\s+/)
            .map(term => term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
        if (queryTerms.length === 0 || (queryTerms.length === 1 && queryTerms[0] === '')) {
            return escapeHtml(text);
        }
        const regex = new RegExp(`(${queryTerms.join('|')})`, 'gi');
        const escapedTextContent = escapeHtml(text);
        return escapedTextContent.replace(regex, '<mark>$1</mark>');
    }

    searchForm.addEventListener('submit', async function(event) {
        event.preventDefault();
        resultsDiv.innerHTML = '';
        loader.style.display = 'block';
        const startTime = performance.now();

        try {
            const topK = topKInput.value || '5';
            const currentQuery = queryInput.value.trim();
            const sourceId = sourceSelect.value;

            let searchUrl = `/search?q=${encodeURIComponent(currentQuery)}&top_k=${topK}`;
            if (sourceId) {
                searchUrl += `&source_id=${encodeURIComponent(sourceId)}`;
            }

            if (!currentQuery) {
                alert('Please enter a search query.');
                return;
            }

            const response = await fetch(`/search?q=${encodeURIComponent(currentQuery)}&top_k=${topK}`);

            if (response.status === 503) {
                resultsDiv.innerHTML = '<p class="message warning">Search index is not ready. Please try again later.</p>';
                return;
            }
            if (response.status === 400) {
                const errorData = await response.json();
                resultsDiv.innerHTML = `<p class="message error">Error: ${escapeHtml(errorData.detail || 'Invalid query')}</p>`;
                return;
            }
            if (!response.ok) {
                throw new Error(`Search request failed: ${response.status} ${response.statusText}`);
            }

            const data = await response.json();
            const endTime = performance.now();
            const queryTime = ((endTime - startTime) / 1000).toFixed(2);
            const resultCount = data.results.length;

            let contentHtml = `<h3>Found ${resultCount} result${resultCount === 1 ? '' : 's'} in ${queryTime} seconds</h3>`;

            if (data.results && data.results.length > 0) {
                data.results.sort((a, b) => b[0] - a[0]);

                const itemsHtml = data.results.map(item => {
                    const [score, text, sourceIdRaw] = item;
                    const sourceId = sourceIdRaw || "Unknown Source";
                    const highlightedAndEscapedText = highlightTerms(text, currentQuery);
                    const tooltipText = "Relevance score (0-1 scale): Higher score indicates better match.";
                    const escapedSourceId = escapeHtml(sourceId);

                    return `
                        <div class="result-item">
                            <div class="result-header">
                                <p class="result-score"><strong title="${tooltipText}" data-tooltip="${tooltipText}">Score:</strong> ${score.toFixed(4)}</p>
                                <p class="result-source" style="margin-bottom: 10px;"><strong>Source:</strong> ${escapedSourceId}</p>
                            </div>
                            <p class="result-text">${highlightedAndEscapedText}</p>
                        </div>
                    `;
                }).join('');
                contentHtml += itemsHtml;
            } else {
                contentHtml = `<p class="message info">No results found for "<em>${escapeHtml(currentQuery)}</em>".<br>Try rephrasing your query or using different keywords.</p>`;
            }
            resultsDiv.innerHTML = contentHtml;

        } catch (error) {
            resultsDiv.innerHTML = `<p class="message error">An error occurred: ${escapeHtml(error.message)}</p>`;
            console.error('Search error:', error);
        } finally {
            loader.style.display = 'none';
        }
    });

    loadSources();
    fetchStatus();
    setInterval(fetchStatus, 5000);
});