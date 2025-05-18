document.addEventListener('DOMContentLoaded', () => {
    const searchForm = document.getElementById('searchForm');
    const resultsDiv = document.getElementById('results');
    const statusDiv = document.getElementById('status');
    const loader = document.getElementById('loader');
    const currentYearSpan = document.getElementById('currentYear');

    if (currentYearSpan) {
        currentYearSpan.textContent = new Date().getFullYear();
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

            // Check if isoTimestampString is a non-empty string
            if (typeof isoTimestampString === 'string' && isoTimestampString.trim() !== '') {
                const d = new Date(isoTimestampString);

                if (d instanceof Date && !isNaN(d.valueOf())) {
                    const datePart = d.toLocaleDateString('sv-SE');
                    const timePart = d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });
                    humanReadableTime = `${datePart} ${timePart}`;
                }
            }
            statusDiv.innerHTML = `Indexing: ${data.index_ready ? 'Ready' : 'Indexing...'}<br>Documents: ${data.index_size || 0}<br>Last Indexed Time: ${humanReadableTime}`;
            if (modelNameDisplay) {
                modelNameDisplay.textContent = `Model: ${data.embedding_model_name || 'N/A'}`;
            }
            statusDiv.classList.remove('ready', 'not-ready', 'error');
            if (data.index_ready) {
                statusDiv.classList.add('ready');
            } else {
                statusDiv.classList.add('not-ready');
            }
        } catch (error) {
            statusDiv.textContent = 'Status: Error fetching status.';
            statusDiv.classList.remove('ready', 'not-ready');
            statusDiv.classList.add('error');
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
        let escapedTextContent = escapeHtml(text);
        return escapedTextContent.replace(regex, '<mark>$1</mark>');
    }

    searchForm.addEventListener('submit', async function(event) {
        event.preventDefault();
        resultsDiv.innerHTML = '';
        loader.style.display = 'block';
        const startTime = performance.now();

        try {
            const topK = document.querySelector('#topK').value || '5';
            const currentQuery = document.getElementById('query').value.trim();

            if (!currentQuery) {
                resultsDiv.innerHTML = '';
                loader.style.display = 'none';
                return;
            }

            const response = await fetch(`/search?q=${encodeURIComponent(currentQuery)}&top_k=${topK}`);
            loader.style.display = 'none';

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
            resultsDiv.innerHTML = `<h3>Found ${resultCount} result${resultCount === 1 ? '' : 's'} in ${queryTime} seconds</h3>`;

            if (data.results && data.results.length > 0) {
                data.results.sort((a, b) => b[0] - a[0]); // Sort by score

                let html = ``;
                
                data.results.forEach(item => {
                    const score = item[0];
                    const text = item[1];
                    const sourceId = item[2] || "Unknown Source";

                    const highlightedAndEscapedText = highlightTerms(text, currentQuery);
                    const tooltipText = "Relevance score (0-1 scale): Higher score indicates better match.";
                    const escapedSourceId = escapeHtml(sourceId);

                    html += `
                        <div class="result-item">
                            <div class="result-header">
                                <p class="result-score"><strong title="${tooltipText}" data-tooltip="${tooltipText}">Score:</strong> ${score.toFixed(4)}</p>
                                <p class="result-source" style="margin-bottom: 10px;"><strong>Source:</strong> ${escapedSourceId}</p>
                            </div>
                            <p class="result-text">${highlightedAndEscapedText}</p>
                        </div>
                    `;
                });
                resultsDiv.innerHTML += html;
            } else {
                resultsDiv.innerHTML = `<p class="message info">No results found for "<em>${escapeHtml(currentQuery)}</em>".<br>Try rephrasing your query or using different keywords.</p>`;
            }
        } catch (error) {
            loader.style.display = 'none';
            resultsDiv.innerHTML = `<p class="message error">An error occurred: ${escapeHtml(error.message)}</p>`;
            console.error('Search error:', error);
        }
    });

    fetchStatus();
    setInterval(fetchStatus, 1000); // Check status every second
});