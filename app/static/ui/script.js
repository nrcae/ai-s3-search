// script.js
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
            statusDiv.textContent = `Indexing: ${data.index_ready ? 'Ready' : 'Indexing...'} | Documents: ${data.index_size || 0}`;
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
        let escapedTextContent = escapeHtml(text); // Ensure original text is escaped first
        return escapedTextContent.replace(regex, '<mark>$1</mark>');
    }

    searchForm.addEventListener('submit', async function(event) {
        event.preventDefault();
        resultsDiv.innerHTML = '';
        loader.style.display = 'block';

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

            const data = await response.json(); // API now returns { "results": [ [score, text, source_id], ... ] }

            if (data.results && data.results.length > 0) {
                data.results.sort((a, b) => b[0] - a[0]); // Sort by score

                // Use currentQuery which is already defined and trimmed
                let html = `<p class="results-summary">Found ${data.results.length} result${data.results.length === 1 ? '' : 's'} for "<em>${escapeHtml(currentQuery)}</em>"</p>`;
                html += '<h3>Results:</h3>';
                
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
                resultsDiv.innerHTML = html;
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