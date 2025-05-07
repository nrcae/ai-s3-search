document.addEventListener('DOMContentLoaded', () => {
    const searchForm = document.getElementById('searchForm');
    const queryInput = document.getElementById('query');
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

            // Update status bar class for styling
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

    searchForm.addEventListener('submit', async function(event) {
        event.preventDefault();
        const query = queryInput.value.trim();
        
        if (!query) {
            resultsDiv.innerHTML = '<p class="message error">Please enter a search query.</p>';
            return;
        }

        resultsDiv.innerHTML = ''; // Clear previous results
        loader.style.display = 'block';

        try {
            const topK = document.querySelector('#topK').value || '5';
            const currentQuery = document.getElementById('query').value.trim();
            const response = await fetch(`/search?q=${encodeURIComponent(currentQuery)}&top_k=${topK}`);
            loader.style.display = 'none';

            if (response.status === 503) {
                resultsDiv.innerHTML = '<p class="message warning">Search index is not ready. Please try again later.</p>';
                return;
            }
            if (response.status === 400) {
                const errorData = await response.json();
                resultsDiv.innerHTML = `<p class="message error">Error: ${errorData.detail || 'Invalid query'}</p>`;
                return;
            }
            if (!response.ok) {
                throw new Error(`Search request failed: ${response.status} ${response.statusText}`);
            }

            const data = await response.json();
            if (data.results && data.results.length > 0) {
                data.results.sort((a, b) => b[0] - a[0]);
                let html = '<h3>Results:</h3>';
                data.results.forEach(item => {
                    const score = item[0];
                    const text = item[1];
                    const escapedText = text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
                    html += `
                        <div class="result-item">
                            <p><strong>Score:</strong> ${score}</p>
                            <p>${escapedText}</p>
                        </div>
                    `;
                });
                resultsDiv.innerHTML = html;
            } else {
                resultsDiv.innerHTML = '<p class="message info">No results found for your query.</p>';
            }
        } catch (error) {
            loader.style.display = 'none';
            resultsDiv.innerHTML = `<p class="message error">An error occurred: ${error.message}</p>`;
            console.error('Search error:', error);
        }
    });

    fetchStatus();
    setInterval(fetchStatus, 10000); // Check status every 10 seconds
});