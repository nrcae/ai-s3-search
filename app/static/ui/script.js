document.addEventListener('DOMContentLoaded', () => {
    const searchForm = document.getElementById('searchForm');
    const resultsDiv = document.getElementById('results');
    const statusDiv = document.getElementById('status');
    const loader = document.getElementById('loader');
    const currentYearSpan = document.getElementById('currentYear');

    // Set the current year in the footer, if the element exists
    if (currentYearSpan) {
        currentYearSpan.textContent = new Date().getFullYear();
    }

    async function fetchStatus() {
        try {
            const response = await fetch('/status');
            if (!response.ok) {
                // Handle non-successful HTTP responses (e.g., 404, 500)
                throw new Error(`Status check failed: ${response.status} ${response.statusText}`);
            }
            const data = await response.json();
            // Display current indexing status and document count
            statusDiv.textContent = `Indexing: ${data.index_ready ? 'Ready' : 'Indexing...'} | Documents: ${data.index_size || 0}`;

            // Update status bar class for visual feedback based on index readiness
            statusDiv.classList.remove('ready', 'not-ready', 'error');
            if (data.index_ready) {
                statusDiv.classList.add('ready');
            } else {
                statusDiv.classList.add('not-ready');
            }

        } catch (error) {
            // Handle errors during status fetch (e.g., network issues, server errors)
            statusDiv.textContent = 'Status: Error fetching status.';
            statusDiv.classList.remove('ready', 'not-ready');
            statusDiv.classList.add('error');
            console.error('Error fetching status:', error);
        }
    }

    // Handle search form submissions
    searchForm.addEventListener('submit', async function(event) {
        event.preventDefault(); // Prevent default form submission (which would cause a page reload)

        // Clear previous results and display the loading indicator
        resultsDiv.innerHTML = '';
        loader.style.display = 'block';

        try {
            const topK = document.querySelector('#topK').value || '5'; // Get Top-K value, default to 5 if empty
            const currentQuery = document.getElementById('query').value.trim(); // Get and trim search query

            // If the query is empty, clear results, hide loader, and do nothing further
            if (!currentQuery) {
                resultsDiv.innerHTML = '';
                loader.style.display = 'none';
                return;
            }

            const response = await fetch(`/search?q=${encodeURIComponent(currentQuery)}&top_k=${topK}`);
            // Hide loader once the initial response is received
            loader.style.display = 'none';

            // Handle specific HTTP error status codes from the search endpoint
            if (response.status === 503) { // Service Unavailable (e.g., search index not ready)
                resultsDiv.innerHTML = '<p class="message warning">Search index is not ready. Please try again later.</p>';
                return;
            }
            if (response.status === 400) { // Bad Request (e.g., invalid query parameters)
                const errorData = await response.json(); // Attempt to get detailed error message from response body
                resultsDiv.innerHTML = `<p class="message error">Error: ${errorData.detail || 'Invalid query'}</p>`;
                return;
            }
            if (!response.ok) { // Handle other non-successful HTTP responses
                throw new Error(`Search request failed: ${response.status} ${response.statusText}`);
            }

            const data = await response.json();
            if (data.results && data.results.length > 0) {
                // Sort results by score in descending order (highest score first)
                // Assumes 'item[0]' is the score.
                data.results.sort((a, b) => b[0] - a[0]);

                let html = '<h3>Results:</h3>';
                data.results.forEach(item => {
                    const score = item[0]; // Assumes score is the first element in the result item array
                    const text = item[1];  // Assumes text is the second element
                    const escapedText = text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
                    // Ensure this line includes the title attribute:
                    html += `
                        <div class="result-item">
                            <p><strong title="Relevance score (0-1 scale): Higher score indicates better match.">Score:</strong> ${score.toFixed(4)}</p>
                            <p>${escapedText}</p>
                        </div>
                    `;
                });
                resultsDiv.innerHTML = html; // Display the formatted results
            } else {
                // Inform user if no results were found for their query
                const currentQueryValue = document.getElementById('query').value.trim();
                // Basic escape function for HTML display (reuse or define if not present)
                const escapeHtml = (unsafe) => 
                    unsafe.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
                resultsDiv.innerHTML = `<p class="message info">No results found for "<em>${escapeHtml(currentQueryValue)}</em>".<br>Try rephrasing your query or using different keywords.</p>`;
            }
        } catch (error) {
            // Catch-all for network errors or issues during search processing
            loader.style.display = 'none'; // Ensure loader is hidden on error
            resultsDiv.innerHTML = `<p class="message error">An error occurred: ${error.message}</p>`;
            console.error('Search error:', error);
        }
    });

    fetchStatus();
    setInterval(fetchStatus, 1000); // Check status every second
});