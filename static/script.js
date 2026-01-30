// Streamly Frontend Logic
document.addEventListener('DOMContentLoaded', () => {
    // State Management
    let currentUserId = null;
    let selectedRegion = 'Global';
    let users = [];

    // DOM Elements
    const userSelect = document.getElementById('user-select');
    const regionSelect = document.getElementById('region-select');
    const searchInput = document.getElementById('search-input');
    const clearSearchBtn = document.getElementById('clear-search');

    const currentUserNameSpan = document.getElementById('current-user-name');

    const personalizedSection = document.getElementById('personalized-section');
    const searchSection = document.getElementById('search-section');

    const personalizedGrid = document.getElementById('personalized-grid');
    const trendingGrid = document.getElementById('trending-grid');
    const popularGrid = document.getElementById('popular-grid');
    const searchGrid = document.getElementById('search-grid');

    // Modal Elements
    const modal = document.getElementById('movie-modal');
    const closeModal = document.querySelector('.close-modal');
    const modalTitle = document.getElementById('modal-title');
    const modalYear = document.getElementById('modal-year');
    const modalGenre = document.getElementById('modal-genre');
    const modalScore = document.getElementById('modal-score');
    const modalDesc = document.getElementById('modal-desc');
    const modalExplanation = document.getElementById('modal-explanation');
    const modalPoster = document.getElementById('modal-poster');
    const similarGrid = document.getElementById('similar-grid');

    // --- API HELPER FUNCTIONS ---

    async function fetchData(endpoint) {
        try {
            const response = await fetch(endpoint);
            if (!response.ok) throw new Error(`API Error: ${response.status}`);
            return await response.json();
        } catch (error) {
            console.error(error);
            showNotification('Failed to fetch data from server');
            return null;
        }
    }

    // --- UI RENDER FUNCTIONS ---

    function createMovieCard(movie, isRecommendation = false) {
        const card = document.createElement('div');
        card.className = 'movie-card';

        const posterUrl = movie.poster_url || `https://picsum.photos/seed/${movie.movie_id}/300/450`;

        card.innerHTML = `
            <div class="movie-poster" style="background-image: url('${posterUrl}')"></div>
            <div class="movie-info">
                <h3>${movie.title}</h3>
                <div class="movie-meta">
                    <span>${movie.release_year}</span>
                    <span class="score-tag">${isRecommendation ? (movie.hybrid_score * 10).toFixed(1) : '8.5'} ★</span>
                </div>
            </div>
        `;

        card.addEventListener('click', () => openMovieModal(movie));
        return card;
    }

    async function openMovieModal(movie) {
        modalTitle.textContent = movie.title;
        modalYear.textContent = movie.release_year;
        modalGenre.textContent = movie.genre;
        modalScore.textContent = movie.hybrid_score ? `Match: ${(movie.hybrid_score * 100).toFixed(0)}%` : 'Trending';
        modalDesc.textContent = `An exciting ${movie.genre.split(',')[0]} film that takes you on an unforgettable journey. Experience state-of-the-art storytelling in ${movie.language}.`;

        const posterUrl = movie.poster_url || `https://picsum.photos/seed/${movie.movie_id}/300/450`;
        modalPoster.style.backgroundImage = `url('${posterUrl}')`;

        if (movie.explanation) {
            modalExplanation.textContent = movie.explanation;
        } else {
            modalExplanation.textContent = "Recommended based on overall platform popularity and trending metrics.";
        }

        // Fetch similar movies
        similarGrid.innerHTML = '<div class="loader"></div>';
        const similar = await fetchData(`/api/movie/${movie.movie_id}/similar?n=3`);

        similarGrid.innerHTML = '';
        if (similar && similar.length > 0) {
            similar.forEach(sim => {
                const simCard = document.createElement('div');
                simCard.className = 'similar-card';
                simCard.textContent = sim.title;
                similarGrid.appendChild(simCard);
            });
        }

        modal.style.display = 'block';
    }

    function showNotification(message) {
        const notification = document.getElementById('notification');
        notification.textContent = message;
        notification.classList.add('show');
        setTimeout(() => notification.classList.remove('show'), 3000);
    }

    // --- CORE LOGIC ---

    // 1. Initial Load: Trending and Popular
    async function loadGlobalContent() {
        const trending = await fetchData('/api/trending?n=5');
        const popularEndpoint = selectedRegion === 'Global'
            ? '/api/popular?n=5'
            : `/api/popular/region/${selectedRegion}?n=5`;

        const popular = await fetchData(popularEndpoint);

        if (trending) {
            trendingGrid.innerHTML = '';
            trending.forEach(movie => trendingGrid.appendChild(createMovieCard(movie)));
        }

        if (popular) {
            popularGrid.innerHTML = '';
            const title = document.querySelector('#popular-grid').previousElementSibling.querySelector('h2');
            title.textContent = selectedRegion === 'Global' ? 'All-Time Popular' : `Popular in ${selectedRegion}`;
            popular.forEach(movie => popularGrid.appendChild(createMovieCard(movie)));
        }
    }

    // 2. Initial Load: Users
    async function loadUsers() {
        const userData = await fetchData('/api/users');
        if (userData) {
            users = userData;
            userSelect.innerHTML = '<option value="" disabled selected>Select a profile</option>';
            users.forEach(user => {
                const option = document.createElement('option');
                option.value = user.user_id;
                option.textContent = `User ${user.user_id} (${user.location})`;
                userSelect.appendChild(option);
            });
        }
    }

    // 3. Handle User Change
    userSelect.addEventListener('change', async (e) => {
        const userId = parseInt(e.target.value);
        currentUserId = userId;

        const user = users.find(u => u.user_id === userId);
        currentUserNameSpan.textContent = `User ${userId}`;

        // Auto-switch region to user's region
        if (user && user.location) {
            regionSelect.value = user.location;
            selectedRegion = user.location;
            loadGlobalContent();
        }

        showNotification(`Welcome back, User ${userId}!`);

        // Fetch recommendations
        personalizedGrid.innerHTML = '<div class="loader"></div>';
        personalizedSection.style.display = 'block';

        const response = await fetchData(`/api/recommend/${userId}?n=5`);

        personalizedGrid.innerHTML = '';
        if (response && response.recommendations) {
            response.recommendations.forEach(movie => {
                personalizedGrid.appendChild(createMovieCard(movie, true));
            });

            // Highlight the hero section update
            const topRec = response.recommendations[0];
            document.getElementById('hero-title').textContent = `Recommended: ${topRec.title}`;
            document.getElementById('hero-desc').textContent = `${topRec.explanation}. Dive into this ${topRec.genre} masterpiece now.`;
        }
    });

    // 4. Handle Region Change
    regionSelect.addEventListener('change', (e) => {
        selectedRegion = e.target.value;
        showNotification(`Switching region to ${selectedRegion}`);
        loadGlobalContent();
    });

    // 5. Handle Search
    let searchTimeout;
    searchInput.addEventListener('input', (e) => {
        clearTimeout(searchTimeout);
        const query = e.target.value.trim();

        if (query.length < 2) {
            if (query.length === 0) {
                searchSection.style.display = 'none';
            }
            return;
        }

        searchTimeout = setTimeout(async () => {
            searchGrid.innerHTML = '<div class="loader"></div>';
            searchSection.style.display = 'block';
            document.getElementById('search-query-title').textContent = `Results for "${query}"`;

            const results = await fetchData(`/api/search?q=${encodeURIComponent(query)}&n=10`);

            searchGrid.innerHTML = '';
            if (results && results.length > 0) {
                results.forEach(movie => {
                    searchGrid.appendChild(createMovieCard(movie));
                });
            } else {
                searchGrid.innerHTML = '<p class="no-results">No movies found matching your search.</p>';
            }
        }, 500);
    });

    clearSearchBtn.addEventListener('click', () => {
        searchInput.value = '';
        searchSection.style.display = 'none';
    });

    // Close Modal
    closeModal.onclick = () => modal.style.display = 'none';
    window.onclick = (event) => {
        if (event.target == modal) modal.style.display = 'none';
    };

    // Initialize Page
    loadGlobalContent();
    loadUsers();
});
