(function () {
    const SESSION_KEY = 'sr_panel_unlocked';
    const config = window.SITE_ANALYTICS || {};

    const gateView = document.getElementById('gate-view');
    const dashboardView = document.getElementById('dashboard-view');
    const gateForm = document.getElementById('gate-form');
    const gatePassword = document.getElementById('gate-password');
    const gateError = document.getElementById('gate-error');
    const gateCode = document.querySelector('.gate-code');
    const setupBanner = document.getElementById('setup-banner');
    const rangeSelect = document.getElementById('range-select');
    const refreshBtn = document.getElementById('refresh-btn');
    const logoutBtn = document.getElementById('logout-btn');
    const statusLine = document.getElementById('status-line');

    const metricPageviews = document.getElementById('metric-pageviews');
    const metricVisitors = document.getElementById('metric-visitors');
    const metricReferrers = document.getElementById('metric-referrers');
    const metricCountries = document.getElementById('metric-countries');
    const referrersBody = document.getElementById('referrers-body');
    const countriesBody = document.getElementById('countries-body');
    const pagesBody = document.getElementById('pages-body');

    const refTypeLabels = {
        h: 'HTTP',
        g: 'Generated',
        c: 'Campaign',
        o: 'Other'
    };

    function isConfigured() {
        return Boolean(config.goatcounterEndpoint && config.apiToken);
    }

    function getApiBase() {
        return config.goatcounterEndpoint.replace(/\/count\/?$/, '');
    }

    async function sha256(text) {
        const data = new TextEncoder().encode(text);
        const hash = await crypto.subtle.digest('SHA-256', data);
        return Array.from(new Uint8Array(hash))
            .map((byte) => byte.toString(16).padStart(2, '0'))
            .join('');
    }

    function formatDate(date) {
        return date.toISOString().slice(0, 10);
    }

    function getDateRange() {
        const value = rangeSelect.value;
        const end = new Date();
        const start = new Date(end);

        if (value === 'all') {
            start.setFullYear(2020, 0, 1);
        } else {
            start.setDate(end.getDate() - Number(value));
        }

        return {
            start: formatDate(start),
            end: formatDate(end)
        };
    }

    async function apiFetch(path, params) {
        const url = new URL(`${getApiBase()}/api/v0${path}`);
        Object.entries(params).forEach(([key, val]) => url.searchParams.set(key, val));

        const response = await fetch(url.toString(), {
            headers: {
                Authorization: `Bearer ${config.apiToken}`,
                'Content-Type': 'application/json'
            }
        });

        if (!response.ok) {
            const message = await response.text();
            throw new Error(message || `Request failed (${response.status})`);
        }

        return response.json();
    }

    function renderRows(tbody, rows, columns, emptyText) {
        tbody.innerHTML = '';

        if (!rows.length) {
            const row = document.createElement('tr');
            const cell = document.createElement('td');
            cell.colSpan = columns;
            cell.className = 'empty';
            cell.textContent = emptyText;
            row.appendChild(cell);
            tbody.appendChild(row);
            return;
        }

        rows.forEach((cells) => {
            const row = document.createElement('tr');
            cells.forEach((value) => {
                const cell = document.createElement('td');
                cell.textContent = value;
                row.appendChild(cell);
            });
            tbody.appendChild(row);
        });
    }

    function sumCounts(items) {
        return (items || []).reduce((total, item) => total + (item.count || 0), 0);
    }

    async function loadDashboard() {
        if (!isConfigured()) {
            setupBanner.hidden = false;
            statusLine.textContent = 'Configure GoatCounter to start collecting stats.';
            return;
        }

        setupBanner.hidden = true;
        statusLine.textContent = 'Loading…';

        const range = getDateRange();
        const params = {
            start: range.start,
            end: range.end
        };

        try {
            const [total, referrers, locations, hits] = await Promise.all([
                apiFetch('/stats/total', params),
                apiFetch('/stats/toprefs', params),
                apiFetch('/stats/locations', params),
                apiFetch('/stats/hits', params)
            ]);

            const refStats = referrers.stats || [];
            const locationStats = locations.stats || [];
            const hitStats = hits.hits || [];

            const pageviews = hitStats.reduce((sum, hit) => sum + (hit.count || 0), 0);
            const visitors = total.total || total.total_utc || sumCounts(refStats);

            metricPageviews.textContent = pageviews.toLocaleString();
            metricVisitors.textContent = visitors.toLocaleString();
            metricReferrers.textContent = refStats.length.toLocaleString();
            metricCountries.textContent = locationStats.length.toLocaleString();

            renderRows(
                referrersBody,
                refStats
                    .sort((a, b) => b.count - a.count)
                    .slice(0, 20)
                    .map((item) => [
                        item.name || '(direct / unknown)',
                        refTypeLabels[item.ref_scheme] || 'Other',
                        item.count.toLocaleString()
                    ]),
                3,
                'No referrer data for this period.'
            );

            renderRows(
                countriesBody,
                locationStats
                    .sort((a, b) => b.count - a.count)
                    .slice(0, 20)
                    .map((item) => [
                        item.name || 'Unknown',
                        item.count.toLocaleString()
                    ]),
                2,
                'No country data for this period.'
            );

            renderRows(
                pagesBody,
                hitStats
                    .sort((a, b) => b.count - a.count)
                    .slice(0, 15)
                    .map((item) => [
                        item.path || '/',
                        (item.count || 0).toLocaleString(),
                        (item.count || 0).toLocaleString()
                    ]),
                3,
                'No page data for this period.'
            );

            statusLine.textContent = `Updated ${new Date().toLocaleString()} · ${range.start} to ${range.end}`;
        } catch (error) {
            statusLine.textContent = 'Failed to load analytics. Check API token and GoatCounter settings.';
            console.error(error);
        }
    }

    function showDashboard() {
        gateView.hidden = true;
        dashboardView.hidden = false;
        loadDashboard();
    }

    function lockDashboard() {
        sessionStorage.removeItem(SESSION_KEY);
        dashboardView.hidden = true;
        gateView.hidden = false;
        gateForm.hidden = true;
        gatePassword.value = '';
        gateError.hidden = true;
    }

    async function unlockWithPassword(password) {
        const hash = await sha256(password);
        if (hash !== config.panelPasswordHash) {
            gateError.hidden = false;
            return false;
        }

        sessionStorage.setItem(SESSION_KEY, '1');
        gateError.hidden = true;
        showDashboard();
        return true;
    }

    let gateClicks = 0;
    let gateTimer = null;

    gateCode.addEventListener('click', () => {
        gateClicks += 1;
        window.clearTimeout(gateTimer);
        gateTimer = window.setTimeout(() => {
            gateClicks = 0;
        }, 1200);

        if (gateClicks >= 3) {
            gateForm.hidden = false;
            gatePassword.focus();
            gateClicks = 0;
        }
    });

    gateForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        await unlockWithPassword(gatePassword.value.trim());
    });

    rangeSelect.addEventListener('change', loadDashboard);
    refreshBtn.addEventListener('click', loadDashboard);
    logoutBtn.addEventListener('click', lockDashboard);

    if (sessionStorage.getItem(SESSION_KEY) === '1') {
        showDashboard();
    }
})();
