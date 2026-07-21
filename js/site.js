(function () {
    const focusPhrases = [
        'Medical AI',
        'Multiple Instance Learning',
        'Histopathology',
        'Uncertainty Calibration'
    ];

    const nav = document.getElementById('site-nav');
    const menuToggle = document.getElementById('menu-toggle');
    const backToTop = document.getElementById('back-to-top');
    const focusEl = document.getElementById('rotating-focus');
    const navLinks = nav ? Array.from(nav.querySelectorAll('a[href^="#"]')) : [];
    const sections = Array.from(document.querySelectorAll('main section[id], footer[id]'));
    const revealEls = Array.from(document.querySelectorAll('.reveal'));
    const filterButtons = Array.from(document.querySelectorAll('.venue-switch button'));
    const paperEntries = Array.from(document.querySelectorAll('.paper-entry'));

    function setActiveNav() {
        const offset = window.scrollY + 120;
        let current = sections[0]?.id || '';

        sections.forEach((section) => {
            if (section.offsetTop <= offset) {
                current = section.id;
            }
        });

        navLinks.forEach((link) => {
            link.classList.toggle('is-active', link.getAttribute('href') === `#${current}`);
        });
    }

    function toggleBackToTop() {
        if (!backToTop) return;
        backToTop.classList.toggle('is-visible', window.scrollY > 420);
    }

    if (menuToggle && nav) {
        menuToggle.addEventListener('click', () => {
            const isOpen = nav.classList.toggle('is-open');
            menuToggle.setAttribute('aria-expanded', String(isOpen));
        });

        navLinks.forEach((link) => {
            link.addEventListener('click', () => {
                nav.classList.remove('is-open');
                menuToggle.setAttribute('aria-expanded', 'false');
            });
        });
    }

    if (backToTop) {
        backToTop.addEventListener('click', () => {
            window.scrollTo({ top: 0, behavior: 'smooth' });
        });
    }

    document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
        anchor.addEventListener('click', (event) => {
            const target = document.querySelector(anchor.getAttribute('href'));
            if (!target) return;
            event.preventDefault();
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        });
    });

    if ('IntersectionObserver' in window) {
        // Use threshold 0 so tall sections (e.g. Publications on mobile)
        // still become visible when any part enters the viewport.
        const revealObserver = new IntersectionObserver((entries) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('is-visible');
                    revealObserver.unobserve(entry.target);
                }
            });
        }, { threshold: 0, rootMargin: '0px 0px -8% 0px' });

        revealEls.forEach((el) => revealObserver.observe(el));

        // Fallback: if a section is already on screen (or taller than the
        // viewport), force it visible after layout.
        window.requestAnimationFrame(() => {
            revealEls.forEach((el) => {
                const rect = el.getBoundingClientRect();
                const inView = rect.top < window.innerHeight && rect.bottom > 0;
                if (inView) el.classList.add('is-visible');
            });
        });
    } else {
        revealEls.forEach((el) => el.classList.add('is-visible'));
    }

    if (focusEl) {
        let phraseIndex = 0;
        window.setInterval(() => {
            phraseIndex = (phraseIndex + 1) % focusPhrases.length;
            focusEl.textContent = focusPhrases[phraseIndex];
        }, 3200);
    }

    filterButtons.forEach((button) => {
        button.addEventListener('click', () => {
            const filter = button.dataset.filter;
            filterButtons.forEach((btn) => btn.classList.toggle('is-active', btn === button));

            paperEntries.forEach((entry) => {
                const visible = filter === 'all' || entry.dataset.venue === filter;
                entry.style.display = visible ? '' : 'none';
            });
        });
    });

    let ticking = false;
    window.addEventListener('scroll', () => {
        if (ticking) return;
        ticking = true;
        window.requestAnimationFrame(() => {
            setActiveNav();
            toggleBackToTop();
            ticking = false;
        });
    });

    setActiveNav();
    toggleBackToTop();

    const secretStatsLink = document.getElementById('secret-stats-link');
    const secretStatsTrigger = document.getElementById('panel-photo-trigger');
    const SECRET_STATS_SESSION = 'sr_stats_entry_visible';
    const SECRET_CLICK_WINDOW_MS = 2000;

    if (secretStatsLink && secretStatsTrigger) {
        if (sessionStorage.getItem(SECRET_STATS_SESSION) === '1') {
            secretStatsLink.hidden = false;
        }

        let secretClicks = 0;
        let secretTimer = null;

        secretStatsTrigger.addEventListener('click', () => {
            secretClicks += 1;
            window.clearTimeout(secretTimer);
            secretTimer = window.setTimeout(() => {
                secretClicks = 0;
            }, SECRET_CLICK_WINDOW_MS);

            if (secretClicks >= 3) {
                secretClicks = 0;
                secretStatsLink.hidden = false;
                sessionStorage.setItem(SECRET_STATS_SESSION, '1');
            }
        });
    }
})();
