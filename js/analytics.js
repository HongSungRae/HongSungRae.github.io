(function () {
    const config = window.SITE_ANALYTICS;
    if (!config || !config.goatcounterEndpoint) return;

    const script = document.createElement('script');
    script.async = true;
    script.src = '//gc.zgo.at/count.js';
    script.setAttribute('data-goatcounter', config.goatcounterEndpoint);
    document.head.appendChild(script);
})();
