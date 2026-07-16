/**
 * Private analytics configuration (not linked from the public site).
 *
 * Setup:
 * 1. Create a free account at https://www.goatcounter.com
 * 2. Add site URL: https://hongsungRae.github.io/
 * 3. Copy the count endpoint (Settings → Sites → Code)
 * 4. Create an API key (top menu → API) with read access
 * 5. Fill goatcounterEndpoint and apiToken below
 *
 * Panel password: change panelPasswordHash after running:
 *   echo -n 'YOUR_PASSWORD' | sha256sum
 */
window.SITE_ANALYTICS = {
    goatcounterEndpoint: '',
    apiToken: '',
    panelPasswordHash: '0f4a4fe4ab1a443c32a9c01311007142d96f5d9cde3d3bf57a9b8b3e6997090f'
};
